import os
import sys
import json
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox,
    QTextEdit, QPushButton, QStatusBar, QFileDialog, QTabWidget  # 添加 QFileDialog
)
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QFont, QAction

# ================= 1. 模型加载函数（不依赖 QApplication）=================
MODEL_PATH = "./frailty_bert_demo_model"


def load_bert_model():
    """加载 BERT 模型，失败时打印错误并退出（不弹窗）"""
    if not os.path.exists(MODEL_PATH):
        print(f"错误：未找到 BERT 模型目录 {MODEL_PATH}")
        sys.exit(1)
    try:
        print("正在加载衰弱识别模型，请稍候...")
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ 模型加载完毕，运行于 {device}")
        return tokenizer, model, device
    except Exception as e:
        print(f"错误：加载 BERT 模型时出错：{str(e)}")
        sys.exit(1)


# ================= 2. 核心预测函数 =================
def predict_frailty(text, tokenizer, model, device):
    """返回预测类别和置信度"""
    if not text or not text.strip():
        return {"prediction": "未知", "confidence": 0.0}
    inputs = tokenizer(
        text, padding="max_length", truncation=True,
        max_length=128, return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = probs.argmax().item()
        conf = probs[0][pred].item()
    return {
        "prediction": "衰弱" if pred == 1 else "非衰弱",
        "confidence": conf
    }


# ================= 3. RAG 解释生成（JSON 模式）=================
EXPECTED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "fried_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "result": {"type": "string", "enum": ["阳性", "阴性"]},
                    "evidence": {"type": "string"}
                },
                "required": ["name", "result", "evidence"]
            }
        },
        "positive_count": {"type": "integer"},
        "conclusion": {"type": "string", "enum": ["衰弱", "衰弱前期", "非衰弱"]},
        "suggestions": {"type": "string"}
    },
    "required": ["fried_items", "positive_count", "conclusion", "suggestions"]
}


def generate_explanation(text, bert_pred, bert_conf):
    knowledge = """
Fried衰弱表型标准（5项，满足≥3项为衰弱）：
1. 不明原因体重下降：过去1年内体重下降≥5%或≥4.5kg
2. 疲乏/精力减退：自我报告"疲乏无力"、"整天没精神"等
3. 握力下降：用手握力计测量，根据性别和BMI调整阈值
4. 步速减慢：行走4.57米时间延长（根据身高、性别调整）
5. 活动量减少：每周消耗热量低于标准
"""
    prompt = f"""你是一位老年医学专家。请根据以下医学知识，对患者进行衰弱评估。

医学知识：{knowledge}

【辅助模型初步判断】
预测结果：{bert_pred}（置信度：{bert_conf:.2%}）

【患者信息】
{text}

请严格输出一个 JSON 对象，其格式必须完全符合下面的 Schema：
{json.dumps(EXPECTED_JSON_SCHEMA, ensure_ascii=False, indent=2)}

要求：
- fried_items 数组包含5个元素，顺序对应上述5项标准。
- 每项需给出 result（"阳性"/"阴性"）和 evidence（病历中的依据，若无明确依据可写"未提及"）。
- positive_count 为阳性的条目数。
- conclusion 根据阳性计数判断：≥3项为"衰弱"，1-2项为"衰弱前期"，0项为"非衰弱"。
- suggestions 提供营养、运动、药物等多学科干预建议。
只输出纯 JSON，不要有任何额外文本或注释。
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        if response.status_code == 200:
            raw_json = response.json()["response"]
            try:
                data = json.loads(raw_json)
                if all(k in data for k in ["fried_items", "positive_count", "conclusion", "suggestions"]):
                    return data
                else:
                    return {"error": "LLM 返回的 JSON 缺少必要字段", "raw": raw_json}
            except json.JSONDecodeError:
                return {"error": "LLM 返回的不是有效 JSON", "raw": raw_json}
        else:
            return {"error": f"Ollama 返回错误，状态码：{response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到 Ollama，请确保已启动 ollama serve 并下载了相应模型。"}
    except Exception as e:
        return {"error": f"请求异常：{str(e)}"}


def format_explanation_html(data):
    if "error" in data:
        return f"<p style='color:red;'>❌ 生成解释失败：{data['error']}</p><pre>{data.get('raw', '')}</pre>"

    html = "<h3>📋 衰弱评估详细报告</h3>"
    html += "<h4>1. Fried 衰弱表型逐项核对</h4><ul>"
    for item in data.get("fried_items", []):
        name = item.get("name", "未知指标")
        result = item.get("result", "未知")
        evidence = item.get("evidence", "")
        color = "green" if result == "阴性" else "red"
        html += f"<li><b>{name}</b>：<span style='color:{color};'>{result}</span><br>📌 证据：{evidence}</li>"
    html += "</ul>"

    positive_count = data.get("positive_count", 0)
    html += f"<p><b>2. 阳性指标总数：</b>{positive_count} 项</p>"

    conclusion = data.get("conclusion", "未知")
    color_conclusion = "red" if conclusion == "衰弱" else ("orange" if conclusion == "衰弱前期" else "green")
    html += f"<p><b>3. 最终评估结论：</b><span style='color:{color_conclusion}; font-weight:bold;'>{conclusion}</span></p>"

    suggestions = data.get("suggestions", "")
    html += f"<h4>4. 临床建议</h4><p>{suggestions}</p>"
    return html


# ================= 4. 后台线程 =================
class ExplanationThread(QThread):
    finished = Signal(dict)

    def __init__(self, text, pred, conf):
        super().__init__()
        self.text = text
        self.pred = pred
        self.conf = conf

    def run(self):
        result = generate_explanation(self.text, self.pred, self.conf)
        self.finished.emit(result)


class OllamaHealthCheckThread(QThread):
    status_signal = Signal(bool, str)

    def run(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=3)
            if resp.status_code == 200:
                data = resp.json()
                models = [m["name"] for m in data.get("models", [])]
                if any("qwen2.5:3b" in m for m in models):
                    self.status_signal.emit(True, "✅ Ollama 服务正常，qwen2.5:3b 模型已就绪")
                else:
                    self.status_signal.emit(False,
                                            "⚠️ Ollama 服务正常，但未找到 qwen2.5:3b 模型，请运行 'ollama pull qwen2.5:3b'")
            else:
                self.status_signal.emit(False, f"❌ Ollama 服务异常，状态码：{resp.status_code}")
        except requests.exceptions.ConnectionError:
            self.status_signal.emit(False, "❌ 无法连接到 Ollama，请启动 ollama serve")
        except Exception as e:
            self.status_signal.emit(False, f"❌ 检测 Ollama 时出错：{str(e)}")


# ================= 5. 主窗口 =================

class FrailtyMainWindow(QMainWindow):
    def __init__(self, tokenizer, model, device):
        super().__init__()
        print("初始化主窗口...")
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

        # 加载新版 UI 文件
        ui_file = os.path.join(os.path.dirname(__file__), "tools/frailty_ui_v2.ui")
        if not os.path.exists(ui_file):
            QMessageBox.critical(self, "UI 文件缺失", f"未找到界面文件：{ui_file}")
            sys.exit(1)

        loader = QUiLoader()
        try:
            self.ui = loader.load(ui_file, None)
            if self.ui is None:
                raise RuntimeError("QUiLoader 返回 None")
        except Exception as e:
            QMessageBox.critical(self, "UI 加载失败", f"无法加载 UI 文件：{str(e)}")
            sys.exit(1)

        self.setCentralWidget(self.ui)

        # ----- 核心控件 -----
        self.text_input = self.ui.findChild(QTextEdit, "text_input")
        self.result_tabs = self.ui.findChild(QTabWidget, "result_tabs")
        self.result_display = self.ui.findChild(QTextEdit, "result_display")
        self.explain_display = self.ui.findChild(QTextEdit, "explain_display")

        # 输入区按钮（仅两个）
        self.btn_example = self.ui.findChild(QPushButton, "btn_example")
        self.btn_clear = self.ui.findChild(QPushButton, "btn_clear")

        # 菜单/工具栏动作
        self.action_load_case = self.ui.findChild(QAction, "action_load_case")
        self.action_batch_import = self.ui.findChild(QAction, "action_batch_import")
        self.action_exit = self.ui.findChild(QAction, "action_exit")
        self.action_model_info = self.ui.findChild(QAction, "action_model_info")
        self.action_ollama_status = self.ui.findChild(QAction, "action_ollama_status")
        self.action_settings = self.ui.findChild(QAction, "action_settings")
        self.action_about = self.ui.findChild(QAction, "action_about")
        self.action_usage_guide = self.ui.findChild(QAction, "action_usage_guide")
        self.action_quick_predict = self.ui.findChild(QAction, "action_quick_predict")
        self.action_explain_predict = self.ui.findChild(QAction, "action_explain_predict")
        self.action_clear = self.ui.findChild(QAction, "action_clear")

        self.status_bar = self.statusBar()
        self.status_bar.showMessage("系统就绪")

        # ----- 信号槽连接 -----
        # 按钮
        self.btn_example.clicked.connect(self.fill_example)
        self.btn_clear.clicked.connect(lambda: self.text_input.clear())
        # 菜单/工具栏动作
        self.action_quick_predict.triggered.connect(self.on_quick_predict)
        self.action_explain_predict.triggered.connect(self.on_explain_predict)
        self.action_clear.triggered.connect(lambda: self.text_input.clear())
        self.action_load_case.triggered.connect(self.load_case_from_file)
        self.action_exit.triggered.connect(self.close)

        # 占位功能
        def placeholder(title):
            QMessageBox.information(self, title, "该功能将在后续版本中提供。")
        self.action_batch_import.triggered.connect(lambda: placeholder("批量导入"))
        self.action_model_info.triggered.connect(lambda: placeholder("模型信息"))
        self.action_ollama_status.triggered.connect(lambda: placeholder("Ollama 状态"))
        self.action_settings.triggered.connect(lambda: placeholder("系统设置"))
        self.action_about.triggered.connect(lambda: placeholder("关于软件"))
        self.action_usage_guide.triggered.connect(lambda: placeholder("使用指南"))

        # 启动 Ollama 健康检查
        self.ollama_check_thread = OllamaHealthCheckThread()
        self.ollama_check_thread.status_signal.connect(self.on_ollama_status)
        self.ollama_check_thread.start()

        self.explain_thread = None
        print("主窗口初始化完成")

    @Slot(bool, str)
    def on_ollama_status(self, is_available, message):
        self.status_bar.showMessage(message, 5000)
        self.action_explain_predict.setEnabled(is_available)
        if not is_available:
            self.action_explain_predict.setToolTip(message)
            QMessageBox.warning(self, "服务不可用", message + "\n详细解释功能将被禁用。")
        else:
            self.action_explain_predict.setToolTip("")

    @Slot()
    def on_quick_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return
        self.status_bar.showMessage("正在进行快速预测...")
        result = predict_frailty(text, self.tokenizer, self.model, self.device)
        output = f"预测结论：{result['prediction']}\n置信度：{result['confidence']:.4f}"
        self.result_display.setPlainText(output)
        if self.result_tabs is not None:
            self.result_tabs.setCurrentIndex(0)
        self.status_bar.showMessage("预测完成", 3000)

    @Slot()
    def on_explain_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return
        bert_result = predict_frailty(text, self.tokenizer, self.model, self.device)
        if bert_result["prediction"] == "未知":
            QMessageBox.warning(self, "提示", "病历文本无效，无法生成详细解释。")
            return
        pred, conf = bert_result["prediction"], bert_result["confidence"]
        if self.explain_thread is not None and self.explain_thread.isRunning():
            QMessageBox.information(self, "提示", "已有分析任务进行中，请稍后再试。")
            return

        self.action_explain_predict.setEnabled(False)
        self.status_bar.showMessage("正在调用本地大模型进行详细分析，请稍候...")
        self.explain_display.setPlainText("等待大模型分析中...")
        if self.result_tabs is not None:
            self.result_tabs.setCurrentIndex(1)

        self.explain_thread = ExplanationThread(text, pred, conf)
        self.explain_thread.finished.connect(self.on_explanation_ready)
        self.explain_thread.finished.connect(lambda: self.action_explain_predict.setEnabled(True))
        self.explain_thread.start()

    @Slot(dict)
    def on_explanation_ready(self, explanation_dict):
        html_content = format_explanation_html(explanation_dict)
        self.explain_display.setHtml(html_content)
        if self.result_tabs is not None:
            self.result_tabs.setCurrentIndex(1)
        self.status_bar.showMessage("详细解释已生成", 5000)

    @Slot()
    def load_case_from_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择病历文件", "", "文本文件 (*.txt);;所有文件 (*)"
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_input.setPlainText(content)
                self.status_bar.showMessage(f"已加载：{os.path.basename(file_path)}", 3000)
            except Exception as e:
                QMessageBox.critical(self, "读取失败", f"无法读取文件：\n{str(e)}")

    def fill_example(self):
        example = """患者女性，78岁，退休教师。主诉：近1年体重下降8kg（原体重60kg），自觉"浑身没劲"，走路速度明显变慢，以前能轻松走完小区一圈，现在中途需要休息。家属反映患者近半年几乎不出门，日常家务也懒得做。既往史：高血压、骨关节炎。查体：步速0.6m/s，握力（右手）14kg（参考同龄女性正常值>18kg）。"""
        self.text_input.setPlainText(example)


# ================= 6. 启动 =================
if __name__ == "__main__":
    print("创建 QApplication...")
    app = QApplication(sys.argv)
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    print("加载 BERT 模型...")
    tokenizer, model, device = load_bert_model()

    print("创建主窗口...")
    window = FrailtyMainWindow(tokenizer, model, device)
    window.show()
    print("窗口已显示，进入事件循环...")
    sys.exit(app.exec())