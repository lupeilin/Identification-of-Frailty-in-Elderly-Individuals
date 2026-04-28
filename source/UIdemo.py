import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import threading
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QPushButton, QTextEdit, QLabel,
    QStatusBar, QMessageBox
)
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtGui import QFont


# ================= 1. 全局加载 BERT 模型 =================
MODEL_PATH = "./frailty_bert_demo_model"   # 请确保路径正确

print("正在加载衰弱识别模型，请稍候...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✅ 模型加载完毕，运行于 {device}")


# ================= 2. 核心预测函数（本地 BERT）=================
def predict_frailty(text):
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


# ================= 3. RAG 解释生成（调用本地 Ollama）=================
def generate_explanation(text, bert_pred, bert_conf):
    """
    使用本地大模型（Ollama）生成详细的临床解释。
    如果 Ollama 未启动或调用失败，返回错误提示。
    """
    knowledge = """
Fried衰弱表型标准（5项，满足≥3项为衰弱）：
1. 不明原因体重下降：过去1年内体重下降≥5%或≥4.5kg
2. 疲乏/精力减退：自我报告“疲乏无力”、“整天没精神”等
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

请严格按以下格式回答：
1. 逐项核对Fried五项指标（标明“阳性”或“阴性”，并给出病历中的证据）
2. 阳性指标总数
3. 最终评估结论（衰弱/衰弱前期/非衰弱）
4. 临床建议（如营养、运动、药物等多学科干预建议）
"""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5:3b",   # 改成你下载的模型名
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"❌ Ollama 返回错误，状态码：{response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 Ollama，请确保已启动 ollama serve 并下载了相应模型。"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"


# ================= 4. 后台线程处理 RAG（避免界面冻结）=================
class ExplanationThread(QThread):
    finished = Signal(str)

    def __init__(self, text, pred, conf):
        super().__init__()
        self.text = text
        self.pred = pred
        self.conf = conf

    def run(self):
        explanation = generate_explanation(self.text, self.pred, self.conf)
        self.finished.emit(explanation)


# ================= 5. 主窗口 =================
class FrailtyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("老年衰弱智能评估系统")
        self.setGeometry(100, 100, 800, 650)

        # 全局字体
        font = QFont("Microsoft YaHei", 10)
        QApplication.setFont(font)

        # 核心控件
        self.text_input = QTextEdit()
        self.text_input.setPlaceholderText("请在此处粘贴脱敏后的电子病历文本...")

        self.btn_quick_predict = QPushButton("快速预测")
        self.btn_explain_predict = QPushButton("详细解释 (RAG)")
        self.btn_clear = QPushButton("清空")
        self.btn_example = QPushButton("填入示例")

        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)

        # 状态栏
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("系统就绪")

        # 界面布局
        self._setup_ui()

        # 事件绑定
        self.btn_quick_predict.clicked.connect(self.on_quick_predict)
        self.btn_explain_predict.clicked.connect(self.on_explain_predict)
        self.btn_clear.clicked.connect(lambda: self.text_input.clear())
        self.btn_example.clicked.connect(self.fill_example)

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # 标题
        title_label = QLabel("老年衰弱自动识别系统")
        title_font = QFont("Microsoft YaHei", 16, QFont.Bold)
        title_label.setFont(title_font)
        main_layout.addWidget(title_label)

        main_layout.addWidget(self.text_input, 2)  # 拉伸因子2

        # 按钮行
        button_layout = QHBoxLayout()
        self.btn_quick_predict.setStyleSheet("background-color: #4CAF50; color: white;")
        self.btn_explain_predict.setStyleSheet("background-color: #FF9800; color: white;")
        self.btn_clear.setStyleSheet("background-color: #f44336; color: white;")
        self.btn_example.setStyleSheet("background-color: #2196F3; color: white;")

        button_layout.addWidget(self.btn_quick_predict)
        button_layout.addWidget(self.btn_explain_predict)
        button_layout.addWidget(self.btn_clear)
        button_layout.addWidget(self.btn_example)
        main_layout.addLayout(button_layout)

        # 评估结果标签
        result_label = QLabel("评估结果：")
        result_label.setFont(QFont("Microsoft YaHei", 11, QFont.Bold))
        main_layout.addWidget(result_label)
        main_layout.addWidget(self.result_display, 3)

    # ---------- 槽函数 ----------
    @Slot()
    def on_quick_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return

        self.status_bar.showMessage("正在进行快速预测...")
        result = predict_frailty(text)
        output = f"预测结论：{result['prediction']}\n置信度：{result['confidence']:.4f}"
        self.result_display.setPlainText(output)
        self.status_bar.showMessage("预测完成", 3000)

    @Slot()
    def on_explain_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return

        # 先获取 BERT 预测结果
        bert_result = predict_frailty(text)
        pred = bert_result["prediction"]
        conf = bert_result["confidence"]

        # 禁用按钮，启动后台线程
        self.btn_explain_predict.setEnabled(False)
        self.status_bar.showMessage("正在调用本地大模型进行详细分析，请稍候...")
        self.result_display.setPlainText("等待大模型分析中...")

        # 启动后台解释线程
        self.explain_thread = ExplanationThread(text, pred, conf)
        self.explain_thread.finished.connect(self.on_explanation_ready)
        self.explain_thread.finished.connect(lambda: self.btn_explain_predict.setEnabled(True))
        self.explain_thread.start()

    @Slot(str)
    def on_explanation_ready(self, explanation):
        self.result_display.setPlainText(explanation)
        self.status_bar.showMessage("详细解释已生成", 5000)

    def fill_example(self):
        example = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
        self.text_input.setPlainText(example)


# ================= 6. 启动应用 =================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 模型已在全局加载，此处无需重复
    window = FrailtyMainWindow()
    window.show()
    sys.exit(app.exec())