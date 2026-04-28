import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
import threading
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox, QTextEdit, QPushButton, QStatusBar
from PySide6.QtCore import QThread, Signal, Slot
from PySide6.QtUiTools import QUiLoader
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


# ================= 2. 核心预测函数 =================
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


# ================= 3. RAG 解释生成 =================
def generate_explanation(text, bert_pred, bert_conf):
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
                "model": "qwen2.5:3b",   # 改成你实际下载的模型名
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


# ================= 4. 后台解释线程 =================
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


# ================= 5. 主窗口（从 UI 文件加载） =================
class FrailtyMainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # 加载 UI 文件
        loader = QUiLoader()
        self.ui = loader.load(".\\tools\\frailty_ui.ui", None)  # 确保 UI 文件在相同目录
        self.setCentralWidget(self.ui)

        # 获取控件引用
        self.text_input = self.ui.findChild(QTextEdit, "text_input")
        self.result_display = self.ui.findChild(QTextEdit, "result_display")
        self.btn_quick_predict = self.ui.findChild(QPushButton, "btn_quick_predict")
        self.btn_explain_predict = self.ui.findChild(QPushButton, "btn_explain_predict")
        self.btn_clear = self.ui.findChild(QPushButton, "btn_clear")
        self.btn_example = self.ui.findChild(QPushButton, "btn_example")
        self.status_bar = self.ui.findChild(QStatusBar, "statusbar")

        # 设置窗口标题和大小（UI 文件中已定义，可覆盖）
        self.setWindowTitle("老年衰弱智能评估系统")
        self.resize(800, 650)

        # 连接信号槽
        self.btn_quick_predict.clicked.connect(self.on_quick_predict)
        self.btn_explain_predict.clicked.connect(self.on_explain_predict)
        self.btn_clear.clicked.connect(lambda: self.text_input.clear())
        self.btn_example.clicked.connect(self.fill_example)

        # 初始状态栏信息
        if self.status_bar:
            self.status_bar.showMessage("系统就绪")

    @Slot()
    def on_quick_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return

        if self.status_bar:
            self.status_bar.showMessage("正在进行快速预测...")
        result = predict_frailty(text)
        output = f"预测结论：{result['prediction']}\n置信度：{result['confidence']:.4f}"
        self.result_display.setPlainText(output)
        if self.status_bar:
            self.status_bar.showMessage("预测完成", 3000)

    @Slot()
    def on_explain_predict(self):
        text = self.text_input.toPlainText().strip()
        if not text:
            QMessageBox.warning(self, "提示", "请输入病历文本！")
            return

        bert_result = predict_frailty(text)
        pred = bert_result["prediction"]
        conf = bert_result["confidence"]

        self.btn_explain_predict.setEnabled(False)
        if self.status_bar:
            self.status_bar.showMessage("正在调用本地大模型进行详细分析，请稍候...")
        self.result_display.setPlainText("等待大模型分析中...")

        self.explain_thread = ExplanationThread(text, pred, conf)
        self.explain_thread.finished.connect(self.on_explanation_ready)
        self.explain_thread.finished.connect(lambda: self.btn_explain_predict.setEnabled(True))
        self.explain_thread.start()

    @Slot(str)
    def on_explanation_ready(self, explanation):
        self.result_display.setPlainText(explanation)
        if self.status_bar:
            self.status_bar.showMessage("详细解释已生成", 5000)

    def fill_example(self):
        example = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
        self.text_input.setPlainText(example)


# ================= 6. 启动 =================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    # 设置全局字体
    font = QFont("Microsoft YaHei", 10)
    app.setFont(font)

    # 模型已在全局加载，此处直接创建窗口
    window = FrailtyMainWindow()
    window.show()
    sys.exit(app.exec())