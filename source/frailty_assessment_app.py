#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frailty_assessment_app.py
老年衰弱智能评估系统（带知识库查询）
用法：python frailty_assessment_app.py
前提：已运行 build_knowledge_base.py 构建知识库
"""

import os
import sys
import json
import requests
import torch
from transformers import BertTokenizer, BertForSequenceClassification

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QProgressDialog, QTableWidgetItem, QHeaderView,
    QTextEdit, QPushButton, QStatusBar, QFileDialog, QTabWidget, QTableWidget
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QDateTime
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QFont, QAction, QColor
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import shutil

# ================= 本地知识库模块 =================
chroma_client = chromadb.PersistentClient(path="./geriatric_knowledge_db")
collection = chroma_client.get_or_create_collection(name="geriatric_guidelines")


def load_embedding_model():
    """
    加载本地嵌入模型，优先使用中文医疗专用模型 Ganymede-Health-BERT，
    下载失败时自动回退到通用多语言模型。
    """
    model_repo = "BAAI/bge-large-zh-v1.5"          # 中文医疗 BERT（主模型）
    fallback_repo = "paraphrase-multilingual-MiniLM-L12-v2"   # 通用多语言备选

    # 模型存放目录（与 frailty_assessment_app.py 的 models 目录平行）
    models_root = "models"       # 模型根目录
    local_model_dir = models_root + "/BAAI/bge-large-zh-v1.5"

    config_path = os.path.join(local_model_dir, "config.json")
    # 如果本地已存在，直接加载
    if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
        print(f"✅ 从本地加载嵌入模型：{local_model_dir}")
        return SentenceTransformer(local_model_dir)

    # 下载并保存到本地
    print(f"⏳ 正在下载 {model_repo} ...（约 1.3 GB，首次需等待）")
    try:
        model = SentenceTransformer(model_repo)
        model.save(local_model_dir)
        print(f"✅ 模型已保存至 {local_model_dir}")
        return model
    except Exception as e:
        print(f"❌ 下载 {model_repo} 失败：{e}")
        print("尝试回退到通用模型...")
        fallback_dir = os.path.join(models_root, "paraphrase-multilingual-MiniLM-L12-v2")
        if not os.path.exists(fallback_dir):
            try:
                fallback_model = SentenceTransformer(fallback_repo)
                fallback_model.save(fallback_dir)
                return fallback_model
            except Exception as e2:
                print(f"❌ 回退模型下载也失败：{e2}")
                return SentenceTransformer(fallback_repo)  # 在线使用
        else:
            return SentenceTransformer(fallback_dir)



embedding_model = load_embedding_model()


def retrieve_relevant_guidelines(query, top_k=5, distance_threshold=0.5):
    """从知识库检索相关指南片段，支持余弦距离过滤并返回来源信息"""
    if collection.count() == 0:
        return []

    query_embedding = embedding_model.encode([query]).tolist()
    # 多要一些候选，再过滤
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(10, collection.count()),
        include=["metadatas", "distances"]
    )

    documents = []
    if results and results["metadatas"]:
        for meta_list, dist_list in zip(results["metadatas"], results["distances"]):
            for meta, dist in zip(meta_list, dist_list):
                # 余弦距离越小越相似，这里 distance_threshold 应为余弦距离阈值（0.5 约等于相似度 0.5）
                if dist <= distance_threshold:
                    content = meta.get("content", "")
                    source = meta.get("source", "未知来源")
                    documents.append({"content": content, "source": source})

    # 按距离排序（因为余弦距离是升序，已经排序过），取前 top_k
    documents = documents[:top_k]
    return documents

# ================= BERT 模型加载与预测（保持不变） =================
MODEL_PATH = "./frailty_bert_demo_model"

def load_bert_model():
    if not os.path.exists(MODEL_PATH):
        print(f"错误：未找到 BERT 模型目录 {MODEL_PATH}")
        sys.exit(1)
    try:
        tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
        model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        return tokenizer, model, device
    except Exception as e:
        print(f"BERT 模型加载失败：{str(e)}")
        sys.exit(1)

def predict_frailty(text, tokenizer, model, device):
    if not text or not text.strip():
        return {"prediction": "未知", "confidence": 0.0}
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = probs.argmax().item()
        conf = probs[0][pred].item()
    return {"prediction": "衰弱" if pred == 1 else "非衰弱", "confidence": conf}

# ================= RAG 解释生成 =================
EXPECTED_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "fried_items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "enum": [
                            "不明原因体重下降",
                            "疲乏/精力减退",
                            "握力下降",
                            "步速减慢",
                            "活动量减少"
                        ]
                    },
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

EXPECTED_NAMES = ["不明原因体重下降", "疲乏/精力减退", "握力下降", "步速减慢", "活动量减少"]

def validate_fried_items(data):
    if "fried_items" not in data:
        return False
    for item in data["fried_items"]:
        if item.get("name") not in EXPECTED_NAMES:
            return False
    return True


def generate_explanation(text, bert_pred, bert_conf):
    knowledge = """
    【Fried衰弱表型标准】（5项，满足≥3项为衰弱）
    1. 不明原因体重下降：过去1年内体重下降≥5%或≥4.5kg
    2. 疲乏/精力减退：自我报告“疲乏无力”、“整天没精神”等
    3. 握力下降：用手握力计测量，根据性别和BMI调整阈值
    4. 步速减慢：行走4.57米时间延长（根据身高、性别调整）
    5. 活动量减少：每周消耗热量低于标准
    """

    # 检索本地知识
    retrieved = retrieve_relevant_guidelines(text, top_k=5)
    if retrieved:
        # 格式化为带来源的引用
        ref_lines = []
        for i, item in enumerate(retrieved, 1):
            ref_lines.append(f"[{i}] 来源：{item['source']}\n{item['content']}")
        extra = "\n\n".join(ref_lines)
        knowledge += f"\n\n【本地循证指南】\n{extra}"

    prompt = f"""你是一位老年医学专家。请严格依据【Fried衰弱表型标准】对患者进行逐项评估，不得使用任何非标准术语。

    {knowledge}

    【患者信息】
    {text}

    请逐项判断，并严格按以下JSON Schema输出。注意：
    - fried_items数组中每一项的"name"必须**完全使用上述标准名称**（例如“不明原因体重下降”），不得自创或替换为疾病名称。
    - evidence字段必须直接引用病历中与该标准相关的原始描述，若病历未提及则填写“未提及”。
    - conclusion仅依据阳性计数得出（≥3项为“衰弱”，1-2项为“衰弱前期”，0项为“非衰弱”）。
    - suggestions中如需引用知识库，请注明来源编号。

    JSON Schema:
    {json.dumps(EXPECTED_JSON_SCHEMA, ensure_ascii=False, indent=2)}
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
    html += f"<p><b>2. 阳性指标总数：</b>{data.get('positive_count', 0)} 项</p>"
    conclusion = data.get("conclusion", "未知")
    color_conclusion = "red" if conclusion == "衰弱" else ("orange" if conclusion == "衰弱前期" else "green")
    html += f"<p><b>3. 最终评估结论：</b><span style='color:{color_conclusion}; font-weight:bold;'>{conclusion}</span></p>"
    html += f"<h4>4. 临床建议</h4><p>{data.get('suggestions', '')}</p>"
    return html

# ================= 后台线程（保持不变） =================
class BatchProcessThread(QThread):
    progress = Signal(int, int, str)
    result_ready = Signal(dict)

    def __init__(self, records, tokenizer, model, device):
        super().__init__()
        self.records = records
        self.tokenizer = tokenizer
        self.model = model
        self.device = device

    def run(self):
        total = len(self.records)
        for idx, rec in enumerate(self.records, start=1):
            text = rec.get("text", "")
            if not text.strip():
                continue
            result = predict_frailty(text, self.tokenizer, self.model, self.device)
            result["file_name"] = rec.get("file_name", "未知")
            result["name"] = rec.get("name", "")
            result["patient_id"] = rec.get("patient_id", "")
            result["timestamp"] = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")
            result["full_text"] = text
            self.result_ready.emit(result)
            self.progress.emit(idx, total, rec.get("display_name", rec.get("file_name", "未知")))
        self.progress.emit(total, total, "完成")

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
                    self.status_signal.emit(False, "⚠️ Ollama 服务正常，但未找到 qwen2.5:3b 模型，请运行 'ollama pull qwen2.5:3b'")
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
        self.setWindowTitle("老年衰弱智能评估系统")
        self.setGeometry(100, 100, 800, 650)
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

        # 患者列表表格
        self.patient_table = self.ui.findChild(QTableWidget, "patient_table")
        self._setup_patient_table()  # 调用表格初始化函数

        # 患者列表按钮
        self.btn_export_excel = self.ui.findChild(QPushButton, "btn_export_excel")
        self.btn_clear_list = self.ui.findChild(QPushButton, "btn_clear_list")

        # 导出动作（菜单/工具栏）
        self.action_export_excel = self.ui.findChild(QAction, "action_export_excel")

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

        self.btn_export_excel.clicked.connect(self.on_export_excel)
        self.btn_clear_list.clicked.connect(self.on_clear_patient_list)
        if self.action_export_excel:
            self.action_export_excel.triggered.connect(self.on_export_excel)

        # 双击表格行查看详情
        self.patient_table.cellDoubleClicked.connect(self.on_table_row_double_clicked)

        self.action_batch_import.triggered.connect(self.on_batch_import)

        # 占位功能
        def placeholder(title):
            QMessageBox.information(self, title, "该功能将在后续版本中提供。")
        #self.action_batch_import.triggered.connect(lambda: placeholder("批量导入"))
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
        if "error" in explanation_dict or not validate_fried_items(explanation_dict):
            # 显示错误或要求重新生成
            self.explain_display.setHtml("<p style='color:red;'>❌ 评估格式错误，未正确使用Fried标准，请重试。</p>")
            return
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

    # 批量导入
    @Slot()
    def on_batch_import(self):
        # 允许同时选择 txt 和表格文件
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择病历文件（可多选或混合类型）", "",
            "支持的文件 (*.txt *.csv *.xlsx *.xls);;文本文件 (*.txt);;表格文件 (*.csv *.xlsx *.xls);;所有文件 (*)"
        )
        if not files:
            return

        records = []
        # 分类处理
        txt_files = [f for f in files if f.lower().endswith(".txt")]
        table_files = [f for f in files if f.lower().endswith((".csv", ".xlsx", ".xls"))]

        # 处理 TXT 文件
        for file_path in txt_files:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            except Exception as e:
                QMessageBox.warning(self, "读取失败", f"无法读取 {os.path.basename(file_path)}: {str(e)}")
                continue
            if not text.strip():
                continue
            records.append({
                "text": text,
                "file_name": os.path.basename(file_path),
                "display_name": os.path.basename(file_path),
                "name": "",
                "patient_id": ""
            })

        # 处理表格文件
        for file_path in table_files:
            try:
                if file_path.lower().endswith(".csv"):
                    df = pd.read_csv(file_path, dtype=str).fillna("")
                else:
                    df = pd.read_excel(file_path, dtype=str).fillna("")
            except Exception as e:
                QMessageBox.warning(self, "表格读取失败", f"无法解析 {os.path.basename(file_path)}: {str(e)}")
                continue

            # 列名自动匹配
            col_map = {
                "text": ["病历文本", "text", "主诉现病史", "入院记录", "病情描述"],
                "name": ["姓名", "name", "患者姓名"],
                "patient_id": ["住院号", "patient_id", "病历号", "登记号"]
            }
            mapping = {}
            for key, candidates in col_map.items():
                for col in df.columns:
                    if col.strip().lower() in [c.lower() for c in candidates]:
                        mapping[key] = col
                        break
            if "text" not in mapping:
                QMessageBox.warning(self, "缺少必要列",
                                    f"文件 {os.path.basename(file_path)} 未找到病历文本列，已跳过")
                continue

            for _, row in df.iterrows():
                text = str(row[mapping["text"]])
                if not text.strip():
                    continue
                name = str(row.get(mapping.get("name", ""), ""))
                pid = str(row.get(mapping.get("patient_id", ""), ""))
                records.append({
                    "text": text,
                    "name": name,
                    "patient_id": pid,
                    "file_name": os.path.basename(file_path),
                    "display_name": f"{name or '未知'} ({os.path.basename(file_path)})"
                })

        if not records:
            QMessageBox.information(self, "无有效记录", "所选文件中没有可评估的病历文本。")
            return

        # 创建进度对话框
        progress_dialog = QProgressDialog("正在批量评估...", "取消", 0, len(records), self)
        progress_dialog.setWindowTitle("批量处理")
        progress_dialog.setWindowModality(Qt.WindowModal)
        progress_dialog.show()

        # 启动后台线程
        self.batch_thread = BatchProcessThread(records, self.tokenizer, self.model, self.device)

        def update_progress(current, total, name):
            progress_dialog.setValue(current)
            progress_dialog.setLabelText(f"正在处理：{name} ({current}/{total})")
            if current == total:
                progress_dialog.close()
                self.status_bar.showMessage(f"批量评估完成，共 {total} 条记录", 5000)

        self.batch_thread.progress.connect(update_progress)
        self.batch_thread.result_ready.connect(self._add_patient_row)
        progress_dialog.canceled.connect(self.batch_thread.terminate)
        self.batch_thread.start()



    #  双击行查看详情
    @Slot(int, int)
    def on_table_row_double_clicked(self, row, col):
        # 从文件名列获取完整文本
        item = self.patient_table.item(row, 3)  # 文件名列存储了 full_text
        if item is None:
            return
        full_text = item.data(Qt.UserRole)
        if full_text:
            self.text_input.setPlainText(full_text)
            # 可选：自动切换到输入标签页（如果输入区域不在标签页里，可以忽略）
            # self.result_tabs.setCurrentIndex(0)  # 假设输入区在第一个标签页，但实际UI中病历输入在左侧独立GroupBox，无需切换
            self.status_bar.showMessage("已加载选中患者病历，可进行快速预测或详细分析", 3000)
        else:
            QMessageBox.information(self, "提示", "该患者病历文本不可用")

    #导出 Excel
    @Slot()
    def on_export_excel(self):
        row_count = self.patient_table.rowCount()
        if row_count == 0:
            QMessageBox.information(self, "无数据", "患者列表为空，请先进行评估。")
            return

        file_path, _ = QFileDialog.getSaveFileName(
            self, "保存 Excel 文件", "衰弱评估结果.xlsx", "Excel 表格 (*.xlsx)"
        )
        if not file_path:
            return

        try:
            data = []
            for row in range(row_count):
                # 收集所有可见列数据
                row_data = []
                for col in range(self.patient_table.columnCount()):
                    item = self.patient_table.item(row, col)
                    row_data.append(item.text() if item else "")
                # 附加原始病历文本（从文件名列的 UserRole 获取）
                text_item = self.patient_table.item(row, 3)  # 文件名列存储了 full_text
                full_text = text_item.data(Qt.UserRole) if text_item else ""
                row_data.append(full_text[:500])  # 截取前500字符防止单元格过大
                data.append(row_data)

            # 构建完整表头
            headers = ["序号", "姓名", "住院号", "文件名", "预测结论", "置信度", "评估时间", "病历文本摘要"]
            df = pd.DataFrame(data, columns=headers)
            df.to_excel(file_path, index=False)
            QMessageBox.information(self, "导出成功", f"已保存至：{os.path.basename(file_path)}")
            self.status_bar.showMessage("导出 Excel 完成", 3000)
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"写入 Excel 时出错：{str(e)}")


    # 清空列表
    @Slot()
    def on_clear_patient_list(self):
        self.patient_table.setRowCount(0)
        self.status_bar.showMessage("患者列表已清空", 2000)

    def fill_example(self):
        example = """患者女性，78岁，退休教师。主诉：近1年体重下降8kg（原体重60kg），自觉"浑身没劲"，走路速度明显变慢，以前能轻松走完小区一圈，现在中途需要休息。家属反映患者近半年几乎不出门，日常家务也懒得做。既往史：高血压、骨关节炎。查体：步速0.6m/s，握力（右手）14kg（参考同龄女性正常值>18kg）。"""
        self.text_input.setPlainText(example)

    # 向表格添加行
    # 表格初始化
    def _setup_patient_table(self):
        """配置患者列表表格外观"""
        headers = ["序号", "姓名", "住院号", "文件名", "预测结论", "置信度", "评估时间"]
        self.patient_table.setColumnCount(len(headers))
        self.patient_table.setHorizontalHeaderLabels(headers)
        self.patient_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Stretch)
        self.patient_table.verticalHeader().setVisible(False)
        self.patient_table.setRowCount(0)

        # 设置表格本身与内部文字的背景/前景色，与浅色UI统一
        self.patient_table.setStyleSheet("""
            QTableWidget {
                background-color: white;
                alternate-background-color: #f5f7fa;
                color: #2c3e50;
                gridline-color: #dcdcdc;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
            }
        """)
        self.patient_table.setAlternatingRowColors(True)

    def _add_patient_row(self, result):
        """添加一行评估结果到患者列表，只改变姓名列字体颜色"""
        row = self.patient_table.rowCount()
        self.patient_table.insertRow(row)

        # 基础数据
        items = [
            str(row + 1),  # 序号
            result.get("name", ""),  # 姓名
            str(result.get("patient_id", "")),  # 住院号
            result.get("file_name", ""),  # 文件名
            result["prediction"],  # 预测结论
            f"{result['confidence']:.4f}",  # 置信度
            result.get("timestamp", "")  # 评估时间
        ]

        # 决定姓名列的字体颜色
        color_map = {"衰弱": Qt.red, "衰弱前期": Qt.darkYellow, "非衰弱": Qt.darkGreen}
        name_color = color_map.get(result["prediction"], Qt.black)

        for col, text in enumerate(items):
            item = QTableWidgetItem(text)
            # 只给姓名列 (col=1) 设置颜色
            if col == 1:
                item.setForeground(QColor(name_color))
                # 可选加粗
                font = item.font()
                font.setBold(True)
                item.setFont(font)
            self.patient_table.setItem(row, col, item)

        # 保存完整病历文本到“文件名”列的用户角色中，用于双击加载
        full_text = result.get("full_text", "")
        self.patient_table.item(row, 3).setData(Qt.UserRole, full_text)

# ================= 6. 启动 =================
if __name__ == "__main__":
    # 已经移除了 txt 读取和 initialize_knowledge_base 调用，
    # 因为知识库已通过 build_knowledge_base.py 构建完成。
    print("创建 QApplication...")
    app = QApplication(sys.argv)
    app.setFont(QFont("Microsoft YaHei", 10))

    print("加载 BERT 模型...")
    tokenizer, model, device = load_bert_model()

    # 直接创建主窗口，无需再次初始化知识库
    print("创建主窗口...")
    window = FrailtyMainWindow(tokenizer, model, device)
    window.show()
    print("窗口已显示，进入事件循环...")
    sys.exit(app.exec())