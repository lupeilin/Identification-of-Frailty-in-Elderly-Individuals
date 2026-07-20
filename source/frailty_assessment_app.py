#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
frailty_assessment_app.py
老年衰弱智能评估系统（带知识库查询 + 反馈收集 + 全自动持续学习）
用法：python frailty_assessment_app.py
前提：已运行 build_knowledge_base.py 构建知识库
"""

import os
import sys
import json
import requests
import sqlite3
import hashlib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from numpy import dot
from numpy.linalg import norm

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QMessageBox, QProgressDialog, QTableWidgetItem, QHeaderView,
    QTextEdit, QPushButton, QStatusBar, QFileDialog, QTabWidget, QTableWidget, QDialog,
    QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, QComboBox, QDialogButtonBox, QTextEdit as QDialogTextEdit,
    QWidget, QGroupBox, QFrame
)
from PySide6.QtCore import QThread, Signal, Slot, Qt, QDateTime
from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QFont, QAction, QColor
import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer
from huggingface_hub import snapshot_download
import shutil

from batch_annotation_tool import (
    init_batch_annotation_db, import_batch_records, auto_annotate_with_llm,
    get_records_for_review, submit_manual_label, analyze_annotation_quality,
    export_annotated_data, full_retrain_bert, BATCH_ANNOTATION_DB
)

# ================= 路径常量（基于脚本所在目录） =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DB_DIR = os.path.join(BASE_DIR, "geriatric_knowledge_db")
BERT_MODEL_PATH = os.path.join(BASE_DIR, "frailty_bert_demo_model")
UI_DIR = os.path.join(BASE_DIR, "tools")

# ================= 本地知识库模块 =================
chroma_client = chromadb.PersistentClient(path=DB_DIR)
collection = chroma_client.get_or_create_collection(name="geriatric_guidelines")


def load_embedding_model():
    """
    加载本地嵌入模型，使用 BAAI/bge-large-zh-v1.5 中文嵌入模型，
    下载失败时自动回退到通用多语言模型。
    """
    model_repo = "BAAI/bge-large-zh-v1.5"
    fallback_repo = "paraphrase-multilingual-MiniLM-L12-v2"

    local_model_dir = os.path.join(MODELS_DIR, "BAAI", "bge-large-zh-v1.5")
    fallback_dir = os.path.join(MODELS_DIR, "paraphrase-multilingual-MiniLM-L12-v2")

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


def retrieve_relevant_guidelines(query, top_k=5, distance_threshold=0.3):
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

# ================= BERT 模型加载与预测（支持热切换） =================
MODEL_PATH = BERT_MODEL_PATH

def load_bert_model(model_path=None):
    """加载 BERT 模型，支持传入自定义路径实现热切换"""
    if model_path is None:
        model_path = MODEL_PATH
    if not os.path.exists(model_path):
        print(f"错误：未找到 BERT 模型目录 {model_path}")
        sys.exit(1)
    try:
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"✅ BERT 模型加载完成：{model_path}")
        return tokenizer, model, device
    except Exception as e:
        print(f"BERT 模型加载失败：{str(e)}")
        sys.exit(1)

def reload_bert_model(model_path):
    """热切换 BERT 模型，不重启程序"""
    global tokenizer, model, device
    print(f"🔄 热切换 BERT 模型到：{model_path}")
    tokenizer, model, device = load_bert_model(model_path)
    return tokenizer, model, device

def predict_frailty(text, tokenizer, model, device):
    if not text or not text.strip():
        return {"prediction": "未知", "confidence": 0.0}
    inputs = tokenizer(text, padding="max_length", truncation=True, max_length=256, return_tensors="pt")
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
        name = item.get("name", "").strip()
        if name not in EXPECTED_NAMES:
            return False
    return True


# Few-shot 示例，用于引导 LLM 正确输出格式和生成个性化临床建议
FEW_SHOT_EXAMPLE = """
【示例1：衰弱（5项阳性）】
患者信息：患者男性，82岁，近1年体重下降8kg，自觉疲乏无力，握力下降（右手12kg），步速0.5m/s，近半年几乎不出门。

正确输出：
{
  "fried_items": [
    {"name": "不明原因体重下降", "result": "阳性", "evidence": "近1年体重下降8kg"},
    {"name": "疲乏/精力减退", "result": "阳性", "evidence": "自觉疲乏无力"},
    {"name": "握力下降", "result": "阳性", "evidence": "右手握力12kg（低于同龄男性正常值）"},
    {"name": "步速减慢", "result": "阳性", "evidence": "步速0.5m/s（低于正常值0.65m/s）"},
    {"name": "活动量减少", "result": "阳性", "evidence": "近半年几乎不出门"}
  ],
  "positive_count": 5,
  "conclusion": "衰弱",
  "suggestions": "【营养干预】患者存在明显体重下降，建议：①每日蛋白质摄入1.2-1.5g/kg（约60-75g），优选鸡蛋、牛奶、鱼肉；②口服营养补充剂（如安素）400-600kcal/日；③补充维生素D 800-1000IU/日 + 钙1000mg/日。\n【运动康复】患者握力下降、步速减慢、活动量少，建议：①渐进性抗阻训练（弹力带/轻哑铃），每周2-3次，每次8-12次/组，1-3组；②平衡训练（太极拳/单腿站立），每周≥3次，每次20-30分钟；③有氧运动（室内步行），从每天10分钟开始，逐步增至30分钟。\n【跌倒预防】患者步速减慢、活动量少，跌倒风险高：①居家安装扶手、防滑垫；②使用助行器辅助行走；③每3个月进行跌倒风险评估。\n【随访】每3个月复查体重、握力、步速，评估干预效果。建议转诊老年衰弱门诊，多学科团队（老年科、营养科、康复科）联合管理。"
}

【示例2：非衰弱（0项阳性）】
患者信息：患者女性，70岁，退休教师，体检发现血压升高。无体重下降，精神好，握力正常，每天散步1小时，做家务。

正确输出：
{
  "fried_items": [
    {"name": "不明原因体重下降", "result": "阴性", "evidence": "无体重下降"},
    {"name": "疲乏/精力减退", "result": "阴性", "evidence": "精神好"},
    {"name": "握力下降", "result": "阴性", "evidence": "握力正常"},
    {"name": "步速减慢", "result": "阴性", "evidence": "每天散步1小时"},
    {"name": "活动量减少", "result": "阴性", "evidence": "做家务，活动量正常"}
  ],
  "positive_count": 0,
  "conclusion": "非衰弱",
  "suggestions": "【健康维持】患者目前无衰弱表现，建议：①继续保持规律运动（每周≥150分钟中等强度有氧运动）；②均衡饮食，蛋白质摄入≥1.0g/kg/日；③每年进行一次衰弱筛查（FRAIL量表或Fried表型）；④控制血压，定期监测。\n【预防建议】虽然当前非衰弱，但70岁以上老年人衰弱风险随年龄增加，建议：①维持社交活动，预防抑郁；②预防跌倒（居家安全检查）；③定期体检，早期发现慢性病。"
}

【示例3：衰弱前期（2项阳性）】
患者信息：患者男性，75岁，近半年体重下降3kg（原体重70kg），自觉有些疲乏，但握力正常，步速正常，每天仍出门买菜。

正确输出：
{
  "fried_items": [
    {"name": "不明原因体重下降", "result": "阳性", "evidence": "近半年体重下降3kg（约4.3%，接近5%阈值）"},
    {"name": "疲乏/精力减退", "result": "阳性", "evidence": "自觉有些疲乏"},
    {"name": "握力下降", "result": "阴性", "evidence": "握力正常"},
    {"name": "步速减慢", "result": "阴性", "evidence": "步速正常"},
    {"name": "活动量减少", "result": "阴性", "evidence": "每天出门买菜"}
  ],
  "positive_count": 2,
  "conclusion": "衰弱前期",
  "suggestions": "【早期干预】患者处于衰弱前期，存在体重下降和疲乏，建议积极干预以逆转为健康状态：\n①营养干预：增加蛋白质摄入至1.2g/kg/日，每日补充鸡蛋1-2个、牛奶300ml；排查体重下降原因（食欲？吞咽？消化？）。\n②疲乏管理：评估是否存在贫血、甲状腺功能减退、抑郁等可逆因素；保证睡眠7-8小时/日。\n③运动强化：在现有活动基础上，增加抗阻训练（每周2次），预防肌少症。\n【随访】每3个月复查体重和疲乏程度，若体重继续下降或新增阳性指标，需及时转诊老年科。"
}
"""

# 全局开关：是否启用动态相似案例嵌入（阶段4）
_ENABLE_SIMILAR_CASES = False

def generate_explanation(text, bert_pred, bert_conf, retry_count=0):
    knowledge = """
    【Fried衰弱表型标准】（5项，满足≥3项为衰弱）
    1. 不明原因体重下降：过去1年内体重下降≥5%或≥4.5kg
    2. 疲乏/精力减退：自我报告"疲乏无力"、"整天没精神"等
    3. 握力下降：用手握力计测量，根据性别和BMI调整阈值
    4. 步速减慢：行走4.57米时间延长（根据身高、性别调整）
    5. 活动量减少：每周消耗热量低于标准
    """

    # 检索本地知识（病历相关 + 干预相关）
    retrieved = retrieve_relevant_guidelines(text, top_k=5)
    # 额外检索干预相关指南
    intervention_queries = ["衰弱 营养 干预", "衰弱 运动 康复", "衰弱 跌倒 预防", "衰弱 药物 管理"]
    for q in intervention_queries:
        extra_retrieved = retrieve_relevant_guidelines(q, top_k=2)
        if extra_retrieved:
            retrieved.extend(extra_retrieved)
    # 去重
    seen = set()
    unique_retrieved = []
    for item in retrieved:
        key = item.get("content", "")[:100]
        if key not in seen:
            seen.add(key)
            unique_retrieved.append(item)
    retrieved = unique_retrieved[:8]  # 最多8条
    
    if retrieved:
        # 格式化为带来源的引用
        ref_lines = []
        for i, item in enumerate(retrieved, 1):
            ref_lines.append(f"[{i}] 来源：{item['source']}\n{item['content']}")
        extra = "\n\n".join(ref_lines)
        knowledge += f"\n\n【本地循证指南】\n{extra}"

    # 阶段4：动态嵌入相似分歧案例（反馈≥100条时自动启用）
    similar_cases = ""
    if _ENABLE_SIMILAR_CASES:
        similar_cases = retrieve_similar_feedback_cases(text, top_k=2)
        if similar_cases:
            similar_cases = f"\n\n【历史类似病例参考】\n{similar_cases}"

    prompt = f"""你是一位老年医学专家。请严格依据【Fried衰弱表型标准】对患者进行逐项评估，不得使用任何非标准术语。

    {knowledge}

    {FEW_SHOT_EXAMPLE}
    {similar_cases}

    【临床建议生成规则】（必须遵守）
    1. 建议必须根据阳性指标的具体组合生成，不得使用千篇一律的模板。
    2. 每项阳性指标必须对应具体的干预措施：
       - 体重下降阳性 → 营养干预：蛋白质摄入量、ONS补充、维生素D/钙、排查原因
       - 疲乏阳性 → 评估可逆因素（贫血、甲减、抑郁、睡眠）、能量管理
       - 握力下降阳性 → 抗阻训练方案（频率、强度、肌群）、蛋白质补充
       - 步速减慢阳性 → 平衡训练、有氧运动、辅助器具、跌倒预防
       - 活动量减少阳性 → 渐进性运动计划、社交活动、心理支持
    3. 建议应分层：
       - 第一层：针对具体阳性指标的精准干预
       - 第二层：综合管理和随访计划
       - 第三层：转诊建议（如需要多学科团队）
    4. 建议中必须包含具体的数值（如蛋白质1.2g/kg、维生素D 800IU等）。
    5. 非衰弱患者：给出健康维持和预防建议，而非治疗建议。
    6. 衰弱前期患者：强调早期干预和逆转可能性，给出积极可操作的方案。
    7. 如需引用知识库，请在建议末尾注明来源编号。

    【患者信息】
    {text}

    请逐项判断，并严格按以下JSON Schema输出。注意：
    - fried_items数组中每一项的"name"必须**完全使用上述标准名称**（例如"不明原因体重下降"），不得自创或替换为疾病名称。
    - evidence字段必须直接引用病历中与该标准相关的原始描述，若病历未提及则填写"未提及"。
    - conclusion仅依据阳性计数得出（≥3项为"衰弱"，1-2项为"衰弱前期"，0项为"非衰弱"）。
    - suggestions必须根据上述【临床建议生成规则】生成个性化、具体、可操作的临床建议，不得使用笼统模板。

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
                    # 校验通过后返回
                    if validate_fried_items(data):
                        return data
                    else:
                        # 格式校验失败，尝试重试
                        if retry_count < 3:
                            print(f"⚠️ LLM 输出格式校验失败，第 {retry_count + 1} 次重试...")
                            return generate_explanation(text, bert_pred, bert_conf, retry_count + 1)
                        else:
                            return {"error": "LLM 多次重试后仍返回格式错误", "raw": raw_json}
                else:
                    if retry_count < 3:
                        print(f"⚠️ LLM 返回缺少必要字段，第 {retry_count + 1} 次重试...")
                        return generate_explanation(text, bert_pred, bert_conf, retry_count + 1)
                    else:
                        return {"error": "LLM 多次重试后仍缺少必要字段", "raw": raw_json}
            except json.JSONDecodeError:
                if retry_count < 3:
                    print(f"⚠️ LLM 返回非有效 JSON，第 {retry_count + 1} 次重试...")
                    return generate_explanation(text, bert_pred, bert_conf, retry_count + 1)
                else:
                    return {"error": "LLM 多次重试后仍返回无效 JSON", "raw": raw_json}
        else:
            return {"error": f"Ollama 返回错误，状态码：{response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到 Ollama，请确保已启动 ollama serve 并下载了相应模型。"}
    except Exception as e:
        return {"error": f"请求异常：{str(e)}"}


# ================= 反馈数据库模块（含 JSONL 防丢失 + 导出 + 同步） =================
FEEDBACK_DB_PATH = os.path.join(BASE_DIR, "feedback.db")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "feedback_log.jsonl")

def init_feedback_db():
    """初始化反馈数据库，创建 feedback 表，并从 JSONL 日志同步缺失记录"""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text_hash TEXT NOT NULL,
            patient_text TEXT NOT NULL,
            bert_prediction TEXT NOT NULL,
            bert_confidence REAL NOT NULL,
            llm_conclusion TEXT,
            llm_positive_count INTEGER,
            llm_suggestions TEXT,
            doctor_correction TEXT,
            correction_reason TEXT,
            doctor_name TEXT,
            created_at TEXT NOT NULL,
            UNIQUE(text_hash, created_at)
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ 反馈数据库初始化完成：{FEEDBACK_DB_PATH}")
    # 启动时从日志同步（崩溃恢复）
    sync_feedback_from_log()

def _append_feedback_log(record: dict):
    """追加写入 JSONL 日志文件，每行一条独立 JSON，确保即使系统崩溃也不丢失"""
    try:
        with open(FEEDBACK_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())  # 强制刷盘
    except Exception as e:
        print(f"⚠️ 反馈日志写入失败：{e}")

def save_feedback(patient_text, bert_pred, bert_conf, llm_conclusion, llm_positive_count,
                  llm_suggestions, doctor_correction, correction_reason, doctor_name=""):
    """保存医生反馈到数据库，同时追加 JSONL 日志防丢失"""
    text_hash = hashlib.sha256(patient_text.encode("utf-8")).hexdigest()[:16]
    created_at = QDateTime.currentDateTime().toString("yyyy-MM-dd hh:mm:ss")

    # 1. 先写日志（即使数据库失败，日志也保留）
    log_record = {
        "text_hash": text_hash,
        "patient_text": patient_text,
        "bert_prediction": bert_pred,
        "bert_confidence": bert_conf,
        "llm_conclusion": llm_conclusion,
        "llm_positive_count": llm_positive_count,
        "llm_suggestions": llm_suggestions,
        "doctor_correction": doctor_correction,
        "correction_reason": correction_reason,
        "doctor_name": doctor_name,
        "created_at": created_at
    }
    _append_feedback_log(log_record)

    # 2. 再写数据库
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        INSERT INTO feedback 
        (text_hash, patient_text, bert_prediction, bert_confidence, llm_conclusion,
         llm_positive_count, llm_suggestions, doctor_correction, correction_reason, doctor_name, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (text_hash, patient_text, bert_pred, bert_conf, llm_conclusion,
          llm_positive_count, llm_suggestions, doctor_correction, correction_reason, doctor_name, created_at))
    conn.commit()
    conn.close()
    return True

def get_feedback_stats():
    """获取反馈统计信息"""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM feedback")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT COUNT(*) FROM feedback WHERE doctor_correction != llm_conclusion")
    disagreements = cursor.fetchone()[0]
    conn.close()
    return {"total": total, "disagreements": disagreements}

def export_feedback_to_csv(csv_path=None):
    """导出所有反馈为 CSV 文件，供离线分析"""
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, f"feedback_export_{QDateTime.currentDateTime().toString('yyyyMMdd_hhmmss')}.csv")
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    df = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at DESC", conn)
    conn.close()
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    return csv_path

def get_high_frequency_disagreements(top_n=5):
    """获取高频分歧案例（医生校正与模型结论不一致），用于优化 Prompt"""
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT patient_text, bert_prediction, llm_conclusion, doctor_correction, correction_reason
        FROM feedback
        WHERE doctor_correction != llm_conclusion
        ORDER BY created_at DESC
        LIMIT ?
    """, (top_n,))
    rows = cursor.fetchall()
    conn.close()
    return [
        {
            "patient_text": r[0],
            "bert_prediction": r[1],
            "llm_conclusion": r[2],
            "doctor_correction": r[3],
            "correction_reason": r[4]
        }
        for r in rows
    ]

def sync_feedback_from_log():
    """从 JSONL 日志文件同步缺失的记录到数据库（崩溃恢复用）"""
    if not os.path.exists(FEEDBACK_LOG_PATH):
        return 0
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    cursor = conn.cursor()
    synced = 0
    with open(FEEDBACK_LOG_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                cursor.execute(
                    "SELECT COUNT(*) FROM feedback WHERE text_hash = ? AND created_at = ?",
                    (record["text_hash"], record["created_at"])
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute("""
                        INSERT INTO feedback 
                        (text_hash, patient_text, bert_prediction, bert_confidence, llm_conclusion,
                         llm_positive_count, llm_suggestions, doctor_correction, correction_reason, doctor_name, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        record["text_hash"], record["patient_text"], record["bert_prediction"],
                        record["bert_confidence"], record["llm_conclusion"], record["llm_positive_count"],
                        record["llm_suggestions"], record["doctor_correction"], record["correction_reason"],
                        record["doctor_name"], record["created_at"]
                    ))
                    synced += 1
            except Exception as e:
                print(f"⚠️ 日志同步跳过一行：{e}")
    conn.commit()
    conn.close()
    if synced > 0:
        print(f"✅ 从日志同步 {synced} 条反馈到数据库")
    return synced


# ================= 持续学习：全自动优化调度器 =================

# 阶段阈值配置
STAGE_THRESHOLDS = {
    "prompt_update": 20,      # 阶段2：自动更新 Prompt
    "bert_finetune": 50,    # 阶段3：自动增量微调 BERT
    "similar_cases": 100,     # 阶段4：启用动态相似案例嵌入
}

# 记录已触发的阶段（避免重复触发）
_triggered_stages = set()

def _check_and_trigger_learning():
    """
    检查反馈数量，达到阈值时自动触发对应的优化阶段。
    在每次 save_feedback 后调用。
    """
    global _ENABLE_SIMILAR_CASES
    stats = get_feedback_stats()
    total = stats["total"]
    disagreements = stats["disagreements"]

    print(f"📊 当前反馈统计：总计 {total} 条，分歧 {disagreements} 条")

    # 阶段2：≥20 条分歧 → 自动更新 Prompt
    if disagreements >= STAGE_THRESHOLDS["prompt_update"] and "prompt_update" not in _triggered_stages:
        print(f"🎯 触发阶段2：分歧样本达到 {disagreements} 条，自动更新 Prompt...")
        update_prompt_with_feedback()
        _triggered_stages.add("prompt_update")

    # 阶段3：≥50 条分歧 → 自动增量微调 BERT（在后台线程中执行，避免阻塞 UI）
    if disagreements >= STAGE_THRESHOLDS["bert_finetune"] and "bert_finetune" not in _triggered_stages:
        print(f"🎯 触发阶段3：分歧样本达到 {disagreements} 条，自动增量微调 BERT...")
        # 启动后台线程执行微调，避免阻塞 UI
        thread = AutoFinetuneThread()
        thread.finished.connect(_on_bert_finetune_complete)
        thread.start()
        _triggered_stages.add("bert_finetune")

    # 阶段4：≥100 条分歧 → 启用动态相似案例嵌入
    if disagreements >= STAGE_THRESHOLDS["similar_cases"] and "similar_cases" not in _triggered_stages:
        print(f"🎯 触发阶段4：分歧样本达到 {disagreements} 条，启用动态相似案例嵌入...")
        _ENABLE_SIMILAR_CASES = True
        _triggered_stages.add("similar_cases")

def _on_bert_finetune_complete(new_model_path):
    """BERT 增量微调完成后的回调：热切换模型"""
    if new_model_path and os.path.exists(new_model_path):
        print(f"🔄 BERT 新模型已生成：{new_model_path}")
        # 通知用户，建议重启或提供热切换选项
        # 实际热切换需要主窗口配合，这里先打印提示
    else:
        print("⚠️ BERT 增量微调失败或未达条件")

class AutoFinetuneThread(QThread):
    """后台自动微调 BERT 的线程，避免阻塞 UI"""
    finished = Signal(str)  # 返回新模型路径

    def run(self):
        try:
            new_path = incremental_finetune_bert(min_samples=50)
            self.finished.emit(new_path or "")
        except Exception as e:
            print(f"❌ 自动微调失败：{e}")
            self.finished.emit("")


# ================= 持续学习：BERT 增量微调 =================

def prepare_feedback_training_data(min_samples=50):
    """
    从反馈数据库中提取医生校正的数据，准备增量微调。
    只取 doctor_correction != bert_prediction 的样本（模型错了的才有学习价值）。
    """
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    df = pd.read_sql_query("""
        SELECT patient_text, doctor_correction 
        FROM feedback 
        WHERE doctor_correction != bert_prediction
    """, conn)
    conn.close()

    if len(df) < min_samples:
        print(f"⚠️ 分歧样本仅 {len(df)} 条，建议积累到 {min_samples} 条再微调")
        return None

    # 标签映射：衰弱前期也归入衰弱类（二分类）
    label_map = {"非衰弱": 0, "衰弱前期": 1, "衰弱": 1}
    df["label"] = df["doctor_correction"].map(label_map)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    return df[["patient_text", "label"]]

def incremental_finetune_bert(min_samples=50, output_dir=None):
    """
    增量微调 BERT：冻结底层，只训练分类头，防止灾难性遗忘。
    保存为新版本，不覆盖原模型。
    """
    if output_dir is None:
        # 自动寻找下一个版本号
        v = 2
        while os.path.exists(os.path.join(BASE_DIR, f"frailty_bert_demo_model_v{v}")):
            v += 1
        output_dir = os.path.join(BASE_DIR, f"frailty_bert_demo_model_v{v}")

    df = prepare_feedback_training_data(min_samples)
    if df is None:
        return None

    print(f"🚀 开始增量微调 BERT，使用 {len(df)} 条分歧样本...")

    # 加载现有模型
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    model = BertForSequenceClassification.from_pretrained(BERT_MODEL_PATH, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # 冻结 BERT 底层（保留通用语义能力），只训练分类头
    for param in model.bert.parameters():
        param.requires_grad = False

    # 数据准备
    train_df, eval_df = train_test_split(df, test_size=0.2, random_state=42)
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(eval_df)
    })

    def tokenize(batch):
        return tokenizer(batch["patient_text"], padding="max_length", truncation=True, max_length=256)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # 训练参数（小学习率，少量步数，防止过拟合）
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "frailty_bert_incremental"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=1e-5,          # 比初始训练更小
        per_device_train_batch_size=4,
        num_train_epochs=3,          # 少量 epoch 即可
        weight_decay=0.01,
        load_best_model_at_end=True,
        logging_steps=10,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
    )

    trainer.train()

    # 保存为新版本
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ 增量微调完成，保存至 {output_dir}")
    return output_dir


# ================= 持续学习：LLM Prompt 动态优化 =================

def generate_negative_few_shot_examples(top_n=3):
    """
    从反馈中提取高频分歧案例，生成【错误案例→正确示例】对照，
    用于更新 Prompt 中的 Few-shot。
    """
    disagreements = get_high_frequency_disagreements(top_n)
    if not disagreements:
        return ""

    examples = []
    for d in disagreements:
        examples.append(f"""
【错误案例】
患者：{d['patient_text'][:100]}...
模型结论：{d['llm_conclusion']}
医生校正：{d['doctor_correction']}
原因：{d['correction_reason']}

【正确做法】
{d['correction_reason']}，因此应判定为"{d['doctor_correction']}"。
""")
    return "\n".join(examples)

def update_prompt_with_feedback():
    """将分歧案例融入 Prompt，作为反例提示"""
    global FEW_SHOT_EXAMPLE
    negative_examples = generate_negative_few_shot_examples(3)

    if negative_examples:
        FEW_SHOT_EXAMPLE = f"""
【正例】
患者信息：患者男性，82岁，近1年体重下降5kg，自觉疲乏无力，握力下降，步速减慢，活动量减少。
正确结论：衰弱

【常见错误警示】
{negative_examples}

请严格区分以下情况：
- 卒中/骨折等疾病导致的肌力下降 ≠ Fried 标准的"握力下降"
- 术后卧床 ≠ Fried 标准的"活动量减少"
- 只有无明确病因的生理性储备下降才符合 Fried 标准
"""
        print("✅ Prompt 已更新，融入历史分歧案例")
    else:
        print("⚠️ 暂无分歧案例，Prompt 保持默认")

def retrieve_similar_feedback_cases(patient_text, top_k=2):
    """
    用文本相似度检索历史分歧案例，嵌入到当前 Prompt。
    使用 BGE 嵌入模型计算病历相似度。
    """
    conn = sqlite3.connect(FEEDBACK_DB_PATH)
    df = pd.read_sql_query("""
        SELECT patient_text, doctor_correction, correction_reason 
        FROM feedback 
        WHERE doctor_correction != llm_conclusion
    """, conn)
    conn.close()

    if len(df) == 0:
        return ""

    # 计算当前病历与历史分歧案例的嵌入相似度
    query_emb = embedding_model.encode([patient_text])
    case_embs = embedding_model.encode(df["patient_text"].tolist())

    # 计算余弦相似度
    similarities = []
    for case_emb in case_embs:
        sim = dot(query_emb, case_emb) / (norm(query_emb) * norm(case_emb))
        similarities.append(sim[0])

    df["similarity"] = similarities
    df = df.sort_values("similarity", ascending=False).head(top_k)

    cases = []
    for _, row in df.iterrows():
        cases.append(f"""
类似病例参考（相似度 {row['similarity']:.3f}）：
患者：{row['patient_text'][:80]}...
医生校正：{row['doctor_correction']}
理由：{row['correction_reason']}
""")

    return "\n".join(cases)


# ================= 反馈对话框 =================
class FeedbackDialog(QDialog):
    def __init__(self, parent, bert_pred, llm_conclusion, patient_text):
        super().__init__(parent)
        self.setWindowTitle("医生反馈 - 评估结果校正")
        self.setGeometry(200, 200, 500, 400)
        self.patient_text = patient_text
        self.bert_pred = bert_pred
        self.llm_conclusion = llm_conclusion

        layout = QVBoxLayout(self)

        # 显示当前评估结果
        info = QLabel(f"<b>BERT 预测：</b>{bert_pred}<br><b>LLM 结论：</b>{llm_conclusion}")
        info.setWordWrap(True)
        layout.addWidget(info)

        # 医生校正选择
        layout.addWidget(QLabel("您认为正确的评估结论："))
        self.combo_correction = QComboBox()
        self.combo_correction.addItems(["衰弱", "衰弱前期", "非衰弱"])
        if llm_conclusion in ["衰弱", "衰弱前期", "非衰弱"]:
            self.combo_correction.setCurrentText(llm_conclusion)
        layout.addWidget(self.combo_correction)

        # 校正原因
        layout.addWidget(QLabel("校正原因（选填）："))
        self.text_reason = QDialogTextEdit()
        self.text_reason.setPlaceholderText("请说明为什么认为当前评估不正确...")
        self.text_reason.setMaximumHeight(80)
        layout.addWidget(self.text_reason)

        # 医生姓名
        layout.addWidget(QLabel("医生姓名（选填）："))
        self.line_doctor = QLineEdit()
        layout.addWidget(self.line_doctor)

        # 按钮
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_feedback_data(self):
        return {
            "doctor_correction": self.combo_correction.currentText(),
            "correction_reason": self.text_reason.toPlainText().strip(),
            "doctor_name": self.line_doctor.text().strip()
        }


# ================= 病历缺失指标检查 =================
FRIED_KEYWORDS = {
    "不明原因体重下降": ["体重下降", "体重减轻", "消瘦", "体重减少", "体重降低"],
    "疲乏/精力减退": ["疲乏", "乏力", "无力", "没精神", "精力减退", "疲惫", "倦怠"],
    "握力下降": ["握力", "手力", "抓握"],
    "步速减慢": ["步速", "步态", "行走缓慢", "走路慢", "步行速度"],
    "活动量减少": ["活动量", "活动减少", "卧床", "不愿活动", "运动减少"]
}

def check_missing_indicators(text):
    """
    检查病历中是否缺失 Fried 衰弱评估所需的关键指标。
    返回缺失的指标列表和匹配到的指标列表。
    """
    text_lower = text.lower()
    missing = []
    matched = []
    for indicator, keywords in FRIED_KEYWORDS.items():
        found = any(kw in text_lower for kw in keywords)
        if found:
            matched.append(indicator)
        else:
            missing.append(indicator)
    return missing, matched

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


# ================= 批量标注后台线程 =================

class BatchImportThread(QThread):
    """后台导入病历文件，避免阻塞UI"""
    progress = Signal(int, int, str)  # current, total, message
    finished_signal = Signal(int, int, str)  # imported, skipped, message

    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path

    def run(self):
        try:
            init_batch_annotation_db()
            imported = import_batch_records(self.file_path)
            self.finished_signal.emit(imported, 0, f"导入完成：新增 {imported} 条记录")
        except Exception as e:
            self.finished_signal.emit(0, 0, f"导入失败：{str(e)}")


class BatchLLMThread(QThread):
    """后台LLM预标注线程"""
    progress = Signal(int, int, str)
    finished_signal = Signal(int, str)

    def __init__(self, limit=None):
        super().__init__()
        self.limit = limit

    def run(self):
        try:
            annotated = auto_annotate_with_llm(limit=self.limit)
            self.finished_signal.emit(annotated, f"LLM预标注完成：{annotated} 条")
        except Exception as e:
            self.finished_signal.emit(0, f"LLM预标注失败：{str(e)}")


class BatchRetrainThread(QThread):
    """后台BERT重训练线程"""
    progress = Signal(str)
    finished_signal = Signal(str, str)  # model_path or None, message

    def __init__(self, min_samples=100, epochs=5):
        super().__init__()
        self.min_samples = min_samples
        self.epochs = epochs

    def run(self):
        try:
            self.progress.emit("正在准备训练数据...")
            output_dir = full_retrain_bert(min_samples=self.min_samples, epochs=self.epochs)
            if output_dir:
                self.finished_signal.emit(output_dir, f"训练完成！新模型：{output_dir}")
            else:
                self.finished_signal.emit("", "训练未执行：样本不足或数据缺失")
        except Exception as e:
            self.finished_signal.emit("", f"训练失败：{str(e)}")


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
        self.action_batch_annotation = self.ui.findChild(QAction, "action_batch_annotation")
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
        if self.action_batch_annotation:
            self.action_batch_annotation.triggered.connect(self._open_batch_annotation_dialog)

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

        # 初始化反馈数据库
        init_feedback_db()

        # 添加反馈按钮（从 UI 文件加载）
        self.btn_feedback = self.ui.findChild(QPushButton, "btn_feedback")
        if self.btn_feedback:
            self.btn_feedback.clicked.connect(self.on_feedback)
        else:
            # 备用：如果 UI 中没有，动态创建
            self.btn_feedback = QPushButton("💡 提交反馈", self)
            self.btn_feedback.setGeometry(600, 580, 120, 30)
            self.btn_feedback.clicked.connect(self.on_feedback)
            self.btn_feedback.setEnabled(False)

        self.explain_thread = None
        self._last_explanation = None
        self._last_patient_text = ""
        self._last_bert_result = None

        # ===== 批量标注与模型训练面板（改为独立窗口，通过工具菜单打开） =====
        self._batch_annotation_dialog = None  # 延迟创建

        print("主窗口初始化完成")

    def _open_batch_annotation_dialog(self):
        """打开批量标注与模型训练独立窗口"""
        if self._batch_annotation_dialog is None:
            self._batch_annotation_dialog = BatchAnnotationDialog(self)
        self._batch_annotation_dialog.show()
        self._batch_annotation_dialog.raise_()
        self._batch_annotation_dialog.activateWindow()


    # ---------- 原有反馈按钮方法 ----------

    def _enable_feedback_button(self):
        """评估完成后启用反馈按钮"""
        if self.btn_feedback:
            self.btn_feedback.setEnabled(True)

    def _disable_feedback_button(self):
        """新评估开始时禁用反馈按钮"""
        if self.btn_feedback:
            self.btn_feedback.setEnabled(False)



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
        # 检查病历中是否缺失 Fried 关键指标
        missing, matched = check_missing_indicators(text)
        if len(missing) >= 3:
            reply = QMessageBox.question(
                self,
                "病历数据缺失警告",
                f"当前病历中缺少以下 {len(missing)} 项 Fried 衰弱评估关键指标：\n\n" +
                "\n".join([f"  • {m}" for m in missing]) +
                f"\n\n仅匹配到 {len(matched)} 项：" + (", ".join(matched) if matched else "无") +
                "\n\n缺少关键指标可能导致评估结果不可靠，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.status_bar.showMessage("用户取消评估：病历数据缺失", 3000)
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
        # 检查病历中是否缺失 Fried 关键指标
        missing, matched = check_missing_indicators(text)
        if len(missing) >= 3:
            reply = QMessageBox.question(
                self,
                "病历数据缺失警告",
                f"当前病历中缺少以下 {len(missing)} 项 Fried 衰弱评估关键指标：\n\n" +
                "\n".join([f"  • {m}" for m in missing]) +
                f"\n\n仅匹配到 {len(matched)} 项：" + (", ".join(matched) if matched else "无") +
                "\n\n缺少关键指标可能导致评估结果不可靠，是否继续？",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.status_bar.showMessage("用户取消评估：病历数据缺失", 3000)
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
        self.explain_thread.finished.connect(self._enable_feedback_button)
        self._disable_feedback_button()
        self.explain_thread.start()

    @Slot(dict)
    def on_explanation_ready(self, explanation_dict):
        if "error" in explanation_dict or not validate_fried_items(explanation_dict):
            # 显示错误或要求重新生成
            self.explain_display.setHtml("<p style='color:red;'>❌ 评估格式错误，未正确使用Fried标准，请重试。</p>")
            self._disable_feedback_button()
            return
        # 保存当前解释结果，用于后续反馈
        self._last_explanation = explanation_dict
        self._last_patient_text = self.text_input.toPlainText().strip()
        self._last_bert_result = predict_frailty(self._last_patient_text, self.tokenizer, self.model, self.device)
        html_content = format_explanation_html(explanation_dict)
        # 在报告末尾追加反馈按钮提示
        feedback_html = """
        <hr>
        <p><b>💡 医生反馈</b></p>
        <p>如果您认为上述评估结果不正确，请点击窗口右下角的【提交反馈】按钮提交校正意见，帮助我们改进模型。</p>
        """
        self.explain_display.setHtml(html_content + feedback_html)
        if self.result_tabs is not None:
            self.result_tabs.setCurrentIndex(1)
        self.status_bar.showMessage("详细解释已生成", 5000)

    @Slot()
    def on_feedback(self):
        """打开反馈对话框并保存医生校正意见，保存后自动触发持续学习检查"""
        if not hasattr(self, '_last_explanation') or not self._last_explanation:
            QMessageBox.warning(self, "提示", "请先进行一次详细评估，再提交反馈。")
            return
        bert_pred = self._last_bert_result.get("prediction", "未知")
        bert_conf = self._last_bert_result.get("confidence", 0.0)
        llm_conclusion = self._last_explanation.get("conclusion", "未知")
        llm_positive_count = self._last_explanation.get("positive_count", 0)
        llm_suggestions = self._last_explanation.get("suggestions", "")
        patient_text = self._last_patient_text

        dialog = FeedbackDialog(self, bert_pred, llm_conclusion, patient_text)
        if dialog.exec() == QDialog.Accepted:
            data = dialog.get_feedback_data()
            try:
                save_feedback(
                    patient_text=patient_text,
                    bert_pred=bert_pred,
                    bert_conf=bert_conf,
                    llm_conclusion=llm_conclusion,
                    llm_positive_count=llm_positive_count,
                    llm_suggestions=llm_suggestions,
                    doctor_correction=data["doctor_correction"],
                    correction_reason=data["correction_reason"],
                    doctor_name=data["doctor_name"]
                )
                stats = get_feedback_stats()
                QMessageBox.information(
                    self, "反馈已保存",
                    f"感谢您的反馈！\n\n"
                    f"累计反馈：{stats['total']} 条\n"
                    f"与模型结论不一致：{stats['disagreements']} 条"
                )
                self.status_bar.showMessage(f"反馈已保存，累计 {stats['total']} 条", 3000)

                # ===== 自动触发持续学习 =====
                _check_and_trigger_learning()

            except Exception as e:
                QMessageBox.critical(self, "保存失败", f"反馈保存失败：{str(e)}")

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

        # 保存完整病历文本到"文件名"列的用户角色中，用于双击加载
        full_text = result.get("full_text", "")
        self.patient_table.item(row, 3).setData(Qt.UserRole, full_text)


# ================= 批量标注与模型训练独立对话框 =================

class BatchAnnotationDialog(QDialog):
    """批量标注与模型训练独立对话框"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("📚 批量标注与模型训练")
        self.setGeometry(100, 100, 1200, 850)
        self.parent_window = parent
        self._review_records_map = {}
        self._normal_geometry = None  # 保存正常窗口大小

        # 初始化数据库
        init_batch_annotation_db()

        # 设置白色底色
        self.setStyleSheet("""
            QDialog {
                background-color: #ffffff;
            }
            QGroupBox {
                background-color: #ffffff;
            }
            QLabel {
                background-color: transparent;
                color: #2c3e50;
            }
            QPushButton {
                background-color: #f8f9fa;
                color: #2c3e50;
                border: 1px solid #dee2e6;
                border-radius: 4px;
                padding: 6px 12px;
            }
            QPushButton:hover {
                background-color: #e9ecef;
            }
            QLineEdit {
                background-color: #ffffff;
                color: #2c3e50;
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 4px 8px;
            }
            QTextEdit {
                background-color: #fafbfc;
                color: #2c3e50;
                border: 1px solid #ced4da;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #ffffff;
                color: #2c3e50;
                alternate-background-color: #f5f7fa;
                gridline-color: #dcdcdc;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QTableWidget::item {
                color: #2c3e50;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                padding: 6px;
                font-weight: bold;
            }
        """)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 统一字体
        self.setFont(QFont("Microsoft YaHei", 10))

        # ========== 顶部窗口控制栏 ==========
        window_control = QHBoxLayout()
        window_control.addWidget(QLabel("<b>📚 批量标注与模型训练</b>"))
        window_control.addStretch()

        # 窗口控制按钮
        btn_minimize = QPushButton("🗕 最小化")
        btn_minimize.setFixedWidth(80)
        btn_minimize.setToolTip("最小化窗口到任务栏")
        btn_minimize.clicked.connect(self.showMinimized)
        window_control.addWidget(btn_minimize)

        btn_zoom_out = QPushButton("🔍-")
        btn_zoom_out.setFixedWidth(50)
        btn_zoom_out.setToolTip("缩小窗口")
        btn_zoom_out.clicked.connect(self._zoom_out)
        window_control.addWidget(btn_zoom_out)

        btn_zoom_in = QPushButton("🔍+")
        btn_zoom_in.setFixedWidth(50)
        btn_zoom_in.setToolTip("放大窗口")
        btn_zoom_in.clicked.connect(self._zoom_in)
        window_control.addWidget(btn_zoom_in)

        btn_fullscreen = QPushButton("⛶ 全屏")
        btn_fullscreen.setFixedWidth(70)
        btn_fullscreen.setToolTip("全屏显示（按 Esc 退出）")
        btn_fullscreen.clicked.connect(self._toggle_fullscreen)
        window_control.addWidget(btn_fullscreen)

        btn_close = QPushButton("✕ 关闭")
        btn_close.setFixedWidth(70)
        btn_close.setToolTip("关闭窗口")
        btn_close.setStyleSheet("background-color: #dc3545; color: white; border: none;")
        btn_close.clicked.connect(self.hide)
        window_control.addWidget(btn_close)

        main_layout.addLayout(window_control)

        # 分隔线
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #dee2e6;")
        main_layout.addWidget(line)

        # ========== 顶部状态栏 ==========
        status_bar = QHBoxLayout()
        self.batch_status_label = QLabel("📊 批量标注状态：未初始化")
        self.batch_status_label.setStyleSheet("font-size: 13px; color: #2c3e50; font-weight: bold; background-color: transparent;")
        status_bar.addWidget(self.batch_status_label)
        status_bar.addStretch()

        btn_refresh_status = QPushButton("🔄 刷新状态")
        btn_refresh_status.setFixedWidth(120)
        btn_refresh_status.clicked.connect(self._on_batch_refresh_status)
        status_bar.addWidget(btn_refresh_status)
        main_layout.addLayout(status_bar)

        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #dee2e6;")
        main_layout.addWidget(line2)

        # ========== 步骤1：导入病历 ==========
        group_import = self._create_group_box("步骤1：导入脱密病历")
        import_layout = QHBoxLayout(group_import)
        btn_import = QPushButton("📁 选择 CSV/Excel 文件")
        btn_import.setStyleSheet("font-weight: bold; padding: 8px 20px;")
        btn_import.clicked.connect(self._on_batch_import_records)
        import_layout.addWidget(btn_import)
        self.batch_import_info = QLabel("未导入")
        self.batch_import_info.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        import_layout.addWidget(self.batch_import_info)
        import_layout.addStretch()
        main_layout.addWidget(group_import)

        # ========== 步骤2：LLM 预标注 ==========
        group_llm = self._create_group_box("步骤2：LLM 详细预标注")
        llm_layout = QHBoxLayout(group_llm)
        btn_auto_llm = QPushButton("🤖 运行 LLM 预标注")
        btn_auto_llm.setStyleSheet("padding: 8px 20px;")
        btn_auto_llm.clicked.connect(self._on_batch_auto_llm)
        llm_layout.addWidget(btn_auto_llm)
        self.batch_llm_info = QLabel("未执行")
        self.batch_llm_info.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        llm_layout.addWidget(self.batch_llm_info)
        llm_layout.addStretch()
        main_layout.addWidget(group_llm)

        # ========== 步骤3-4：人工审核标注 ==========
        group_review = self._create_group_box("步骤3-4：人工审核标注")
        review_main = QVBoxLayout(group_review)

        # 加载按钮
        review_header = QHBoxLayout()
        btn_load_review = QPushButton("🔄 加载待审核记录")
        btn_load_review.setStyleSheet("padding: 6px 16px;")
        btn_load_review.clicked.connect(self._on_batch_load_review)
        review_header.addWidget(btn_load_review)
        review_header.addWidget(QLabel("<span style='color:#7f8c8d;'>（优先显示不确定样本，黄色高亮）</span>"))
        review_header.addStretch()
        review_main.addLayout(review_header)

        # 左右分栏：左侧表格，右侧病历文本
        review_split = QHBoxLayout()

        # 左侧：表格 + 审核按钮
        left_panel = QVBoxLayout()
        self.review_table = QTableWidget()
        self.review_table.setColumnCount(5)
        self.review_table.setHorizontalHeaderLabels([
            "ID", "患者ID", "BERT预测", "BERT置信度", "LLM结论"
        ])
        self.review_table.horizontalHeader().setSectionResizeMode(4, QHeaderView.Stretch)
        self.review_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.review_table.setSelectionMode(QTableWidget.SingleSelection)
        self.review_table.setStyleSheet("""
            QTableWidget {
                background-color: #ffffff;
                alternate-background-color: #f5f7fa;
                color: #2c3e50;
                gridline-color: #dcdcdc;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
            }
            QTableWidget::item {
                color: #2c3e50;
                padding: 4px;
            }
            QTableWidget::item:selected {
                background-color: #3498db;
                color: #ffffff;
            }
            QHeaderView::section {
                background-color: #ecf0f1;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                padding: 6px;
                font-weight: bold;
            }
        """)
        self.review_table.setAlternatingRowColors(True)
        self.review_table.itemSelectionChanged.connect(self._on_review_row_selected)
        left_panel.addWidget(self.review_table)

        # 审核按钮区
        review_actions = QHBoxLayout()
        review_actions.addWidget(QLabel("<b>审核人：</b>"))
        self.review_annotator_input = QLineEdit()
        self.review_annotator_input.setPlaceholderText("输入医生姓名")
        self.review_annotator_input.setFixedWidth(150)
        review_actions.addWidget(self.review_annotator_input)
        review_actions.addSpacing(20)

        btn_label_frail = QPushButton("✅ 衰弱")
        btn_label_frail.setStyleSheet("background-color: #e74c3c; color: white; padding: 6px 16px; font-weight: bold; border-radius: 4px;")
        btn_label_frail.clicked.connect(lambda: self._on_batch_label_selected("衰弱"))
        review_actions.addWidget(btn_label_frail)

        btn_label_pre = QPushButton("⚠️ 衰弱前期")
        btn_label_pre.setStyleSheet("background-color: #f39c12; color: white; padding: 6px 16px; font-weight: bold; border-radius: 4px;")
        btn_label_pre.clicked.connect(lambda: self._on_batch_label_selected("衰弱前期"))
        review_actions.addWidget(btn_label_pre)

        btn_label_non = QPushButton("❌ 非衰弱")
        btn_label_non.setStyleSheet("background-color: #27ae60; color: white; padding: 6px 16px; font-weight: bold; border-radius: 4px;")
        btn_label_non.clicked.connect(lambda: self._on_batch_label_selected("非衰弱"))
        review_actions.addWidget(btn_label_non)
        review_actions.addStretch()
        left_panel.addLayout(review_actions)

        review_split.addLayout(left_panel, stretch=3)

        # 右侧：病历文本显示区
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("<b>📋 病历原文</b>"))
        self.review_text_display = QTextEdit()
        self.review_text_display.setReadOnly(True)
        self.review_text_display.setPlaceholderText("点击左侧表格中的某一行，此处将显示该患者的完整病历文本...")
        self.review_text_display.setStyleSheet("""
            QTextEdit {
                background-color: #fafbfc;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 4px;
                padding: 10px;
                font-family: "Microsoft YaHei", sans-serif;
                font-size: 12px;
                line-height: 1.6;
            }
        """)
        right_panel.addWidget(self.review_text_display)
        review_split.addLayout(right_panel, stretch=2)

        review_main.addLayout(review_split)
        main_layout.addWidget(group_review, stretch=3)

        # 分隔线
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #bdc3c7;")
        main_layout.addWidget(line2)

        # ========== 步骤5：数据分析 ==========
        group_analyze = self._create_group_box("步骤5：数据分析")
        analyze_layout = QHBoxLayout(group_analyze)
        btn_analyze = QPushButton("📊 查看标注质量分析")
        btn_analyze.setStyleSheet("padding: 8px 20px;")
        btn_analyze.clicked.connect(self._on_batch_analyze)
        analyze_layout.addWidget(btn_analyze)
        self.batch_analyze_info = QLabel("未分析")
        self.batch_analyze_info.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        analyze_layout.addWidget(self.batch_analyze_info)
        analyze_layout.addStretch()
        main_layout.addWidget(group_analyze)

        # ========== 步骤6：重训练 BERT ==========
        group_retrain = self._create_group_box("步骤6：模型重训练")
        retrain_layout = QHBoxLayout(group_retrain)
        btn_retrain = QPushButton("🚀 全量重训练 BERT")
        btn_retrain.setStyleSheet("font-weight: bold; padding: 8px 20px; background-color: #3498db; color: white; border-radius: 4px;")
        btn_retrain.clicked.connect(self._on_batch_retrain)
        retrain_layout.addWidget(btn_retrain)
        retrain_layout.addSpacing(20)

        retrain_layout.addWidget(QLabel("最少样本："))
        self.retrain_min_samples = QLineEdit("100")
        self.retrain_min_samples.setFixedWidth(60)
        self.retrain_min_samples.setStyleSheet("padding: 4px; border: 1px solid #bdc3c7; border-radius: 3px;")
        retrain_layout.addWidget(self.retrain_min_samples)

        retrain_layout.addWidget(QLabel("Epoch："))
        self.retrain_epochs = QLineEdit("5")
        self.retrain_epochs.setFixedWidth(50)
        self.retrain_epochs.setStyleSheet("padding: 4px; border: 1px solid #bdc3c7; border-radius: 3px;")
        retrain_layout.addWidget(self.retrain_epochs)

        self.batch_retrain_info = QLabel("未训练")
        self.batch_retrain_info.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        retrain_layout.addWidget(self.batch_retrain_info)
        retrain_layout.addStretch()
        main_layout.addWidget(group_retrain)

        # ========== 步骤7：导出 ==========
        group_export = self._create_group_box("步骤7：数据导出")
        export_layout = QHBoxLayout(group_export)
        btn_export = QPushButton("💾 导出标注数据（CSV）")
        btn_export.setStyleSheet("padding: 8px 20px;")
        btn_export.clicked.connect(self._on_batch_export)
        export_layout.addWidget(btn_export)
        self.batch_export_info = QLabel("未导出")
        self.batch_export_info.setStyleSheet("color: #7f8c8d; margin-left: 10px;")
        export_layout.addWidget(self.batch_export_info)
        export_layout.addStretch()
        main_layout.addWidget(group_export)

        main_layout.addStretch()
        self._on_batch_refresh_status()

    def _create_group_box(self, title):
        """创建统一风格的分组框"""
        group = QGroupBox(title)
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                color: #2c3e50;
                border: 1px solid #bdc3c7;
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 8px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 6px;
            }
        """)
        return group

    # ---------- 窗口控制方法 ----------

    def _zoom_in(self):
        """放大窗口（每次增加100x80像素）"""
        geo = self.geometry()
        new_width = min(geo.width() + 100, 1920)  # 最大1920
        new_height = min(geo.height() + 80, 1080)  # 最大1080
        self.setGeometry(geo.x(), geo.y(), new_width, new_height)

    def _zoom_out(self):
        """缩小窗口（每次减少100x80像素）"""
        geo = self.geometry()
        new_width = max(geo.width() - 100, 800)  # 最小800
        new_height = max(geo.height() - 80, 600)  # 最小600
        self.setGeometry(geo.x(), geo.y(), new_width, new_height)

    def _toggle_fullscreen(self):
        """切换全屏/正常模式"""
        if self.isFullScreen():
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self.showFullScreen()

    def keyPressEvent(self, event):
        """按 Esc 退出全屏"""
        if event.key() == Qt.Key_Escape and self.isFullScreen():
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
        else:
            super().keyPressEvent(event)

    # ---------- 信号槽方法 ----------

    def _on_batch_refresh_status(self):
        try:
            stats = analyze_annotation_quality()
            fb_stats = get_feedback_stats()
            text = (f"📊 批量标注：总计 {stats['total']} 条 | 已标注 {stats['annotated']} 条 | "
                    f"不确定 {stats['uncertain']} 条 | BERT-LLM一致 {stats['llm_agreement']}/{stats['llm_total']} | "
                    f"医生反馈：{fb_stats['total']} 条（分歧 {fb_stats['disagreements']} 条）")
            self.batch_status_label.setText(text)
        except Exception as e:
            self.batch_status_label.setText(f"📊 状态刷新失败：{e}")

    def _on_batch_import_records(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择脱密病历文件", "",
            "CSV/Excel (*.csv *.xlsx *.xls);;所有文件 (*)"
        )
        if not file_path:
            return
        self.batch_import_info.setText("⏳ 正在导入...")
        self._batch_import_thread = BatchImportThread(file_path)
        self._batch_import_thread.finished_signal.connect(self._on_batch_import_finished)
        self._batch_import_thread.start()

    def _on_batch_import_finished(self, imported, skipped, message):
        self.batch_import_info.setText(message)
        QMessageBox.information(self, "导入完成", message)
        self._on_batch_refresh_status()

    def _on_batch_auto_llm(self):
        reply = QMessageBox.question(
            self, "确认", "将对未标注记录调用 LLM 进行详细预标注，这可能需要较长时间。\n\n是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.batch_llm_info.setText("⏳ LLM 预标注中...")
        self._batch_llm_thread = BatchLLMThread()
        self._batch_llm_thread.finished_signal.connect(self._on_batch_llm_finished)
        self._batch_llm_thread.start()

    def _on_batch_llm_finished(self, annotated, message):
        self.batch_llm_info.setText(message)
        QMessageBox.information(self, "LLM 预标注", message)
        self._on_batch_refresh_status()

    def _on_batch_load_review(self):
        records = get_records_for_review(limit=50, priority_uncertain=True)
        self.review_table.setRowCount(0)
        self._review_records_map = {}  # id -> patient_text
        if not records:
            QMessageBox.information(self, "提示", "没有待审核的记录，请先导入病历。")
            return
        for rec in records:
            row = self.review_table.rowCount()
            self.review_table.insertRow(row)
            self.review_table.setItem(row, 0, QTableWidgetItem(str(rec["id"])))
            self.review_table.setItem(row, 1, QTableWidgetItem(rec.get("patient_id", "")))
            self.review_table.setItem(row, 2, QTableWidgetItem(rec.get("auto_bert_pred", "")))
            conf = rec.get("auto_bert_conf", 0)
            self.review_table.setItem(row, 3, QTableWidgetItem(f"{conf:.3f}" if conf else "N/A"))
            self.review_table.setItem(row, 4, QTableWidgetItem(rec.get("auto_llm_conclusion", "未标注") or "未标注"))
            # 存储病历文本用于点击显示
            self._review_records_map[rec["id"]] = rec.get("patient_text", "")
            # 标记不确定样本
            if rec.get("is_uncertain", 0):
                for c in range(5):
                    self.review_table.item(row, c).setBackground(QColor("#fff3cd"))
        self.review_text_display.setPlainText("点击左侧表格中的某一行，此处将显示该患者的完整病历文本。")

    def _on_review_row_selected(self):
        """点击表格行时，在右侧显示病历文本"""
        selected = self.review_table.selectedItems()
        if not selected:
            return
        row = selected[0].row()
        record_id_item = self.review_table.item(row, 0)
        if record_id_item is None:
            return
        record_id = int(record_id_item.text())
        text = self._review_records_map.get(record_id, "")
        if text:
            self.review_text_display.setPlainText(text)
        else:
            self.review_text_display.setPlainText("（该记录暂无病历文本）")

    def _on_batch_label_selected(self, label):
        selected = self.review_table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "提示", "请先选中一行记录再标注。")
            return
        row = selected[0].row()
        record_id_item = self.review_table.item(row, 0)
        if record_id_item is None:
            return
        record_id = int(record_id_item.text())
        annotator = self.review_annotator_input.text().strip() or "未知医生"
        try:
            submit_manual_label(record_id, label, annotator=annotator)
            self._review_records_map.pop(record_id, None)
            self.review_table.removeRow(row)
            self.review_text_display.setPlainText("已标注，请选择下一条记录。")
            self._on_batch_refresh_status()
        except Exception as e:
            QMessageBox.critical(self, "标注失败", f"保存标注失败：{str(e)}")

    def _on_batch_analyze(self):
        try:
            stats = analyze_annotation_quality()
            report = (f"📊 批量标注数据质量报告\n"
                      f"━━━━━━━━━━━━━━━━━━━━\n"
                      f"总记录数：{stats['total']}\n"
                      f"已人工标注：{stats['annotated']} ({stats['annotated']/max(stats['total'],1)*100:.1f}%)\n"
                      f"BERT 不确定样本：{stats['uncertain']}\n")
            if stats['llm_total'] > 0:
                report += f"BERT-LLM 一致性：{stats['llm_agreement']}/{stats['llm_total']} ({stats['llm_agreement']/max(stats['llm_total'],1)*100:.1f}%)\n"
            report += "━━━━━━━━━━━━━━━━━━━━"
            self.batch_analyze_info.setText(f"已分析：{stats['annotated']}/{stats['total']} 已标注")
            QMessageBox.information(self, "标注质量分析", report)
        except Exception as e:
            QMessageBox.critical(self, "分析失败", f"{str(e)}")

    def _on_batch_retrain(self):
        try:
            min_samples = int(self.retrain_min_samples.text())
            epochs = int(self.retrain_epochs.text())
        except ValueError:
            QMessageBox.warning(self, "参数错误", "样本数和 Epoch 必须是整数。")
            return
        reply = QMessageBox.question(
            self, "确认重训练",
            f"将使用反馈数据 + 批量标注数据进行全量重训练。\n"
            f"最少样本：{min_samples} | Epoch：{epochs}\n\n"
            f"训练过程可能需要较长时间，是否继续？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.batch_retrain_info.setText("⏳ 训练中...")
        self._batch_retrain_thread = BatchRetrainThread(min_samples, epochs)
        self._batch_retrain_thread.finished_signal.connect(self._on_batch_retrain_finished)
        self._batch_retrain_thread.start()

    def _on_batch_retrain_finished(self, model_path, message):
        self.batch_retrain_info.setText(message)
        if model_path and os.path.exists(model_path):
            reply = QMessageBox.question(
                self, "训练完成",
                f"{message}\n\n是否立即切换到新模型？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                reload_bert_model(model_path)
                QMessageBox.information(self, "模型切换", "已切换到新训练的模型！")
        else:
            QMessageBox.information(self, "训练结果", message)
        self._on_batch_refresh_status()

    def _on_batch_export(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self, "导出标注数据", "annotated_data.csv", "CSV 文件 (*.csv)"
        )
        if not file_path:
            return
        try:
            path = export_annotated_data(file_path)
            self.batch_export_info.setText(f"已导出：{os.path.basename(path)}")
            QMessageBox.information(self, "导出成功", f"已导出到：\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"{str(e)}")


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
