#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
batch_annotation_tool.py
批量病例标注与模型重训练工具
用法：python batch_annotation_tool.py
功能：
  1. 批量导入脱密病历（CSV/Excel/txt）
  2. 自动预标注（BERT + LLM）
  3. 人工审核修正
  4. 一键重训练 BERT（全量训练，非增量）
  5. 融合反馈数据 + 批量标注数据联合训练
"""

import os
import sys
import json
import sqlite3
import hashlib
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

import pandas as pd
import numpy as np
from datetime import datetime

# ================= 配置（避免循环导入，直接定义） =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BERT_MODEL_PATH = os.path.join(BASE_DIR, "frailty_bert_demo_model")
FEEDBACK_DB_PATH = os.path.join(BASE_DIR, "feedback.db")
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, "feedback_log.jsonl")

BATCH_ANNOTATION_DB = os.path.join(BASE_DIR, "batch_annotation.db")
BATCH_RAW_DIR = os.path.join(BASE_DIR, "batch_raw")  # 存放原始脱密病历


# 延迟导入主应用函数，避免循环导入
_frailty_app = None

def _get_frailty_app():
    global _frailty_app
    if _frailty_app is None:
        import frailty_assessment_app as _frailty_app
    return _frailty_app


def init_batch_annotation_db():
    """初始化批量标注数据库"""
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS batch_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_file TEXT NOT NULL,
            patient_id TEXT,
            patient_text TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            -- 自动预标注结果
            auto_bert_pred TEXT,
            auto_bert_conf REAL,
            auto_llm_conclusion TEXT,
            auto_llm_positive_count INTEGER,
            -- 人工审核结果
            manual_label TEXT,  -- 衰弱/衰弱前期/非衰弱/未标注
            manual_fried_items TEXT,  -- JSON: [{"name": "...", "result": "..."}, ...]
            annotated_by TEXT,
            annotated_at TEXT,
            -- 数据质量标记
            is_uncertain INTEGER DEFAULT 0,  -- 1=模型不确定，需要优先审核
            is_high_quality INTEGER DEFAULT 0,  -- 1=高质量标注，可用于训练
            -- 时间戳
            created_at TEXT NOT NULL,
            UNIQUE(text_hash)
        )
    """)
    conn.commit()
    conn.close()
    print(f"✅ 批量标注数据库初始化完成：{BATCH_ANNOTATION_DB}")


# ================= 步骤1：批量导入脱密病历 =================

def import_batch_records(file_path):
    """
    导入脱密病历文件（CSV/Excel/txt 文件夹）。
    支持的列名：patient_id, patient_text, text, 病历文本, 入院记录...
    """
    records = []
    
    if file_path.lower().endswith('.csv'):
        df = pd.read_csv(file_path, dtype=str).fillna("")
    elif file_path.lower().endswith(('.xlsx', '.xls')):
        df = pd.read_excel(file_path, dtype=str).fillna("")
    else:
        raise ValueError("仅支持 CSV/Excel 文件")
    
    # 列名自动匹配
    text_cols = ['patient_text', 'text', '病历文本', '入院记录', '病情描述', '病历内容']
    id_cols = ['patient_id', 'id', '住院号', '病历号', '患者ID']
    
    text_col = None
    for col in text_cols:
        if col in df.columns:
            text_col = col
            break
    
    id_col = None
    for col in id_cols:
        if col in df.columns:
            id_col = col
            break
    
    if text_col is None:
        raise ValueError(f"未找到病历文本列，可用列名：{list(df.columns)}")
    
    # 加载 BERT 用于预标注
    print("⏳ 加载 BERT 模型进行预标注...")
    app = _get_frailty_app()
    tokenizer, model, device = app.load_bert_model()
    
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    imported = 0
    skipped = 0
    
    for idx, row in df.iterrows():
        text = str(row[text_col]).strip()
        if not text or len(text) < 20:
            skipped += 1
            continue
        
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        
        # 检查是否已存在
        cursor.execute("SELECT COUNT(*) FROM batch_records WHERE text_hash = ?", (text_hash,))
        if cursor.fetchone()[0] > 0:
            skipped += 1
            continue
        
        # BERT 预标注
        bert_result = app.predict_frailty(text, tokenizer, model, device)
        
        # 标记不确定样本（置信度 < 0.7）
        is_uncertain = 1 if bert_result['confidence'] < 0.7 else 0
        
        pid = str(row.get(id_col, '')) if id_col else f"BATCH_{idx:05d}"
        
        cursor.execute("""
            INSERT INTO batch_records 
            (source_file, patient_id, patient_text, text_hash,
             auto_bert_pred, auto_bert_conf, manual_label,
             is_uncertain, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            os.path.basename(file_path), pid, text, text_hash,
            bert_result['prediction'], bert_result['confidence'], '未标注',
            is_uncertain, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        ))
        imported += 1
        
        if (idx + 1) % 100 == 0:
            print(f"  已处理 {idx + 1}/{len(df)} 条，导入 {imported} 条")
    
    conn.commit()
    conn.close()
    print(f"✅ 导入完成：新增 {imported} 条，跳过 {skipped} 条（已存在或无效）")
    return imported


# ================= 步骤2：自动预标注（LLM 详细评估） =================

def auto_annotate_with_llm(batch_size=50, limit=None):
    """
    对未标注的批量记录调用 LLM 进行详细预标注。
    优先处理不确定样本（置信度 < 0.7）。
    """
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    
    # 先处理不确定样本
    cursor.execute("""
        SELECT id, patient_text, auto_bert_pred 
        FROM batch_records 
        WHERE auto_llm_conclusion IS NULL
        ORDER BY is_uncertain DESC, auto_bert_conf ASC
        LIMIT ?
    """, (limit or 999999,))
    
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("⚠️ 没有需要 LLM 预标注的记录")
        return 0
    
    print(f"🚀 开始 LLM 预标注：{len(rows)} 条记录...")
    annotated = 0
    
    for rec_id, text, bert_pred in rows:
        try:
            # 调用 LLM 生成详细解释
            app = _get_frailty_app()
            llm_result = app.generate_explanation(text, bert_pred, 0.8)
            
            if "error" in llm_result:
                print(f"  ⚠️ 记录 {rec_id} LLM 标注失败：{llm_result['error']}")
                continue
            
            conclusion = llm_result.get('conclusion', '未知')
            positive_count = llm_result.get('positive_count', 0)
            
            conn = sqlite3.connect(BATCH_ANNOTATION_DB)
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE batch_records 
                SET auto_llm_conclusion = ?, auto_llm_positive_count = ?
                WHERE id = ?
            """, (conclusion, positive_count, rec_id))
            conn.commit()
            conn.close()
            annotated += 1
            
            if annotated % 10 == 0:
                print(f"  已标注 {annotated}/{len(rows)} 条")
                
        except Exception as e:
            print(f"  ❌ 记录 {rec_id} 处理异常：{e}")
    
    print(f"✅ LLM 预标注完成：{annotated} 条")
    return annotated


# ================= 步骤3：人工审核标注 =================

def get_records_for_review(limit=50, priority_uncertain=True):
    """
    获取待审核的记录，优先返回不确定样本。
    返回列表，每条记录包含所有预标注信息。
    """
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    
    order_by = "is_uncertain DESC, auto_bert_conf ASC" if priority_uncertain else "created_at"
    
    cursor.execute(f"""
        SELECT id, patient_id, patient_text, 
               auto_bert_pred, auto_bert_conf, auto_llm_conclusion,
               manual_label, is_uncertain
        FROM batch_records 
        WHERE manual_label = '未标注'
        ORDER BY {order_by}
        LIMIT ?
    """, (limit,))
    
    rows = cursor.fetchall()
    conn.close()
    
    return [
        {
            "id": r[0],
            "patient_id": r[1],
            "patient_text": r[2],
            "auto_bert_pred": r[3],
            "auto_bert_conf": r[4],
            "auto_llm_conclusion": r[5],
            "manual_label": r[6],
            "is_uncertain": r[7]
        }
        for r in rows
    ]


def submit_manual_label(record_id, manual_label, fried_items=None, annotator=""):
    """
    提交人工审核标签。
    manual_label: 衰弱/衰弱前期/非衰弱
    fried_items: JSON 字符串，如 [{"name": "体重下降", "result": "阳性"}, ...]
    """
    is_high_quality = 1  # 人工审核的都标记为高质量
    
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    cursor.execute("""
        UPDATE batch_records 
        SET manual_label = ?, manual_fried_items = ?, annotated_by = ?,
            annotated_at = ?, is_high_quality = ?
        WHERE id = ?
    """, (
        manual_label, 
        json.dumps(fried_items, ensure_ascii=False) if fried_items else None,
        annotator,
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        is_high_quality,
        record_id
    ))
    conn.commit()
    conn.close()
    
    # 同时写入反馈数据库（兼容持续学习流程）
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT patient_text, auto_bert_pred, auto_bert_conf FROM batch_records WHERE id = ?", (record_id,))
    row = cursor.fetchone()
    conn.close()
    
    if row:
        text, bert_pred, bert_conf = row
        # 写入反馈数据库，标记为批量标注来源
        app = _get_frailty_app()
        app.save_feedback(
            patient_text=text,
            bert_pred=bert_pred or "未知",
            bert_conf=bert_conf or 0.0,
            llm_conclusion="未知",
            llm_positive_count=0,
            llm_suggestions="",
            doctor_correction=manual_label,
            correction_reason="批量标注审核",
            doctor_name=annotator or "batch_annotator"
        )
    
    return True


# ================= 步骤4：一键重训练 BERT =================

def prepare_combined_training_data(min_samples=100, use_feedback=True, use_batch=True):
    """
    准备训练数据：融合反馈数据 + 批量标注数据。
    只取高质量标注（人工审核过的批量标注 + 医生反馈）。
    """
    datasets = []
    
    # 1. 批量标注数据（人工审核过的）
    if use_batch:
        conn = sqlite3.connect(BATCH_ANNOTATION_DB)
        df_batch = pd.read_sql_query("""
            SELECT patient_text, manual_label 
            FROM batch_records 
            WHERE manual_label != '未标注' AND is_high_quality = 1
        """, conn)
        conn.close()
        
        if len(df_batch) > 0:
            label_map = {"非衰弱": 0, "衰弱前期": 1, "衰弱": 1}
            df_batch["label"] = df_batch["manual_label"].map(label_map)
            df_batch = df_batch.dropna(subset=["label"])
            df_batch["label"] = df_batch["label"].astype(int)
            datasets.append(df_batch[["patient_text", "label"]])
            print(f"📊 批量标注数据：{len(df_batch)} 条")
    
    # 2. 反馈数据（医生校正的）
    if use_feedback:
        conn = sqlite3.connect(FEEDBACK_DB_PATH)
        df_feedback = pd.read_sql_query("""
            SELECT patient_text, doctor_correction 
            FROM feedback 
            WHERE doctor_correction != bert_prediction
        """, conn)
        conn.close()
        
        if len(df_feedback) > 0:
            label_map = {"非衰弱": 0, "衰弱前期": 1, "衰弱": 1}
            df_feedback["label"] = df_feedback["doctor_correction"].map(label_map)
            df_feedback = df_feedback.dropna(subset=["label"])
            df_feedback["label"] = df_feedback["label"].astype(int)
            datasets.append(df_feedback[["patient_text", "label"]])
            print(f"📊 医生反馈数据：{len(df_feedback)} 条")
    
    if not datasets:
        print("⚠️ 没有可用的训练数据")
        return None
    
    # 合并
    df_combined = pd.concat(datasets, ignore_index=True)
    
    if len(df_combined) < min_samples:
        print(f"⚠️ 总样本仅 {len(df_combined)} 条，建议积累到 {min_samples} 条再训练")
        return None
    
    # 去重（相同文本保留最新标注）
    df_combined = df_combined.drop_duplicates(subset=["patient_text"], keep="last")
    
    print(f"✅ 合并训练数据：{len(df_combined)} 条（去重后）")
    print(f"   衰弱类（1）：{(df_combined['label'] == 1).sum()} 条")
    print(f"   非衰弱类（0）：{(df_combined['label'] == 0).sum()} 条")
    
    return df_combined


def full_retrain_bert(min_samples=100, output_dir=None, epochs=5, unfreeze_layers=3):
    """
    全量重训练 BERT（非增量微调）。
    使用反馈数据 + 批量标注数据联合训练。
    解冻部分底层，比增量微调更充分适应医院数据分布。
    
    Args:
        min_samples: 最少样本数
        output_dir: 输出目录，默认自动生成版本号
        epochs: 训练轮数（全量训练可适当增加）
        unfreeze_layers: 解冻最后 N 层 Transformer（默认3层）
    """
    if output_dir is None:
        v = 2
        while os.path.exists(os.path.join(BASE_DIR, f"frailty_bert_retrained_v{v}")):
            v += 1
        output_dir = os.path.join(BASE_DIR, f"frailty_bert_retrained_v{v}")
    
    df = prepare_combined_training_data(min_samples)
    if df is None:
        return None
    
    print(f"🚀 开始全量重训练 BERT，使用 {len(df)} 条样本，{epochs} epoch...")
    
    # 加载预训练模型（从原始 bert-base-chinese 或当前最佳模型）
    base_model = BERT_MODEL_PATH
    if not os.path.exists(base_model):
        base_model = "bert-base-chinese"
    
    tokenizer = BertTokenizer.from_pretrained(base_model)
    model = BertForSequenceClassification.from_pretrained(base_model, num_labels=2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 解冻策略：解冻最后 N 层 Transformer + 分类头
    # 先冻结所有
    for param in model.bert.parameters():
        param.requires_grad = False
    
    # 解冻最后 unfreeze_layers 层
    if unfreeze_layers > 0:
        for layer in model.bert.encoder.layer[-unfreeze_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        print(f"   解冻最后 {unfreeze_layers} 层 Transformer")
    
    # 分类头始终训练
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # 数据准备
    train_df, eval_df = train_test_split(df, test_size=0.15, random_state=42, stratify=df["label"])
    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "eval": Dataset.from_pandas(eval_df)
    })
    
    def tokenize(batch):
        return tokenizer(batch["patient_text"], padding="max_length", truncation=True, max_length=256)
    
    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    
    # 训练参数（全量训练，学习率稍大，epoch 更多）
    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "frailty_bert_retrain"),
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=epochs,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=50,
    )
    
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = logits.argmax(axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
        acc = accuracy_score(labels, predictions)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["eval"],
        compute_metrics=compute_metrics,
    )
    
    trainer.train()
    
    # 评估
    eval_results = trainer.evaluate()
    print(f"\n📊 评估结果：")
    for k, v in eval_results.items():
        print(f"   {k}: {v:.4f}")
    
    # 保存
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # 保存训练报告
    report = {
        "output_dir": output_dir,
        "base_model": base_model,
        "total_samples": len(df),
        "train_samples": len(train_df),
        "eval_samples": len(eval_df),
        "epochs": epochs,
        "unfreeze_layers": unfreeze_layers,
        "eval_results": {k: float(v) for k, v in eval_results.items()},
        "training_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(os.path.join(output_dir, "training_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 全量重训练完成，保存至 {output_dir}")
    print(f"   训练报告：{os.path.join(output_dir, 'training_report.json')}")
    return output_dir


# ================= 步骤5：数据质量分析与导出 =================

def analyze_annotation_quality():
    """分析批量标注数据质量"""
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    
    # 总体统计
    total = pd.read_sql_query("SELECT COUNT(*) as cnt FROM batch_records", conn).iloc[0]['cnt']
    annotated = pd.read_sql_query("SELECT COUNT(*) as cnt FROM batch_records WHERE manual_label != '未标注'", conn).iloc[0]['cnt']
    uncertain = pd.read_sql_query("SELECT COUNT(*) as cnt FROM batch_records WHERE is_uncertain = 1", conn).iloc[0]['cnt']
    
    # BERT vs LLM 一致性
    agreement = pd.read_sql_query("""
        SELECT COUNT(*) as cnt 
        FROM batch_records 
        WHERE auto_bert_pred = auto_llm_conclusion 
          AND auto_llm_conclusion IS NOT NULL
    """, conn).iloc[0]['cnt']
    
    llm_total = pd.read_sql_query("SELECT COUNT(*) as cnt FROM batch_records WHERE auto_llm_conclusion IS NOT NULL", conn).iloc[0]['cnt']
    
    conn.close()
    
    print("\n📊 批量标注数据质量报告")
    print("=" * 40)
    print(f"总记录数：{total}")
    print(f"已人工标注：{annotated} ({annotated/total*100:.1f}%)")
    print(f"BERT 不确定样本：{uncertain}")
    if llm_total > 0:
        print(f"BERT-LLM 一致性：{agreement}/{llm_total} ({agreement/llm_total*100:.1f}%)")
    print("=" * 40)
    
    return {
        "total": total,
        "annotated": annotated,
        "uncertain": uncertain,
        "llm_agreement": agreement,
        "llm_total": llm_total
    }


def export_annotated_data(csv_path=None, only_high_quality=True):
    """导出已标注数据为 CSV，供外部工具分析"""
    if csv_path is None:
        csv_path = os.path.join(BASE_DIR, f"annotated_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    conn = sqlite3.connect(BATCH_ANNOTATION_DB)
    
    if only_high_quality:
        df = pd.read_sql_query("""
            SELECT * FROM batch_records 
            WHERE manual_label != '未标注' AND is_high_quality = 1
        """, conn)
    else:
        df = pd.read_sql_query("SELECT * FROM batch_records", conn)
    
    conn.close()
    
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"✅ 已导出 {len(df)} 条记录到 {csv_path}")
    return csv_path


# ================= 命令行入口 =================

def print_usage():
    print("""
批量病例标注与模型重训练工具

用法：python batch_annotation_tool.py <命令> [参数]

命令：
  import <文件路径>          导入脱密病历（CSV/Excel）
  auto-llm [数量]            对未标注记录进行 LLM 预标注
  review [数量]            获取待审核记录列表（JSON 输出）
  label <id> <标签> [姓名]   提交人工审核标签
  retrain [样本数] [epoch]  全量重训练 BERT
  analyze                  分析标注数据质量
  export [路径]            导出已标注数据
  status                   查看当前状态

示例：
  python batch_annotation_tool.py import /path/to/hospital_records.csv
  python batch_annotation_tool.py auto-llm 100
  python batch_annotation_tool.py review 10
  python batch_annotation_tool.py label 42 衰弱 张医生
  python batch_annotation_tool.py retrain 200 5
  python batch_annotation_tool.py analyze
    """)


def main():
    if len(sys.argv) < 2:
        print_usage()
        return
    
    command = sys.argv[1]
    
    # 初始化数据库
    init_batch_annotation_db()
    
    if command == "import":
        if len(sys.argv) < 3:
            print("❌ 请指定文件路径")
            return
        import_batch_records(sys.argv[2])
    
    elif command == "auto-llm":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
        auto_annotate_with_llm(limit=limit)
    
    elif command == "review":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        records = get_records_for_review(limit=limit)
        print(json.dumps(records, ensure_ascii=False, indent=2))
    
    elif command == "label":
        if len(sys.argv) < 4:
            print("❌ 用法：label <id> <衰弱/衰弱前期/非衰弱> [审核人姓名]")
            return
        record_id = int(sys.argv[2])
        label = sys.argv[3]
        annotator = sys.argv[4] if len(sys.argv) > 4 else ""
        submit_manual_label(record_id, label, annotator=annotator)
        print(f"✅ 记录 {record_id} 已标注为：{label}")
    
    elif command == "retrain":
        min_samples = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        epochs = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        full_retrain_bert(min_samples=min_samples, epochs=epochs)
    
    elif command == "analyze":
        analyze_annotation_quality()
    
    elif command == "export":
        path = sys.argv[2] if len(sys.argv) > 2 else None
        export_annotated_data(path)
    
    elif command == "status":
        stats = analyze_annotation_quality()
        app = _get_frailty_app()
        fb_stats = app.get_feedback_stats()
        print(f"\n医生反馈：总计 {fb_stats['total']}，分歧 {fb_stats['disagreements']}")
    
    else:
        print(f"❌ 未知命令：{command}")
        print_usage()


if __name__ == "__main__":
    main()
