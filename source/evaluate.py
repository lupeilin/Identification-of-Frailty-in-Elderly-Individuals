import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

# ---------- 加载模型 ----------
MODEL_PATH = "./frailty_bert_demo_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH).to(device)
model.eval()

# ---------- 加载真实数据 ----------
df = pd.read_csv("real_annotated_records.csv")   # 含 "text" 和 "label" 两列

# ---------- 批量预测 ----------
def predict_proba(texts):
    inputs = tokenizer(list(texts), padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[:, 1]   # 衰弱类的概率
    return probs.cpu().numpy()

# 计算概率（用于AUC）
probs = predict_proba(df["text"])

# 计算二值预测（阈值0.5）
preds = (probs > 0.5).astype(int)

# ---------- 指标计算 ----------
acc = accuracy_score(df["label"], preds)
precision = precision_score(df["label"], preds)
recall = recall_score(df["label"], preds)
f1 = f1_score(df["label"], preds)
auc = roc_auc_score(df["label"], probs)
cm = confusion_matrix(df["label"], preds)

print("="*50)
print("模型性能评估报告")
print("="*50)
print(f"准确率 (Accuracy):  {acc:.4f}")
print(f"精确率 (Precision): {precision:.4f}")
print(f"召回率 (Recall):    {recall:.4f}")
print(f"F1分数 (F1-Score):  {f1:.4f}")
print(f"AUC:               {auc:.4f}")
print("\n混淆矩阵:")
print("         预测非衰弱  预测衰弱")
print(f"实际非衰弱    {cm[0,0]:5d}      {cm[0,1]:5d}")
print(f"实际衰弱      {cm[1,0]:5d}      {cm[1,1]:5d}")
print("\n详细分类报告:")
print(classification_report(df["label"], preds, target_names=["非衰弱", "衰弱"]))