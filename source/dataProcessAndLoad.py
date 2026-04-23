from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import pandas as pd
import torch

# 读取数据
df = pd.read_csv("frailty_fake_data.csv")

# 划分训练集和测试集
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

# 转换为HuggingFace Dataset格式
train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])
dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

# 加载tokenizer（使用中文BERT）
model_name = "bert-base-chinese"  # 也可以换成 "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext" 但中文场景建议用中文预训练模型
tokenizer = BertTokenizer.from_pretrained(model_name)

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        padding="max_length",
        truncation=True,
        max_length=256
    )

tokenized_datasets = dataset.map(tokenize_function, batched=True)