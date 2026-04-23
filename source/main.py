"""
老年衰弱自动识别系统 - 完整演示脚本
包含：假数据生成 -> BERT微调 -> RAG问答
"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import pandas as pd
import random
import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support


# ========== 1. 生成假数据 ==========
def generate_fake_data():
    FRIED_INDICATORS = {
        "weight_loss": {
            "positive": ["近半年体重下降约6kg", "体重较前下降5公斤", "消瘦明显，半年体重减轻8%"],
            "negative": ["体重无明显变化", "体重稳定"]
        },
        "exhaustion": {
            "positive": ["自感疲乏无力", "精神差，易疲劳", "整天没精神"],
            "negative": ["精神状态可", "无明显疲乏感"]
        },
        "weakness": {
            "positive": ["握力下降", "双手无力", "拧毛巾困难"],
            "negative": ["肌力正常", "双手握力可"]
        },
        "slowness": {
            "positive": ["步速减慢", "行走缓慢", "步态不稳需搀扶"],
            "negative": ["步态平稳", "行走自如"]
        },
        "low_activity": {
            "positive": ["活动量减少", "长期卧床", "不愿下床活动"],
            "negative": ["日常活动可自理", "活动量正常"]
        }
    }

    data = []
    for i in range(100):
        # 衰弱组
        text = f"患者{random.choice(['男', '女'])}，{random.randint(70, 90)}岁。主诉："
        positives = random.sample(list(FRIED_INDICATORS.keys()), random.randint(3, 5))
        phrases = []
        for ind in positives:
            phrases.append(random.choice(FRIED_INDICATORS[ind]["positive"]))
        text += "，".join(phrases) + "。"
        data.append({"text": text, "label": 1})

        # 非衰弱组
        text = f"患者{random.choice(['男', '女'])}，{random.randint(65, 85)}岁。主诉："
        positives = random.sample(list(FRIED_INDICATORS.keys()), random.randint(0, 1))
        phrases = []
        for ind in FRIED_INDICATORS:
            if ind in positives:
                phrases.append(random.choice(FRIED_INDICATORS[ind]["positive"]))
            else:
                phrases.append(random.choice(FRIED_INDICATORS[ind]["negative"]))
        text += "，".join(random.sample(phrases, 3)) + "。"
        data.append({"text": text, "label": 0})

    df = pd.DataFrame(data)
    df.to_csv("frailty_demo_data.csv", index=False)
    print(f"✅ 生成{len(df)}条假数据")
    return df


# ========== 2. BERT微调 ==========
def train_bert_model(data_path="frailty_demo_data.csv"):
    df = pd.read_csv(data_path)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])

    dataset = DatasetDict({
        "train": Dataset.from_pandas(train_df),
        "test": Dataset.from_pandas(test_df)
    })

    # tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    model_name = "bert-base-chinese"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # model = BertForSequenceClassification.from_pretrained("bert-base-chinese", num_labels=2)

    training_args = TrainingArguments(
        output_dir="./frailty_bert_demo",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        # logging_dir="./logs",        # ← 删除或注释掉
        logging_steps=5,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = logits.argmax(-1)
        return {"accuracy": accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics,
    )

    print("🚀 开始训练BERT...")
    trainer.train()
    model.save_pretrained("./frailty_bert_demo_model")
    tokenizer.save_pretrained("./frailty_bert_demo_model")
    print("✅ BERT模型训练完成，已保存")
    return model, tokenizer

# ========== 3. RAG问答（简化版，无需真实向量库） ==========
def rag_demo():
    """演示RAG流程的核心逻辑（使用Prompt直接调用LLM）"""
    print("\n📚 RAG演示模式")
    print("=" * 50)

    # 模拟知识库内容
    knowledge = """
Fried衰弱表型标准：
1. 体重下降：1年内下降≥5%
2. 疲乏：自我报告疲乏感
3. 握力下降：握力计测量低于阈值
4. 步速减慢：行走4.57米时间延长
5. 活动量减少：每周消耗热量降低
满足≥3项为衰弱，1-2项为衰弱前期，0项为非衰弱。
"""

    test_case = "患者女性，78岁，近半年体重下降6kg，精神状态可，步速减慢。"

    # 构造提示
    prompt = f"""根据以下医学知识判断患者衰弱状态。

知识：{knowledge}

患者信息：{test_case}

请按JSON格式输出评估结果。"""

    print("患者信息：", test_case)
    print("\n提示词示例：")
    print(prompt[:300] + "...")
    print("\n💡 实际使用时，将调用本地LLM（如Qwen/Ollama）获取完整回答。")


# ========== 主函数 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("老年衰弱自动识别系统 - 演示流程")
    print("=" * 60)

    # Step 1: 生成假数据
    df = generate_fake_data()

    # Step 2: 训练BERT模型
    model, tokenizer = train_bert_model()

    # Step 3: 测试BERT推理
    print("\n🔍 BERT推理测试：")
    test_text = "患者女性，28岁，体重无明显变化，自感疲乏无力，步态平稳。"
    inputs = tokenizer(test_text, return_tensors="pt", truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        pred = outputs.logits.argmax(-1).item()
    print(f"输入：{test_text}")
    print(f"预测结果：{'衰弱' if pred == 1 else '非衰弱'}")

    # Step 4: RAG演示
    rag_demo()

    print("\n" + "=" * 60)
    print("✅ 完整流程演示结束")