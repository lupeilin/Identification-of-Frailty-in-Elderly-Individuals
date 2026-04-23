# 加载预训练模型（二分类）
from transformers import BertForSequenceClassification, TrainingArguments, Trainer

from source.dataProcessAndLoad import tokenized_datasets, tokenizer

model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# 训练参数配置
training_args = TrainingArguments(
    output_dir="./frailty_bert_results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)

# 定义评估指标
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="binary")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

# 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

# 开始训练
trainer.train()

# 评估模型
eval_results = trainer.evaluate()
print("评估结果：", eval_results)

# 保存模型
model.save_pretrained("./frailty_bert_model")
tokenizer.save_pretrained("./frailty_bert_model")