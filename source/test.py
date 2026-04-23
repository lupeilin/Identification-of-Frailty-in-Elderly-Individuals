import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"  # 保持镜像设置

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# ========== 1. 加载模型和分词器 ==========
model_path = "./frailty_bert_demo_model"  # 指向你保存的最终模型目录

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

# 将模型设置为评估模式（关闭 Dropout 等训练特有层）
model.eval()

# 如果有GPU，可以移到GPU上（可选）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

print(f"✅ 模型加载成功，运行在 {device}")


# ========== 2. 定义预测函数 ==========
def predict_frailty(text):
    """
    输入：病历文本字符串
    输出：字典，包含预测类别、置信度和原始logits
    """
    # 分词，转换为模型输入格式
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )

    # 将输入移到与模型相同的设备上
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 禁用梯度计算（推理时不需要梯度，节省内存）
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_class].item()

    return {
        "text": text,
        "prediction": "衰弱" if predicted_class == 1 else "非衰弱",
        "confidence": confidence,
        "logits": logits.squeeze().tolist()
    }


# ========== 3. 测试 ==========
if __name__ == "__main__":
    # 测试用例1：符合衰弱特征
    test_text_1 = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
    result_1 = predict_frailty(test_text_1)
    print("\n🔍 测试1：")
    print(f"输入：{result_1['text']}")
    print(f"预测：{result_1['prediction']} (置信度：{result_1['confidence']:.4f})")

    # 测试用例2：不符合衰弱特征
    test_text_2 = "患者男性，68岁，体重稳定，精神状态良好，行走自如，日常活动正常。"
    result_2 = predict_frailty(test_text_2)
    print("\n🔍 测试2：")
    print(f"输入：{result_2['text']}")
    print(f"预测：{result_2['prediction']} (置信度：{result_2['confidence']:.4f})")

    # 测试用例3：自定义输入（可以改成你自己的病历文本）
    custom_text = "患者，82岁，近一年体重下降5公斤，主诉全身没劲，走路需要人扶，平时不愿下床。"
    result_3 = predict_frailty(custom_text)
    print("\n🔍 测试3（自定义）：")
    print(f"输入：{result_3['text']}")
    print(f"预测：{result_3['prediction']} (置信度：{result_3['confidence']:.4f})")