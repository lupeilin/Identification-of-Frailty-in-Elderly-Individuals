import requests
import json


def generate_clinical_explanation(text, bert_pred, bert_conf):
    """
    text: 病历原文
    bert_pred: BERT预测的类别（"衰弱" / "非衰弱"）
    bert_conf: BERT预测的置信度
    """
    # 医学知识库（固定背景）
    knowledge = """
Fried衰弱表型标准（5项，≥3项为衰弱）：
1. 体重下降：1年内下降≥5%或≥4.5kg
2. 疲乏/精力减退
3. 握力下降
4. 步速减慢
5. 活动量减少
"""

    # 构造提示词，把BERT的预测结果作为“初步判断”告诉LLM
    prompt = f"""你是一位老年医学专家。请参考下面的医学知识和辅助诊断模型的初步判断，对患者进行详细分析。

医学知识：{knowledge}

辅助诊断模型判断结果：{bert_pred}（置信度 {bert_conf:.2%}）

患者信息：{text}

请按以下格式逐一分析：
1. 逐项核对Fried五项指标（明确：阳性/阴性，并给出证据）
2. 阳性指标总数
3. 最终评估结论（衰弱/衰弱前期/非衰弱）
4. 临床建议
"""

    # 调用本地 Ollama API（默认端口11434）
    ollama_url = "http://localhost:11434/api/generate"
    payload = {
        "model": "qwen2.5:3b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.1}  # 低温度保证回答稳定
    }

    response = requests.post(ollama_url, json=payload)
    if response.status_code == 200:
        return response.json()["response"]
    else:
        return f"LLM调用失败：{response.text}"


# 使用示例（假设 bert_model 已加载）：
explanation = generate_clinical_explanation(
    "患者女性，78岁，近半年体重下降6kg，自感疲乏无力，步速减慢。",
    "衰弱",
    0.998
)
print(explanation)