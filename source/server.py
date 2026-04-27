import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import requests
from transformers import BertTokenizer, BertForSequenceClassification

# ========== 1. 初始化FastAPI应用 ==========
app = FastAPI(title="老年衰弱识别系统", version="2.0")

# ========== 2. 加载BERT模型（全局缓存） ==========
MODEL_PATH = "./frailty_bert_demo_model"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(device)
model.eval()


# ========== 3. 定义数据模型 ==========
class PatientRecord(BaseModel):
    text: str  # 病历文本


class PredictionResult(BaseModel):
    prediction: str
    confidence: float
    input_text: str


class ExplanationResult(BaseModel):
    prediction: str
    confidence: float
    explanation: str  # RAG生成的详细解释


# ========== 4. BERT直接预测接口 ==========
@app.post("/predict", response_model=PredictionResult)
def predict_frailty(record: PatientRecord):
    if not record.text.strip():
        raise HTTPException(status_code=400, detail="输入文本不能为空")

    inputs = tokenizer(record.text, padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)
        cls_id = probs.argmax().item()
        confidence = probs[0][cls_id].item()

    return PredictionResult(
        prediction="衰弱" if cls_id == 1 else "非衰弱",
        confidence=round(confidence, 4),
        input_text=record.text
    )


# ========== 5. RAG解释生成函数 ==========
def generate_clinical_explanation(text: str, bert_pred: str, bert_conf: float) -> str:
    """调用本地LLM（通过Ollama API）生成衰弱的临床解释"""

    # 内嵌医学知识（也可以从外部文件读取）
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
预测结果：{bert_pred}
置信度：{bert_conf:.2%}

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
                "model": "qwen2.5:3b",  # 根据实际下载的模型名修改
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"[错误] LLM调用失败，状态码：{response.status_code}"
    except Exception as e:
        return f"[错误] 无法连接Ollama服务：{str(e)}"


# ========== 6. 带解释的预测接口 ==========
@app.post("/predict_with_explanation", response_model=ExplanationResult)
def predict_with_explanation(record: PatientRecord):
    # 先调用BERT获取预测结果
    basic = predict_frailty(record)

    # 再调用RAG生成解释
    explanation = generate_clinical_explanation(
        text=record.text,
        bert_pred=basic.prediction,
        bert_conf=basic.confidence
    )

    return ExplanationResult(
        prediction=basic.prediction,
        confidence=basic.confidence,
        explanation=explanation
    )

# ========== 启动方式 ==========
# uvicorn server:app --host 0.0.0.0 --port 8000