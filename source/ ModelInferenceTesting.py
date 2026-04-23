# 加载训练好的模型
import torch
from transformers import BertForSequenceClassification, BertTokenizer

model = BertForSequenceClassification.from_pretrained("./frailty_bert_model")
tokenizer = BertTokenizer.from_pretrained("./frailty_bert_model")
model.eval()

def predict_frailty(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    return {
        "prediction": "衰弱" if pred == 1 else "非衰弱",
        "confidence": probs[0][pred].item()
    }

# 测试
test_text = "患者女性，78岁。主诉：近半年体重下降约6kg，自感疲乏无力，步速减慢。既往史：高血压病史15年。"
result = predict_frailty(test_text)
print(result)
# 预期输出：{'prediction': '衰弱', 'confidence': 0.98...}