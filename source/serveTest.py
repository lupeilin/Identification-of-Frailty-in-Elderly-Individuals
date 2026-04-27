import requests

# 测试普通预测接口
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"}
)
print("=== /predict 结果 ===")
print(response.json())

# 测试带RAG解释的接口
response2 = requests.post(
    "http://localhost:8000/predict_with_explanation",
    json={"text": "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"}
)
print("\n=== /predict_with_explanation 结果 ===")
print(response2.json())