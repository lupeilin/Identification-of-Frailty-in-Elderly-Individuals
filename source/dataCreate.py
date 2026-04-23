import pandas as pd
import random
import json

# Fried衰弱表型5项指标及对应的病历文本描述模板
FRIED_INDICATORS = {
    "weight_loss": {
        "positive": ["近半年体重下降约6kg", "体重较前下降5公斤", "消瘦明显，半年体重减轻8%", "自述体重持续下降"],
        "negative": ["体重无明显变化", "体重稳定", "体重正常范围"]
    },
    "exhaustion": {
        "positive": ["自感疲乏无力", "精神差，易疲劳", "整天没精神", "自述浑身没劲"],
        "negative": ["精神状态可", "无明显疲乏感", "精力尚可"]
    },
    "weakness": {
        "positive": ["握力下降", "双手无力", "拧毛巾困难", "持物不稳"],
        "negative": ["肌力正常", "双手握力可", "肢体肌力V级"]
    },
    "slowness": {
        "positive": ["步速减慢", "行走缓慢", "步态不稳需搀扶", "走路费劲"],
        "negative": ["步态平稳", "行走自如", "活动灵活"]
    },
    "low_activity": {
        "positive": ["活动量减少", "长期卧床", "不愿下床活动", "日常活动需协助"],
        "negative": ["日常活动可自理", "活动量正常", "可独立行走"]
    }
}


# 生成假病历函数
def generate_fake_record(patient_id, is_frailty):
    """
    is_frailty: True表示衰弱（≥3项阳性），False表示非衰弱（≤1项阳性）
    """
    if is_frailty:
        # 衰弱患者：随机选择3-5项为阳性
        positive_count = random.randint(3, 5)
    else:
        # 非衰弱患者：随机选择0-1项为阳性
        positive_count = random.randint(0, 1)

    # 随机决定哪些指标为阳性
    indicators = list(FRIED_INDICATORS.keys())
    positive_indicators = set(random.sample(indicators, positive_count))

    # 构建病历文本
    text_parts = [f"患者{random.choice(['男', '女'])}，{random.randint(65, 92)}岁。"]

    # 添加主诉和现病史
    chief_complaint = []
    for ind in indicators:
        if ind in positive_indicators:
            phrase = random.choice(FRIED_INDICATORS[ind]["positive"])
        else:
            phrase = random.choice(FRIED_INDICATORS[ind]["negative"])
        chief_complaint.append(phrase)

    text_parts.append("主诉：" + "，".join(random.sample(chief_complaint, random.randint(2, 4))) + "。")

    # 添加既往史（随机慢性病）
    comorbidities = ["高血压", "糖尿病", "冠心病", "慢性心衰", "COPD", "骨关节炎"]
    text_parts.append(f"既往史：{random.choice(comorbidities)}病史{random.randint(3, 20)}年。")

    # 添加体格检查
    text_parts.append(f"查体：BP {random.randint(110, 160)}/{random.randint(60, 95)}mmHg，")
    text_parts.append(f"HR {random.randint(60, 100)}bpm，BMI {round(random.uniform(18, 30), 1)}kg/m²。")

    return "".join(text_parts), list(positive_indicators)


# 生成数据集
random.seed(42)
data = []
for i in range(200):  # 200条数据
    is_frailty = i < 100  # 前100条为衰弱，后100条为非衰弱
    text, positives = generate_fake_record(f"P{i:04d}", is_frailty)
    data.append({
        "patient_id": f"P{i:04d}",
        "text": text,
        "label": 1 if is_frailty else 0,
        "positive_indicators": positives
    })

df = pd.DataFrame(data)
df.to_csv("frailty_fake_data.csv", index=False, encoding="utf-8")
print(f"数据集生成完成，共{len(df)}条")
print(df.head(2))