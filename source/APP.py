import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tkinter as tk
from tkinter import scrolledtext, messagebox
import threading
import requests

# ================= 1. 加载 BERT 模型（程序启动时加载一次）=================
MODEL_PATH = "./frailty_bert_demo_model"

print("正在加载衰弱识别模型，请稍候...")
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()
print(f"✅ 模型加载完毕，运行于 {device}")

# ================= 2. 预测函数（直接本地计算）=================
def predict_frailty(text):
    """返回预测类别和置信度"""
    inputs = tokenizer(text, padding="max_length", truncation=True,
                       max_length=128, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = probs.argmax().item()
        conf = probs[0][pred].item()
    return {
        "prediction": "衰弱" if pred == 1 else "非衰弱",
        "confidence": conf
    }

# ================= 3. RAG 解释生成（调用本地 Ollama API）=================
def generate_explanation(text, bert_pred, bert_conf):
    """
    使用本地大模型（Ollama）生成详细的临床解释。
    如果 Ollama 未启动或调用失败，返回错误提示。
    """
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
预测结果：{bert_pred}（置信度：{bert_conf:.2%}）

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
                "model": "qwen2.5:3b",   # 改成你实际下载的模型名称
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=60
        )
        if response.status_code == 200:
            return response.json()["response"]
        else:
            return f"❌ Ollama 返回错误，状态码：{response.status_code}"
    except requests.exceptions.ConnectionError:
        return "❌ 无法连接到 Ollama，请确保已启动 ollama serve 并下载了相应模型。"
    except Exception as e:
        return f"❌ 请求异常：{str(e)}"

# ================= 4. 图形界面 =================
class FrailtyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("老年衰弱自动识别系统 (独立版)")
        self.root.geometry("750x650")
        self.root.resizable(True, True)

        # 输入区域
        lbl_input = tk.Label(root, text="请输入电子病历文本：", font=("微软雅黑", 12))
        lbl_input.pack(pady=10)

        self.txt_input = scrolledtext.ScrolledText(
            root, width=80, height=8, font=("微软雅黑", 10))
        self.txt_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # 按钮框架
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_quick = tk.Button(btn_frame, text="快速预测", command=self.on_quick_predict,
                                   bg="#4CAF50", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_quick.pack(side=tk.LEFT, padx=5)

        self.btn_explain = tk.Button(btn_frame, text="详细解释 (RAG)", command=self.on_explain_predict,
                                     bg="#FF9800", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_explain.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="清空文本", command=self.on_clear,
                                   bg="#f44336", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        lbl_result = tk.Label(root, text="评估结果：", font=("微软雅黑", 12, "bold"))
        lbl_result.pack(pady=(10, 0))

        self.txt_result = tk.Text(root, height=15, font=("微软雅黑", 10),
                                  bg="#f9f9f9", wrap=tk.WORD)
        self.txt_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.txt_result.config(state=tk.DISABLED)

        # 填入示例
        btn_example = tk.Button(root, text="填入示例文本", command=self.fill_example,
                                bg="#2196F3", fg="white", font=("微软雅黑", 9))
        btn_example.pack(pady=5)

    # ---------- 辅助方法 ----------
    def set_result(self, text):
        """安全更新结果框"""
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, text)
        self.txt_result.config(state=tk.DISABLED)

    def set_buttons_state(self, state):
        self.btn_quick.config(state=state)
        self.btn_explain.config(state=state)

    # ---------- 功能回调 ----------
    def on_quick_predict(self):
        """仅使用 BERT 进行快速预测"""
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入病历文本！")
            return

        self.set_result("正在预测，请稍候...")
        self.set_buttons_state(tk.DISABLED)

        def task():
            try:
                result = predict_frailty(text)
                output = f"预测结论：{result['prediction']}\n"
                output += f"置信度：{result['confidence']:.4f}\n"
                output += f"输入摘要：{text[:60]}..."
            except Exception as e:
                output = f"❌ 预测失败：{str(e)}"
            self.root.after(0, self.set_result, output)
            self.root.after(0, self.set_buttons_state, tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def on_explain_predict(self):
        """BERT预测 + RAG详细解释"""
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入病历文本！")
            return

        # 先显示等待信息
        self.set_result("正在调用大模型生成详细解释，可能需要 10~30 秒，请稍候...")
        self.set_buttons_state(tk.DISABLED)

        def task():
            try:
                # 1. BERT预测
                bert_result = predict_frailty(text)
                pred = bert_result["prediction"]
                conf = bert_result["confidence"]

                # 2. RAG解释
                explanation = generate_explanation(text, pred, conf)

                # 拼接最终结果
                output = f"【BERT 快速预测】\n"
                output += f"结论：{pred}（置信度：{conf:.4f}）\n"
                output += "-" * 50 + "\n"
                output += f"【大模型详细解释】\n{explanation}"
            except Exception as e:
                output = f"❌ 处理失败：{str(e)}"

            self.root.after(0, self.set_result, output)
            self.root.after(0, self.set_buttons_state, tk.NORMAL)

        threading.Thread(target=task, daemon=True).start()

    def on_clear(self):
        self.txt_input.delete("1.0", tk.END)
        self.set_result("")

    def fill_example(self):
        example = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
        self.txt_input.delete("1.0", tk.END)
        self.txt_input.insert("1.0", example)

# ================= 5. 启动 =================
if __name__ == "__main__":
    root = tk.Tk()
    app = FrailtyApp(root)
    root.mainloop()