import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import torch
from transformers import BertTokenizer, BertForSequenceClassification
import tkinter as tk
from tkinter import scrolledtext, messagebox

# ========== 1. 加载模型和分词器 ==========
model_path = "./frailty_bert_demo_model"  # 请确保路径正确
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"✅ 模型加载成功，运行在 {device}")

# ========== 2. 定义预测函数 ==========
def predict_frailty(text):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
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
    }

# ========== 3. 创建图形界面 ==========
class FrailtyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("老年衰弱自动识别系统")
        self.root.geometry("600x500")
        self.root.resizable(True, True)

        # 标签
        lbl_input = tk.Label(root, text="请输入电子病历文本：", font=("微软雅黑", 12))
        lbl_input.pack(pady=10)

        # 多行文本框（用于输入）
        self.txt_input = scrolledtext.ScrolledText(root, width=70, height=10, font=("微软雅黑", 10))
        self.txt_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # 按钮框架
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        self.btn_predict = tk.Button(btn_frame, text="开始评估", command=self.on_predict,
                                     bg="#4CAF50", fg="white", font=("微软雅黑", 10), width=12)
        self.btn_predict.pack(side=tk.LEFT, padx=5)

        self.btn_clear = tk.Button(btn_frame, text="清空文本", command=self.on_clear,
                                   bg="#f44336", fg="white", font=("微软雅黑", 10), width=12)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 结果显示区域
        lbl_result = tk.Label(root, text="评估结果：", font=("微软雅黑", 12, "bold"))
        lbl_result.pack(pady=(10,0))

        self.txt_result = tk.Text(root, height=6, font=("微软雅黑", 11), bg="#f9f9f9", wrap=tk.WORD)
        self.txt_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.txt_result.config(state=tk.DISABLED)  # 只读

        # 示例文本按钮
        btn_example = tk.Button(root, text="填入示例文本", command=self.fill_example,
                                bg="#2196F3", fg="white", font=("微软雅黑", 9))
        btn_example.pack(pady=5)

        # 绑定快捷键 Ctrl+Enter 执行预测
        self.txt_input.bind("<Control-Return>", lambda event: self.on_predict())

    def on_predict(self):
        """获取输入文本，调用预测函数，显示结果"""
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入病历文本！")
            return

        # 显示“正在评估...”提示
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, "正在评估，请稍候...\n")
        self.txt_result.config(state=tk.DISABLED)
        self.root.update()

        try:
            result = predict_frailty(text)
            # 格式化输出
            output = f"输入文本：{result['text'][:80]}...\n\n"
            output += f"预测结论：{result['prediction']}\n"
            output += f"置信度：{result['confidence']:.4f}"
            self.txt_result.config(state=tk.NORMAL)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, output)
            self.txt_result.config(state=tk.DISABLED)
        except Exception as e:
            messagebox.showerror("预测失败", f"发生错误：{str(e)}")
            self.txt_result.config(state=tk.NORMAL)
            self.txt_result.delete("1.0", tk.END)
            self.txt_result.insert(tk.END, "预测过程中出现异常，请检查日志或重试。")
            self.txt_result.config(state=tk.DISABLED)

    def on_clear(self):
        """清空输入框和结果"""
        self.txt_input.delete("1.0", tk.END)
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.config(state=tk.DISABLED)

    def fill_example(self):
        """填入一个示例文本（衰弱典型）"""
        example = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
        self.txt_input.delete("1.0", tk.END)
        self.txt_input.insert("1.0", example)

if __name__ == "__main__":
    root = tk.Tk()
    app = FrailtyApp(root)
    root.mainloop()