import os
import threading
import requests
import tkinter as tk
from tkinter import scrolledtext, messagebox

# FastAPI 服务地址（本地运行）
API_BASE_URL = "http://localhost:8000"

# ========== 辅助函数 ==========
def call_predict(text):
    """调用 /predict 接口"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict",
            json={"text": text},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"服务器返回状态码 {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到API服务，请确认已启动 uvicorn server:app --host 0.0.0.0 --port 8000"}
    except Exception as e:
        return {"error": f"请求异常：{str(e)}"}

def call_predict_with_explanation(text):
    """调用 /predict_with_explanation 接口"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/predict_with_explanation",
            json={"text": text},
            timeout=60          # 解释生成可能较慢，适当延长超时
        )
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"服务器返回状态码 {response.status_code}"}
    except requests.exceptions.ConnectionError:
        return {"error": "无法连接到API服务，请确认已启动 uvicorn server:app --host 0.0.0.0 --port 8000"}
    except Exception as e:
        return {"error": f"请求异常：{str(e)}"}

# ========== 图形界面类 ==========
class FrailtyApp:
    def __init__(self, root):
        self.root = root
        self.root.title("老年衰弱自动识别系统 (API版)")
        self.root.geometry("700x650")
        self.root.resizable(True, True)

        # 输入标签
        lbl_input = tk.Label(root, text="请输入电子病历文本：", font=("微软雅黑", 12))
        lbl_input.pack(pady=10)

        # 多行输入框
        self.txt_input = scrolledtext.ScrolledText(root, width=80, height=8, font=("微软雅黑", 10))
        self.txt_input.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)

        # 按钮框架
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        # 快速预测按钮
        self.btn_quick = tk.Button(btn_frame, text="快速预测", command=self.on_quick_predict,
                                   bg="#4CAF50", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_quick.pack(side=tk.LEFT, padx=5)

        # 详细解释按钮
        self.btn_explain = tk.Button(btn_frame, text="详细解释 (RAG)", command=self.on_explain_predict,
                                     bg="#FF9800", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_explain.pack(side=tk.LEFT, padx=5)

        # 清空按钮
        self.btn_clear = tk.Button(btn_frame, text="清空文本", command=self.on_clear,
                                   bg="#f44336", fg="white", font=("微软雅黑", 10), width=14)
        self.btn_clear.pack(side=tk.LEFT, padx=5)

        # 结果显示标签
        lbl_result = tk.Label(root, text="评估结果：", font=("微软雅黑", 12, "bold"))
        lbl_result.pack(pady=(10, 0))

        # 多行结果显示框
        self.txt_result = tk.Text(root, height=15, font=("微软雅黑", 10), bg="#f9f9f9", wrap=tk.WORD)
        self.txt_result.pack(pady=5, padx=10, fill=tk.BOTH, expand=True)
        self.txt_result.config(state=tk.DISABLED)

        # 示例文本按钮
        btn_example = tk.Button(root, text="填入示例文本", command=self.fill_example,
                                bg="#2196F3", fg="white", font=("微软雅黑", 9))
        btn_example.pack(pady=5)

    def set_result(self, text):
        """安全更新结果框内容"""
        self.txt_result.config(state=tk.NORMAL)
        self.txt_result.delete("1.0", tk.END)
        self.txt_result.insert(tk.END, text)
        self.txt_result.config(state=tk.DISABLED)

    def on_quick_predict(self):
        """快速预测（仅输出衰弱/非衰弱）"""
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入病历文本！")
            return

        self.set_result("正在调用预测服务，请稍候...")
        self.disable_buttons()

        def task():
            result = call_predict(text)
            if "error" in result:
                output = f"❌ 请求失败\n{result['error']}"
            else:
                output = f"预测结论：{result['prediction']}\n"
                output += f"置信度：{result['confidence']:.4f}\n"
                output += f"输入文本摘要：{result['input_text'][:60]}..."
            self.root.after(0, self.set_result, output)
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task, daemon=True).start()

    def on_explain_predict(self):
        """详细解释（调用RAG接口）"""
        text = self.txt_input.get("1.0", tk.END).strip()
        if not text:
            messagebox.showwarning("提示", "请输入病历文本！")
            return

        self.set_result("正在调用大模型生成详细解释，可能需要5~15秒，请稍候...")
        self.disable_buttons()

        def task():
            result = call_predict_with_explanation(text)
            if "error" in result:
                output = f"❌ 请求失败\n{result['error']}"
            else:
                output = f"预测结论：{result['prediction']}（置信度：{result['confidence']:.4f}）\n"
                output += "-" * 40 + "\n"
                output += result.get('explanation', '（未返回解释文本）')
            self.root.after(0, self.set_result, output)
            self.root.after(0, self.enable_buttons)

        threading.Thread(target=task, daemon=True).start()

    def disable_buttons(self):
        self.btn_quick.config(state=tk.DISABLED)
        self.btn_explain.config(state=tk.DISABLED)

    def enable_buttons(self):
        self.btn_quick.config(state=tk.NORMAL)
        self.btn_explain.config(state=tk.NORMAL)

    def on_clear(self):
        self.txt_input.delete("1.0", tk.END)
        self.set_result("")

    def fill_example(self):
        example = "患者女性，78岁，近半年体重下降约6kg，自感疲乏无力，步速减慢。"
        self.txt_input.delete("1.0", tk.END)
        self.txt_input.insert("1.0", example)

if __name__ == "__main__":
    root = tk.Tk()
    app = FrailtyApp(root)
    root.mainloop()