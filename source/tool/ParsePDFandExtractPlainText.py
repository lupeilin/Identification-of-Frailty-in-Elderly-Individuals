import pdfplumber

def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# 用法示例
guideline_text = extract_text_from_pdf("../../data/文献书籍/中国老年衰弱相关内分泌激素管理临床实践指南(2023).pdf")

print(guideline_text)