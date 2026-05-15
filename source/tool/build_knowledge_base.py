#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_knowledge_base.py
功能：从《中国老年衰弱相关内分泌激素管理临床实践指南(2023)》PDF
      提取文本 → 按章节分块 + 推荐条目分块 → 向量化 → 存入 ChromaDB
用法：python build_knowledge_base.py <PDF文件路径>
注意：更换嵌入模型后必须重新运行本脚本，以生成匹配的向量库。
"""

import sys
import os
import re
import pdfplumber
import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- 工具函数 ----------------

def extract_text_from_pdf(pdf_path):
    """使用 pdfplumber 提取 PDF 全文"""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_guideline_by_chapter(text):
    """
    将指南全文按“第X章 ...”分割为章节块。
    过滤目录、参考文献等无关内容。
    返回列表，每个元素为 (章节标题, 正文) 元组。
    """
    first_chapter_match = re.search(r'第[一二三四五六七八九十\d]+章\s+', text)
    if first_chapter_match:
        text = text[first_chapter_match.start():]

    pattern = r'(第[一二三四五六七八九十\d]+章\s+.+)'
    parts = re.split(pattern, text)
    chunks = []
    current_title = "前言"
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if re.match(pattern, part):
            current_title = part
        else:
            chunk = current_title + "\n" + part
            chunk = re.sub(r'\n{3,}', '\n\n', chunk)
            ref_pos = re.search(r'\n(参考文献|附表|本指南编撰委员会名单)', chunk)
            if ref_pos:
                chunk = chunk[:ref_pos.start()]
            chunks.append((current_title, chunk.strip()))
    return chunks


def extract_recommendation_chunks(text):
    """
    从正文中提取所有“推荐X-Y”及其后续说明文字，作为独立条目块。
    返回列表，每个元素为 (推荐编号, 推荐内容) 元组。
    """
    recs = []
    # 匹配 "推荐X-Y"（X、Y 为数字或中文数字），并捕获后面的文本直到下一个推荐或章节标题或空行
    pattern = r'(推荐\d+-\d+[：:]?\s*)([\s\S]*?)(?=\n(?:推荐\d+-\d+|第[一二三四五六七八九十\d]+章|\Z))'
    for match in re.finditer(pattern, text):
        rec_id = match.group(1).strip()
        content = match.group(2).strip()
        if content:
            recs.append((rec_id, content))
    return recs


def load_embedding_model():
    """
    加载本地嵌入模型 BAAI/bge-large-zh-v1.5，不存在则自动下载。
    模型存放于脚本所在目录的 ../models/bge-large-zh-v1.5
    """
    model_repo = "BAAI/bge-large-zh-v1.5"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    models_root = os.path.join(script_dir, "..", "models")
    os.makedirs(models_root, exist_ok=True)
    local_model_dir = os.path.join(models_root, "bge-large-zh-v1.5")

    if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
        print(f"✅ 从本地加载嵌入模型：{local_model_dir}")
        return SentenceTransformer(local_model_dir)

    print(f"⏳ 正在下载 {model_repo} ...（约 1.3 GB）")
    try:
        model = SentenceTransformer(model_repo)
        model.save(local_model_dir)
        print(f"✅ 模型已保存至 {local_model_dir}")
        return model
    except Exception as e:
        print(f"❌ 下载失败：{e}")
        sys.exit(1)


def build_knowledge_base(chapter_chunks, recommendation_chunks):
    """
    创建/重建 ChromaDB 集合，并将所有块向量化存储。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, "..", "geriatric_knowledge_db")
    client = chromadb.PersistentClient(path=db_path)

    # 删除旧集合（如果存在）
    try:
        client.delete_collection("geriatric_guidelines")
        print("已删除旧集合，准备重建。")
    except:
        pass

    # 创建集合，指定余弦距离空间
    collection = client.create_collection(
        name="geriatric_guidelines",
        metadata={"hnsw:space": "cosine"}
    )

    embedding_model = load_embedding_model()

    all_contents = []
    all_metadatas = []
    all_ids = []
    idx = 0

    # 添加章节块
    for title, content in chapter_chunks:
        all_contents.append(content)
        all_metadatas.append({
            "content": content,
            "type": "chapter",
            "source": title
        })
        all_ids.append(f"doc_{idx}")
        idx += 1

    # 添加推荐条目块
    for rec_id, content in recommendation_chunks:
        all_contents.append(content)
        all_metadatas.append({
            "content": content,
            "type": "recommendation",
            "source": rec_id
        })
        all_ids.append(f"doc_{idx}")
        idx += 1

    embeddings = embedding_model.encode(all_contents).tolist()
    collection.add(
        ids=all_ids,
        embeddings=embeddings,
        metadatas=all_metadatas
    )
    print(f"✅ 知识库构建完成，共存储 {len(all_contents)} 个块（{len(chapter_chunks)} 章节，{len(recommendation_chunks)} 条目）")


# ---------------- 主流程 ----------------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        pdf_path = "../../data/文献书籍/中国老年衰弱相关内分泌激素管理临床实践指南(2023).pdf"
        print(f"未指定PDF路径，使用默认路径：{pdf_path}")
    else:
        pdf_path = sys.argv[1]

    if not os.path.exists(pdf_path):
        print(f"错误：找不到 PDF 文件 {pdf_path}")
        sys.exit(1)

    print("提取 PDF 文本...")
    full_text = extract_text_from_pdf(pdf_path)

    print("按章节分块...")
    chapter_chunks = split_guideline_by_chapter(full_text)
    print(f"共得到 {len(chapter_chunks)} 个章节块。")

    print("提取推荐条目块...")
    recommendation_chunks = extract_recommendation_chunks(full_text)
    print(f"共得到 {len(recommendation_chunks)} 个推荐条目块。")

    print("构建向量知识库...")
    build_knowledge_base(chapter_chunks, recommendation_chunks)