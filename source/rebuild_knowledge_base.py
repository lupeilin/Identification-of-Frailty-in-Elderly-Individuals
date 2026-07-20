#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rebuild_knowledge_base.py
功能：从 geriatric_guidelines.txt 重建 ChromaDB 知识库
用法：python rebuild_knowledge_base.py
"""

import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_DIR = os.path.join(BASE_DIR, "geriatric_knowledge_db")
GUIDELINES_PATH = os.path.join(os.path.dirname(BASE_DIR), "data", "geriatric_guidelines.txt")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_embedding_model():
    """加载本地嵌入模型"""
    model_repo = "BAAI/bge-large-zh-v1.5"
    local_model_dir = os.path.join(MODELS_DIR, "BAAI", "bge-large-zh-v1.5")

    if os.path.exists(local_model_dir) and os.path.isdir(local_model_dir):
        print(f"从本地加载嵌入模型：{local_model_dir}")
        return SentenceTransformer(local_model_dir)

    print(f"正在下载 {model_repo} ...（约 1.3 GB，首次需等待）")
    model = SentenceTransformer(model_repo)
    model.save(local_model_dir)
    print(f"模型已保存至 {local_model_dir}")
    return model


def split_text_into_chunks(text, max_length=800, overlap=100):
    """将长文本按段落分割成块，每块不超过 max_length 字符"""
    chunks = []
    current_chunk = ""
    current_title = ""

    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue

        # 检测标题行（以 ## 开头）
        if line.startswith('##'):
            # 保存之前的块
            if current_chunk:
                chunks.append((current_title, current_chunk.strip()))
            current_title = line.replace('##', '').strip()
            current_chunk = line + '\n'
        else:
            # 如果当前块太长，先保存
            if len(current_chunk) + len(line) > max_length and current_chunk:
                chunks.append((current_title, current_chunk.strip()))
                # 保留最后 overlap 字符作为下一块的开头
                if len(current_chunk) > overlap:
                    current_chunk = current_chunk[-overlap:] + '\n' + line + '\n'
                else:
                    current_chunk = line + '\n'
            else:
                current_chunk += line + '\n'

    # 保存最后一个块
    if current_chunk:
        chunks.append((current_title, current_chunk.strip()))

    return chunks


def rebuild_knowledge_base():
    """重建知识库"""
    # 读取指南文件
    if not os.path.exists(GUIDELINES_PATH):
        print(f"错误：找不到指南文件 {GUIDELINES_PATH}")
        sys.exit(1)

    with open(GUIDELINES_PATH, 'r', encoding='utf-8') as f:
        text = f.read()

    print(f"已读取指南文件：{len(text)} 字符")

    # 分割成块
    chunks = split_text_into_chunks(text, max_length=800, overlap=100)
    print(f"分割为 {len(chunks)} 个知识块")

    # 加载嵌入模型
    embedding_model = load_embedding_model()

    # 初始化 ChromaDB
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_or_create_collection(name="geriatric_guidelines")

    # 清空旧数据
    old_count = collection.count()
    if old_count > 0:
        print(f"清空旧知识库（{old_count} 条记录）...")
        # 新版 ChromaDB 需要指定 ids 或使用 where={"$and": []} 清空
        all_ids = collection.get()["ids"]
        if all_ids:
            collection.delete(ids=all_ids)

    # 向量化并入库
    print("正在向量化并入库...")
    for i, (title, content) in enumerate(chunks):
        embedding = embedding_model.encode([content]).tolist()
        collection.add(
            ids=[f"chunk_{i}"],
            embeddings=embedding,
            metadatas=[{
                "title": title,
                "content": content,
                "source": "geriatric_guidelines.txt"
            }]
        )
        if (i + 1) % 10 == 0:
            print(f"  已处理 {i + 1}/{len(chunks)} 块")

    print(f"知识库重建完成！共 {collection.count()} 条记录")
    print(f"   存储路径：{DB_DIR}")


if __name__ == "__main__":
    rebuild_knowledge_base()
