"""知识库构建脚本 - 支持批量处理"""

import os
import sys
import json
import time
import warnings
from pathlib import Path
from typing import List

# 抑制 transformers 库的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from config_manager import get_config


def load_medical_documents(json_path: str, max_content_length: int = 500) -> List[Document]:
    """从JSON文件加载医疗文档"""
    documents = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = eval(line)  # 使用eval处理JSON对象
                
                content_parts = []
                if name := data.get("name"):
                    content_parts.append(f"疾病名称：{name}")
                if desc := data.get("desc"):
                    # 限制描述长度
                    desc = desc[:300] + "..." if len(desc) > 300 else desc
                    content_parts.append(f"疾病描述：{desc}")
                if symptom := data.get("symptom"):
                    sym_str = ', '.join(symptom) if isinstance(symptom, list) else symptom
                    content_parts.append(f"主要症状：{sym_str}")
                if cause := data.get("cause"):
                    # 限制病因长度
                    cause = cause[:150] + "..." if len(cause) > 150 else cause
                    content_parts.append(f"病因：{cause}")
                if cure_way := data.get("cure_way"):
                    cure_str = ', '.join(cure_way) if isinstance(cure_way, list) else cure_way
                    content_parts.append(f"治疗方法：{cure_str}")
                if check := data.get("check"):
                    check_str = ', '.join(check[:3]) if isinstance(check, list) else check
                    content_parts.append(f"检查项目：{check_str}")
                
                content = "\n".join(content_parts)
                
                if content and len(content) > 20:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": data.get("name", "未知疾病"),
                            "disease_name": data.get("name", "")
                        }
                    )
                    documents.append(doc)
                    
            except Exception:
                continue
    
    return documents


def check_gpu_available():
    """检查GPU是否可用"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def build_knowledge_base_batch(
    json_path: str,
    persist_dir: str = "./vector_store",
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    batch_size: int = 64,
    device: str = None
):
    """批量构建知识库"""
    # 自动检测GPU
    if device is None:
        device = check_gpu_available()
    
    device_name = "GPU (CUDA)" if device == "cuda" else "CPU"
    print(f"开始构建知识库...")
    print(f"  知识库路径: {json_path}")
    print(f"  向量库路径: {persist_dir}")
    print(f"  Embedding: {embedding_model}")
    print(f"  计算设备: {device_name}")
    print(f"  批处理大小: {batch_size}")
    print()
    
    if not os.path.exists(json_path):
        print(f"错误: 知识库文件不存在: {json_path}")
        return False
    
    try:
        # 加载文档
        print("正在加载医学知识库...")
        documents = load_medical_documents(json_path)
        total_docs = len(documents)
        print(f"成功加载 {total_docs} 条记录")
        
        if not documents:
            print("错误: 知识库为空")
            return False
        
        # 创建embedding
        print(f"\n正在初始化Embedding模型...")
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
        
        # 清理旧目录
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
        os.makedirs(persist_dir, exist_ok=True)
        
        # 批量处理
        print(f"\n正在构建FAISS向量库 (分批处理)...")
        start_time = time.time()
        
        # 第一批创建索引
        first_batch_size = min(batch_size, total_docs)
        first_batch = documents[:first_batch_size]
        texts = [doc.page_content for doc in first_batch]
        metadatas = [doc.metadata for doc in first_batch]
        
        print(f"  处理第 1/{(total_docs + batch_size - 1) // batch_size} 批 (0-{first_batch_size})...")
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas
        )
        
        # 后续批次追加
        for i in range(first_batch_size, total_docs, batch_size):
            batch_end = min(i + batch_size, total_docs)
            batch = documents[i:batch_end]
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            
            batch_num = i // batch_size + 1
            total_batches = (total_docs + batch_size - 1) // batch_size
            print(f"  处理第 {batch_num}/{total_batches} 批 ({i}-{batch_end})...")
            
            vector_store.add_texts(texts=texts, metadatas=metadatas)
        
        # 保存
        elapsed = time.time() - start_time
        print(f"\n正在保存向量库...")
        vector_store.save_local(persist_dir)
        
        print(f"\n✅ 知识库构建成功!")
        print(f"   存储路径: {persist_dir}")
        print(f"   记录数量: {total_docs}")
        print(f"   耗时: {elapsed:.1f} 秒")
        return True
        
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="构建医疗知识库")
    parser.add_argument("--json_path", type=str, default="../medical.json")
    parser.add_argument("--persist_dir", type=str, default="./vector_store")
    parser.add_argument("--embedding_model", type=str, default="Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", type=str, choices=["cpu", "cuda"], default=None, help="计算设备 (默认自动检测)")
    
    args = parser.parse_args()
    
    success = build_knowledge_base_batch(
        json_path=args.json_path,
        persist_dir=args.persist_dir,
        embedding_model=args.embedding_model,
        batch_size=args.batch_size,
        device=args.device
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
