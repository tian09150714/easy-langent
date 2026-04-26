"""向量数据库管理器 - 支持FAISS和Chroma"""

import os
import json
import warnings
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

# 抑制 transformers 库的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from config_manager import get_config, AppConfig


class VectorStoreManager:
    """向量数据库管理器"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self._embeddings = None
        self._vector_store = None
    
    @property
    def embeddings(self):
        """获取嵌入模型"""
        if self._embeddings is None:
            self._embeddings = self._create_embeddings()
        return self._embeddings
    
    def _create_embeddings(self):
        """创建嵌入模型"""
        emb_cfg = self.config.embedding
        
        if emb_cfg.provider == "openai":
            return OpenAIEmbeddings(
                model=emb_cfg.model,
                api_key=emb_cfg.api_key or None
            )
        else:  # huggingface / modelscope
            return HuggingFaceEmbeddings(
                model_name=emb_cfg.model,
                model_kwargs={"device": emb_cfg.device},
                encode_kwargs={"normalize_embeddings": True}
            )
    
    def _load_vector_store(self) -> Any:
        """尝试加载已存在的向量库"""
        vs_cfg = self.config.vector_store
        persist_dir = vs_cfg.persist_dir
        
        if not os.path.exists(persist_dir):
            return None
        
        try:
            if vs_cfg.store_type.lower() == "chroma":
                return Chroma(
                    persist_directory=persist_dir,
                    embedding_function=self.embeddings
                )
            else:  # faiss
                index_path = os.path.join(persist_dir, "index.faiss")
                if os.path.exists(index_path):
                    return FAISS.load_local(
                        persist_dir,
                        self.embeddings,
                        allow_dangerous_deserialization=True
                    )
        except Exception as e:
            print(f"加载向量库失败: {e}，将重新创建")
        
        return None
    
    def _create_vector_store_from_docs(self, documents: List[Document]) -> Any:
        """从文档创建向量库"""
        vs_cfg = self.config.vector_store
        
        if vs_cfg.store_type.lower() == "chroma":
            vs = Chroma(
                persist_directory=vs_cfg.persist_dir,
                embedding_function=self.embeddings
            )
            vs.add_documents(documents)
            return vs
        else:  # faiss (默认)
            # 直接从文档创建FAISS索引
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            return FAISS.from_texts(
                texts=texts,
                embedding=self.embeddings,
                metadatas=metadatas
            )
    
    @property
    def vector_store(self):
        """获取向量数据库"""
        if self._vector_store is None:
            self._vector_store = self._load_vector_store()
        return self._vector_store
    
    def add_documents(self, documents: List[Document]):
        """添加文档到向量库"""
        if not documents:
            return
        
        os.makedirs(self.config.vector_store.persist_dir, exist_ok=True)
        
        if self._vector_store is None:
            # 首次创建
            self._vector_store = self._create_vector_store_from_docs(documents)
        else:
            # 追加文档
            self._vector_store.add_documents(documents)
        
        self._save_vector_store()
    
    def _save_vector_store(self):
        """保存向量库到磁盘"""
        vs_cfg = self.config.vector_store
        persist_dir = vs_cfg.persist_dir
        
        os.makedirs(persist_dir, exist_ok=True)
        
        if vs_cfg.store_type.lower() == "chroma":
            self._vector_store.persist()
        else:  # faiss
            self._vector_store.save_local(persist_dir)
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """相似度检索"""
        return self.vector_store.similarity_search(query, k=k)
    
    def similarity_search_with_score(self, query: str, k: int = 4) -> List[Tuple[Document, float]]:
        """带分数的相似度检索"""
        return self.vector_store.similarity_search_with_score(query, k=k)
    
    def mmr_search(self, query: str, k: int = 4, fetch_k: int = 10, lambda_mult: float = 0.5) -> List[Document]:
        """最大边际相关性检索"""
        return self.vector_store.max_marginal_relevance_search(
            query, k=k, fetch_k=fetch_k, lambda_mult=lambda_mult
        )
    
    def as_retriever(self, search_type: str = "similarity", **kwargs):
        """转换为检索器"""
        search_kwargs = kwargs.copy()
        
        if search_type == "mmr":
            return self.vector_store.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": search_kwargs.pop("k", 4),
                    "fetch_k": search_kwargs.pop("fetch_k", 10),
                    "lambda_mult": search_kwargs.pop("lambda_mult", 0.5)
                }
            )
        elif search_type == "similarity_score_threshold":
            return self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={
                    "k": search_kwargs.pop("k", 4),
                    "score_threshold": search_kwargs.pop("score_threshold", 0.5)
                }
            )
        else:  # similarity (默认)
            return self.vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": search_kwargs.pop("k", 4)}
            )
    
    def exists(self) -> bool:
        """检查向量库是否存在"""
        vs_cfg = self.config.vector_store
        persist_dir = vs_cfg.persist_dir
        
        if not os.path.exists(persist_dir):
            return False
        
        if vs_cfg.store_type.lower() == "chroma":
            return True
        else:  # faiss
            index_path = os.path.join(persist_dir, "index.faiss")
            return os.path.exists(index_path)
    
    def delete(self):
        """删除向量库"""
        vs_cfg = self.config.vector_store
        persist_dir = vs_cfg.persist_dir
        
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
        
        self._vector_store = None
    
    def get_document_count(self) -> int:
        """获取文档数量"""
        if self._vector_store is not None:
            try:
                return self.vector_store.index.ntotal if hasattr(self.vector_store, 'index') else 0
            except:
                pass
        return 0
