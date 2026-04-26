"""配置管理模块 - 支持动态配置读取和保存"""

import os
import yaml
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()


class LLMConfig(BaseModel):
    """大模型配置"""
    provider: str = Field(default="modelscope", description="LLM提供商: openai, modelscope, ollama")
    api_key: str = Field(default="", description="API密钥")
    api_base: str = Field(default="", description="API基础URL")
    model: str = Field(default="gpt-4o-mini", description="模型名称")
    temperature: float = Field(default=0.3, description="温度参数")
    max_tokens: int = Field(default=2048, description="最大生成token数")
    timeout: int = Field(default=60, description="超时时间(秒)")
    max_retries: int = Field(default=2, description="最大重试次数")


class EmbeddingConfig(BaseModel):
    """Embedding配置"""
    provider: str = Field(default="huggingface", description="Embedding提供商: huggingface, openai")
    model: str = Field(default="Qwen/Qwen3-Embedding-0.6B", description="Embedding模型")
    device: str = Field(default="cpu", description="设备: cpu, cuda")
    batch_size: int = Field(default=32, description="批处理大小")


class VectorStoreConfig(BaseModel):
    """向量数据库配置"""
    store_type: str = Field(default="faiss", description="向量库类型: faiss, chroma")
    persist_dir: str = Field(default="./vector_store", description="持久化目录")
    index_type: str = Field(default="Flat", description="FAISS索引类型: Flat, IVF, HNSW")


class RetrieveConfig(BaseModel):
    """检索配置"""
    search_type: str = Field(default="similarity", description="检索类型: similarity, mmr, similarity_score_threshold")
    top_k: int = Field(default=3, description="返回文档数量")
    mmr_fetch_k: int = Field(default=10, description="MMR预取数量")
    mmr_lambda_mult: float = Field(default=0.5, description="MMR多样性系数")
    score_threshold: float = Field(default=0.5, description="相似度阈值")


class KnowledgeBaseConfig(BaseModel):
    """知识库配置"""
    data_path: str = Field(default="./medical.json", description="知识库数据路径")
    vector_store_path: str = Field(default="./vector_store", description="向量库存储路径")


class AppConfig(BaseModel):
    """应用配置"""
    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    knowledge_base: KnowledgeBaseConfig = Field(default_factory=KnowledgeBaseConfig)


def load_config_from_env() -> AppConfig:
    """从环境变量加载配置"""
    # LLM配置
    llm_provider = os.getenv("LLM_PROVIDER", "modelscope")
    
    if llm_provider == "openai":
        llm_config = LLMConfig(
            provider="openai",
            api_key=os.getenv("OPENAI_API_KEY", ""),
            api_base=os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1"),
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2"))
        )
    elif llm_provider == "ollama":
        llm_config = LLMConfig(
            provider="ollama",
            api_key=os.getenv("OLLAMA_API_KEY", "ollama"),
            api_base=os.getenv("OLLAMA_API_BASE", "http://localhost:11434/v1"),
            model=os.getenv("OLLAMA_MODEL", "gemma4:e2b"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2"))
        )
    else:  # modelscope (默认)
        llm_config = LLMConfig(
            provider="modelscope",
            api_key=os.getenv("MODELSCOPE_API_KEY", os.getenv("MOTSCOPE_OPENAI_API_KEY", "")),
            api_base=os.getenv("MODELSCOPE_API_BASE", os.getenv("MOTSCOPE_OPENAI_API_BASE", "https://api-inference.modelscope.cn/v1")),
            model=os.getenv("MODELSCOPE_MODEL", "MiniMax/MiniMax-M2.7"),
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.3")),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "2048")),
            timeout=int(os.getenv("LLM_TIMEOUT", "60")),
            max_retries=int(os.getenv("LLM_MAX_RETRIES", "2"))
        )

    # Embedding配置
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "huggingface")
    embedding_config = EmbeddingConfig(
        provider=embedding_provider,
        model=os.getenv("EMBEDDING_MODEL", "Qwen/Qwen3-Embedding-0.6B"),
        device=os.getenv("EMBEDDING_DEVICE", "cpu")
    )

    # 向量库配置
    vector_store_config = VectorStoreConfig(
        store_type=os.getenv("VECTOR_STORE_TYPE", "faiss"),
        persist_dir=os.getenv("VECTOR_STORE_PATH", "./vector_store"),
        index_type=os.getenv("FAISS_INDEX_TYPE", "Flat")
    )

    # 检索配置
    retrieve_config = RetrieveConfig(
        search_type=os.getenv("SEARCH_TYPE", "similarity"),
        top_k=int(os.getenv("RETRIEVE_TOP_K", "3")),
        mmr_fetch_k=int(os.getenv("MMR_FETCH_K", "10")),
        mmr_lambda_mult=float(os.getenv("MMR_LAMBDA_MULT", "0.5")),
        score_threshold=float(os.getenv("SIMILARITY_SCORE_THRESHOLD", "0.5"))
    )

    # 知识库配置
    knowledge_base_config = KnowledgeBaseConfig(
        data_path=os.getenv("KNOWLEDGE_BASE_PATH", "./medical.json"),
        vector_store_path=os.getenv("VECTOR_STORE_PATH", "./vector_store")
    )

    return AppConfig(
        llm=llm_config,
        embedding=embedding_config,
        vector_store=vector_store_config,
        retrieve=retrieve_config,
        knowledge_base=knowledge_base_config
    )


def load_config_from_yaml(yaml_path: str) -> AppConfig:
    """从YAML文件加载配置"""
    if not os.path.exists(yaml_path):
        return load_config_from_env()
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_config = yaml.safe_load(f)
    
    # 如果YAML中没有值，回退到环境变量
    if yaml_config is None:
        return load_config_from_env()
    
    # 构建配置对象
    llm_cfg = yaml_config.get('llm', {})
    emb_cfg = yaml_config.get('embedding', {})
    vs_cfg = yaml_config.get('vector_store', {})
    ret_cfg = yaml_config.get('retrieve', {})
    kb_cfg = yaml_config.get('knowledge_base', {})
    
    return AppConfig(
        llm=LLMConfig(
            provider=llm_cfg.get('provider', os.getenv('LLM_PROVIDER', 'modelscope')),
            api_key=llm_cfg.get('api_key', ''),
            api_base=llm_cfg.get('api_base', ''),
            model=llm_cfg.get('model', 'gpt-4o-mini'),
            temperature=float(llm_cfg.get('temperature', os.getenv('LLM_TEMPERATURE', '0.3'))),
            max_tokens=int(llm_cfg.get('max_tokens', os.getenv('LLM_MAX_TOKENS', '2048'))),
            timeout=int(llm_cfg.get('timeout', os.getenv('LLM_TIMEOUT', '60'))),
            max_retries=int(llm_cfg.get('max_retries', os.getenv('LLM_MAX_RETRIES', '2')))
        ),
        embedding=EmbeddingConfig(
            provider=emb_cfg.get('provider', os.getenv('EMBEDDING_PROVIDER', 'huggingface')),
            model=emb_cfg.get('model', os.getenv('EMBEDDING_MODEL', 'Qwen/Qwen3-Embedding-0.6B')),
            device=emb_cfg.get('device', os.getenv('EMBEDDING_DEVICE', 'cpu'))
        ),
        vector_store=VectorStoreConfig(
            store_type=vs_cfg.get('store_type', os.getenv('VECTOR_STORE_TYPE', 'faiss')),
            persist_dir=vs_cfg.get('persist_dir', os.getenv('VECTOR_STORE_PATH', './vector_store')),
            index_type=vs_cfg.get('index_type', os.getenv('FAISS_INDEX_TYPE', 'Flat'))
        ),
        retrieve=RetrieveConfig(
            search_type=ret_cfg.get('search_type', os.getenv('SEARCH_TYPE', 'similarity')),
            top_k=int(ret_cfg.get('top_k', os.getenv('RETRIEVE_TOP_K', '3'))),
            mmr_fetch_k=int(ret_cfg.get('mmr_fetch_k', os.getenv('MMR_FETCH_K', '10'))),
            mmr_lambda_mult=float(ret_cfg.get('mmr_lambda_mult', os.getenv('MMR_LAMBDA_MULT', '0.5'))),
            score_threshold=float(ret_cfg.get('score_threshold', os.getenv('SIMILARITY_SCORE_THRESHOLD', '0.5')))
        ),
        knowledge_base=KnowledgeBaseConfig(
            data_path=kb_cfg.get('data_path', os.getenv('KNOWLEDGE_BASE_PATH', './medical.json')),
            vector_store_path=kb_cfg.get('vector_store_path', os.getenv('VECTOR_STORE_PATH', './vector_store'))
        )
    )


def save_config_to_yaml(config: AppConfig, yaml_path: str):
    """保存配置到YAML文件"""
    config_dict = {
        'llm': {
            'provider': config.llm.provider,
            'api_key': config.llm.api_key,
            'api_base': config.llm.api_base,
            'model': config.llm.model,
            'temperature': config.llm.temperature,
            'max_tokens': config.llm.max_tokens,
            'timeout': config.llm.timeout,
            'max_retries': config.llm.max_retries
        },
        'embedding': {
            'provider': config.embedding.provider,
            'model': config.embedding.model,
            'device': config.embedding.device
        },
        'vector_store': {
            'store_type': config.vector_store.store_type,
            'persist_dir': config.vector_store.persist_dir,
            'index_type': config.vector_store.index_type
        },
        'retrieve': {
            'search_type': config.retrieve.search_type,
            'top_k': config.retrieve.top_k,
            'mmr_fetch_k': config.retrieve.mmr_fetch_k,
            'mmr_lambda_mult': config.retrieve.mmr_lambda_mult,
            'score_threshold': config.retrieve.score_threshold
        },
        'knowledge_base': {
            'data_path': config.knowledge_base.data_path,
            'vector_store_path': config.knowledge_base.vector_store_path
        }
    }
    
    os.makedirs(os.path.dirname(yaml_path) if os.path.dirname(yaml_path) else '.', exist_ok=True)
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=False)


# 全局配置实例
_config: Optional[AppConfig] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """获取配置单例"""
    global _config
    if _config is None:
        if config_path and os.path.exists(config_path):
            _config = load_config_from_yaml(config_path)
        else:
            _config = load_config_from_env()
    return _config


def reset_config():
    """重置配置"""
    global _config
    _config = None
