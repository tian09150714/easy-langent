"""医疗RAG系统 - Streamlit Web界面"""

import os
import sys
import warnings

# 抑制 transformers 库的警告
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
from pathlib import Path

# 添加当前目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from config_manager import get_config, reset_config, AppConfig, LLMConfig, EmbeddingConfig, VectorStoreConfig, RetrieveConfig
from medical_rag import MedicalRAG, load_medical_documents
from vector_store_manager import VectorStoreManager
from langchain_openai import ChatOpenAI


# 页面配置
st.set_page_config(
    page_title="医疗RAG诊断系统",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定义CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        padding: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86AB;
        color: white;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D4EDDA;
        border: 1px solid #C3E6CB;
        color: #155724;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #FFF3CD;
        border: 1px solid #FFEAA7;
        color: #856404;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #D1ECF1;
        border: 1px solid #B6D4E7;
        color: #0C5460;
    }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """初始化会话状态"""
    if 'medical_rag' not in st.session_state:
        st.session_state.medical_rag = None
    if 'config' not in st.session_state:
        st.session_state.config = get_config()
    if 'vector_store_built' not in st.session_state:
        st.session_state.vector_store_built = False
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []


def build_vector_store(config: AppConfig):
    """构建向量库"""
    try:
        # 加载文档
        json_path = config.knowledge_base.data_path
        if not os.path.exists(json_path):
            return False, f"知识库文件不存在: {json_path}"
        
        # 检查父目录
        parent_dir = os.path.dirname(json_path)
        if parent_dir and not os.path.exists(parent_dir):
            json_path = os.path.join(os.getcwd(), json_path)
        
        if not os.path.exists(json_path):
            return False, f"知识库文件不存在: {json_path}"
        
        with st.spinner("正在加载医学知识库..."):
            documents = load_medical_documents(json_path)
            
        if not documents:
            return False, "知识库为空或格式错误"
        
        with st.spinner(f"正在构建向量库 (共{len(documents)}条记录)..."):
            # 创建向量库管理器
            vs_manager = VectorStoreManager(config)
            
            # 添加文档
            vs_manager.add_documents(documents)
        
        return True, f"向量库构建成功，共加载 {len(documents)} 条医学记录"
    
    except Exception as e:
        return False, f"构建向量库失败: {str(e)}"


def test_llm_connection(llm_config: LLMConfig) -> tuple[bool, str]:
    """
    测试大模型连接是否正常
    
    Args:
        llm_config: LLM配置
        
    Returns:
        tuple[bool, str]: (是否成功, 消息)
    """
    import time
    
    start_time = time.time()
    try:
        # 处理API密钥
        api_key = llm_config.api_key
        if not api_key or api_key == "ollama":
            api_key = "ollama"
        
        # 创建LLM实例
        llm = ChatOpenAI(
            model=llm_config.model,
            api_key=api_key,
            base_url=llm_config.api_base,
            temperature=0.3,
            timeout=30,  # 测试用较短超时
            max_retries=1
        )
        
        # 发送测试消息
        test_message = "请回复'连接成功'，只需回复这四个字。"
        response = llm.invoke(test_message)
        
        elapsed = time.time() - start_time
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        return True, f"连接成功！响应时间: {elapsed:.2f}秒，模型回复: {response_text[:50]}"
        
    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = str(e)
        
        # 常见错误处理
        if "timeout" in error_msg.lower():
            return False, f"连接超时 ({elapsed:.1f}秒)，请检查网络或更换模型"
        elif "401" in error_msg or "authentication" in error_msg.lower():
            return False, "API密钥无效或已过期，请检查配置"
        elif "404" in error_msg:
            return False, "模型不存在，请检查模型名称是否正确"
        elif "connection" in error_msg.lower():
            return False, f"无法连接到服务器，请检查网络连接"
        elif "null value" in error_msg.lower() or "choices" in error_msg:
            return False, "API返回格式错误，模型可能不支持当前请求"
        else:
            return False, f"连接失败: {error_msg[:100]}"


def get_medical_rag() -> MedicalRAG:
    """获取或创建MedicalRAG实例"""
    if st.session_state.medical_rag is None:
        config = st.session_state.config
        st.session_state.medical_rag = MedicalRAG(config)
    return st.session_state.medical_rag


def render_sidebar():
    """渲染侧边栏配置"""
    with st.sidebar:
        st.markdown("## ⚙️ 系统配置")
        
        config = st.session_state.config
        
        # 标签页
        tab1, tab2, tab3 = st.tabs(["🦙 大模型配置", "🗄️ 向量库配置", "🔍 检索配置"])
        
        with tab1:
            st.markdown("### 大模型配置")
            
            # 提供商选择
            provider_options = {
                "modelscope": "ModelScope (默认)",
                "openai": "OpenAI",
                "ollama": "Ollama (本地)"
            }
            
            selected_provider = st.selectbox(
                "选择大模型提供商",
                options=list(provider_options.keys()),
                format_func=lambda x: provider_options[x],
                index=list(provider_options.keys()).index(config.llm.provider),
                key="llm_provider"
            )
            
            # 根据提供商显示不同配置
            if selected_provider == "openai":
                api_key = st.text_input(
                    "API Key",
                    value=config.llm.api_key or "",
                    type="password",
                    key="openai_api_key"
                )
                api_base = st.text_input(
                    "API Base URL",
                    value=config.llm.api_base or "https://api.openai.com/v1",
                    key="openai_api_base"
                )
                model = st.selectbox(
                    "模型",
                    options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                    index=0,
                    key="openai_model"
                )
            elif selected_provider == "ollama":
                api_base = st.text_input(
                    "API Base URL",
                    value= "http://localhost:11434/v1",
                    key="ollama_api_base"
                )
                model = st.selectbox(
                    "模型",
                    options=["gemma4:e2b", "qwen3.5:9b", "llama3:latest", "mistral:latest"],
                    index=0,
                    key="ollama_model"
                )
                api_key = "ollama"
            else:  # modelscope
                api_key = st.text_input(
                    "ModelScope API Key",
                    value=config.llm.api_key or "",
                    type="password",
                    key="modelscope_api_key"
                )
                api_base = st.text_input(
                    "API Base URL",
                    value=config.llm.api_base or "https://api-inference.modelscope.cn/v1",
                    key="modelscope_api_base"
                )
                model = st.selectbox(
                    "模型",
                    options=["Qwen/Qwen3.5-35B-A3B", "Qwen/Qwen2.5-72B-Instruct"],
                    index=0,
                    key="modelscope_model"
                )
            
            # 生成参数
            st.markdown("#### 生成参数")
            temperature = st.slider(
                "Temperature (创造性)",
                min_value=0.0,
                max_value=1.0,
                value=config.llm.temperature,
                step=0.1,
                key="llm_temperature"
            )
            
            max_tokens = st.number_input(
                "最大Token数",
                min_value=256,
                max_value=8192,
                value=config.llm.max_tokens,
                step=256,
                key="llm_max_tokens"
            )
            
            # 测试连接按钮
            st.markdown("---")
            st.markdown("#### 🔗 连接测试")
            
            if st.button("🧪 测试大模型连接", use_container_width=True, key="test_llm_btn"):
                with st.spinner("正在测试连接..."):
                    # 构建测试用的临时LLMConfig
                    test_llm_config = LLMConfig(
                        provider=selected_provider,
                        api_key=api_key if 'api_key' in dir() else config.llm.api_key,
                        api_base=api_base,
                        model=model,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        timeout=30,
                        max_retries=1
                    )
                    success, message = test_llm_connection(test_llm_config)
                    
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
        
        with tab2:
            st.markdown("### 向量数据库配置")
            
            vs_type = st.selectbox(
                "向量库类型",
                options=["faiss", "chroma"],
                index=0 if config.vector_store.store_type == "faiss" else 1,
                format_func=lambda x: "FAISS (默认)" if x == "faiss" else "Chroma",
                key="vs_type"
            )
            
            persist_dir = st.text_input(
                "向量库存储路径",
                value=config.vector_store.persist_dir,
                key="vs_persist_dir"
            )
            
            # Embedding配置
            st.markdown("#### Embedding配置")
            
            emb_provider = st.selectbox(
                "Embedding提供商",
                options=["huggingface", "openai"],
                index=0 if config.embedding.provider == "huggingface" else 1,
                format_func=lambda x: "HuggingFace (免费)" if x == "huggingface" else "OpenAI",
                key="emb_provider"
            )
            
            if emb_provider == "huggingface":
                emb_model = st.selectbox(
                    "Embedding模型",
                    options=["Qwen/Qwen3-Embedding-0.6B", "sentence-transformers/all-MiniLM-L6-v2", "BAAI/bge-large-zh-v1.5"],
                    index=0,
                    key="emb_model"
                )
                emb_device = st.selectbox(
                    "运行设备",
                    options=["cpu", "cuda"],
                    index=0,
                    format_func=lambda x: "CPU (默认)" if x == "cpu" else "GPU (CUDA)",
                    key="emb_device"
                )
            else:
                emb_model = st.selectbox(
                    "Embedding模型",
                    options=["text-embedding-3-small", "text-embedding-ada-002"],
                    index=0,
                    key="emb_model_openai"
                )
                emb_device = "cpu"
        
        with tab3:
            st.markdown("### 检索配置")
            
            search_type = st.selectbox(
                "检索策略",
                options=["similarity", "mmr", "similarity_score_threshold"],
                index=0,
                format_func=lambda x: {
                    "similarity": "相似度检索 (默认)",
                    "mmr": "MMR (多样性检索)",
                    "similarity_score_threshold": "阈值检索"
                }[x],
                key="search_type"
            )
            
            top_k = st.slider(
                "返回文档数量 (k)",
                min_value=1,
                max_value=10,
                value=config.retrieve.top_k,
                key="retrieve_top_k"
            )
            
            # MMR参数
            if search_type == "mmr":
                fetch_k = st.slider(
                    "MMR预取数量",
                    min_value=5,
                    max_value=30,
                    value=config.retrieve.mmr_fetch_k,
                    key="mmr_fetch_k"
                )
                lambda_mult = st.slider(
                    "MMR多样性系数",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.retrieve.mmr_lambda_mult,
                    step=0.1,
                    key="mmr_lambda_mult"
                )
            else:
                fetch_k = config.retrieve.mmr_fetch_k
                lambda_mult = config.retrieve.mmr_lambda_mult
            
            # 阈值参数
            if search_type == "similarity_score_threshold":
                score_threshold = st.slider(
                    "相似度阈值",
                    min_value=0.0,
                    max_value=1.0,
                    value=config.retrieve.score_threshold,
                    step=0.1,
                    key="score_threshold"
                )
            else:
                score_threshold = config.retrieve.score_threshold
        
        # 保存配置按钮
        st.markdown("---")
        
        if st.button("💾 保存配置", use_container_width=True):
            # 更新配置
            new_config = AppConfig(
                llm=LLMConfig(
                    provider=selected_provider,
                    api_key=api_key if selected_provider == "openai" else (api_key if 'api_key' in dir() else config.llm.api_key),
                    api_base=api_base,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=config.llm.timeout,
                    max_retries=config.llm.max_retries
                ),
                embedding=EmbeddingConfig(
                    provider=emb_provider,
                    model=emb_model,
                    device=emb_device
                ),
                vector_store=VectorStoreConfig(
                    store_type=vs_type,
                    persist_dir=persist_dir,
                    index_type=config.vector_store.index_type
                ),
                retrieve=RetrieveConfig(
                    search_type=search_type,
                    top_k=top_k,
                    mmr_fetch_k=fetch_k,
                    mmr_lambda_mult=lambda_mult,
                    score_threshold=score_threshold
                ),
                knowledge_base=config.knowledge_base.model_dump() if hasattr(config.knowledge_base, 'model_dump') else config.knowledge_base
            )
            
            st.session_state.config = new_config
            st.session_state.medical_rag = None  # 重置RAG实例
            st.success("配置已保存！")
            st.rerun()
        
        # 重建向量库按钮
        st.markdown("---")
        st.markdown("### 📚 知识库管理")
        
        if st.button("🔄 重建向量库", use_container_width=True):
            with st.spinner("正在重建向量库..."):
                success, message = build_vector_store(st.session_state.config)
                if success:
                    st.session_state.vector_store_built = True
                    st.session_state.medical_rag = None
                    st.success(message)
                else:
                    st.error(message)
        
        # 显示向量库状态
        vs_manager = VectorStoreManager(st.session_state.config)
        if vs_manager.exists():
            st.success("✅ 向量库已就绪")
        else:
            st.warning("⚠️ 向量库未构建")
        
        # 知识库路径
        st.markdown("---")
        st.markdown(f"**知识库路径**: `{config.knowledge_base.data_path}`")


def render_main_content():
    """渲染主内容区"""
    
    # 标题
    st.markdown('<h1 class="main-header">🏥 医疗RAG诊断系统</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">基于医学知识库的智能诊断助手</p>', unsafe_allow_html=True)
    
    # 检查向量库
    vs_manager = VectorStoreManager(st.session_state.config)
    if not vs_manager.exists():
        st.warning("⚠️ 向量库尚未构建，请在侧边栏点击「重建向量库」按钮")
        return
    
    # 症状输入
    st.markdown("### 🔍 请描述您的症状")
    
    symptom = st.text_area(
        "输入症状描述",
        placeholder="例如：发烧、咳嗽、胸痛、呼吸困难、乏力...",
        height=100,
        key="symptom_input"
    )
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        submitted = st.button("🩺 开始诊断", type="primary", use_container_width=True)
    
    with col2:
        if st.button("🗑️ 清空输入"):
            st.session_state.symptom_input = ""
            st.rerun()
    
    # 处理诊断
    if submitted and symptom:
        with st.spinner("正在分析症状，请稍候..."):
            try:
                rag = get_medical_rag()
                answer, docs = rag.diagnose(symptom)
                
                # 显示结果
                st.markdown("---")
                st.markdown("### 📋 诊断结果")
                
                if "对不起" in answer and "不在知识库中" in answer:
                    st.markdown(f'<div class="warning-box">{answer}</div>', unsafe_allow_html=True)
                elif "诊断失败" in answer:
                    st.error(answer)
                    st.info("请检查大模型配置是否正确，或尝试更换其他模型")
                    # 显示检索到的文档作为后备
                    if docs:
                        with st.expander("📄 检索到的相关资料"):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**{i}. {doc.metadata.get('source', '未知')}**")
                                st.text(doc.page_content[:800] + "..." if len(doc.page_content) > 800 else doc.page_content)
                else:
                    st.markdown(answer)
                    
                    # 显示检索到的文档
                    if docs:
                        with st.expander("📄 查看检索来源"):
                            for i, doc in enumerate(docs, 1):
                                st.markdown(f"**来源 {i}**: {doc.metadata.get('source', '未知')}")
                                st.text(doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content)
                                st.markdown("---")
                
                # 添加到历史
                st.session_state.chat_history.append({
                    "symptom": symptom,
                    "answer": answer
                })
                
            except Exception as e:
                st.error(f"诊断失败: {str(e)}")
    
    # 诊断历史
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("### 📜 诊断历史")
        
        for i, item in enumerate(reversed(st.session_state.chat_history[-5:])):
            with st.expander(f"症状: {item['symptom'][:50]}..."):
                st.markdown(f"**症状**: {item['symptom']}")
                st.markdown(f"**结果**: {item['answer'][:300]}...")
        
        if st.button("清空历史"):
            st.session_state.chat_history = []
            st.rerun()


def render_footer():
    """渲染页脚"""
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: #666; font-size: 0.8rem;'>"
        "医疗RAG诊断系统 | 仅供参考，不作为医疗建议 | 如有不适请及时就医"
        "</p>",
        unsafe_allow_html=True
    )


def main():
    """主函数"""
    init_session_state()
    render_sidebar()
    render_main_content()
    render_footer()


if __name__ == "__main__":
    main()
