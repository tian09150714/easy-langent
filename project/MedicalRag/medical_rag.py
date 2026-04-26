"""医疗RAG核心组件"""

import os
import json
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from config_manager import get_config, AppConfig, LLMConfig
from vector_store_manager import VectorStoreManager


# 医疗问答系统提示词模板
MEDICAL_QA_SYSTEM_PROMPT = """你是一位专业的医疗助手，擅长根据提供的医学知识库信息为用户提供健康咨询和诊断建议。

**重要提示**：
1. 你的回答必须严格基于提供的知识库内容，不要编造或添加知识库中没有的信息
2. 如果知识库中没有相关信息，请明确告知用户"对不起，该症状不在知识库中记载，请到医院检查"
3. 回答应该专业、准确、易于理解
4. 请用中文回答

**回答格式**：
如果找到相关信息，请按以下格式回答：
- **可能的疾病**：疾病名称
- **疾病描述**：简要描述
- **主要症状**：相关症状列表
- **病因**：疾病原因
- **治疗方案**：推荐的治疗方法
- **注意事项**：预防措施和日常护理建议

如果未找到相关信息：
直接回答"对不起，该症状不在知识库中记载，请到医院检查"
"""

MEDICAL_QA_USER_PROMPT = """请根据以下知识库内容回答用户问题。

**用户问题**：{question}

**知识库内容**：
{context}

请严格按照上述格式回答用户问题。"""


class MedicalRAG:
    """医疗RAG系统核心类"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or get_config()
        self.vector_store_manager = VectorStoreManager(self.config)
        self._llm = None
        self._rag_chain = None
    
    @property
    def llm(self):
        """获取大模型"""
        if self._llm is None:
            self._llm = self._create_llm()
        return self._llm
    
    def _create_llm(self) -> ChatOpenAI:
        """创建大模型"""
        llm_cfg = self.config.llm
        
        # 处理API密钥
        api_key = llm_cfg.api_key
        if not api_key or api_key == "ollama":
            api_key = "ollama"  # Ollama使用此值
        
        return ChatOpenAI(
            model=llm_cfg.model,
            api_key=api_key,
            base_url=llm_cfg.api_base or None,
            temperature=llm_cfg.temperature,
            max_tokens=llm_cfg.max_tokens,
            timeout=llm_cfg.timeout,
            max_retries=llm_cfg.max_retries,
        )
    
    def _format_documents(self, documents: List[Document]) -> str:
        """格式化文档为字符串"""
        if not documents:
            return ""
        
        formatted_docs = []
        for i, doc in enumerate(documents, 1):
            source = doc.metadata.get("source", "未知来源")
            content = doc.page_content
            formatted_docs.append(f"[文档{i}]\n来源: {source}\n内容: {content}")
        
        return "\n\n".join(formatted_docs)
    
    def _build_rag_chain(self, retriever):
        """构建RAG链"""
        # 提示词
        prompt = ChatPromptTemplate.from_messages([
            ("system", MEDICAL_QA_SYSTEM_PROMPT),
            ("human", MEDICAL_QA_USER_PROMPT)
        ])
        
        # RAG链
        rag_chain = (
            {
                "context": retriever | self._format_documents,
                "question": RunnablePassthrough()
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
    
    def _get_retriever(self):
        """获取检索器"""
        ret_cfg = self.config.retrieve
        return self.vector_store_manager.as_retriever(
            search_type=ret_cfg.search_type,
            k=ret_cfg.top_k,
            fetch_k=ret_cfg.mmr_fetch_k,
            lambda_mult=ret_cfg.mmr_lambda_mult,
            score_threshold=ret_cfg.score_threshold
        )
    
    @property
    def rag_chain(self):
        """获取RAG链"""
        if self._rag_chain is None:
            retriever = self._get_retriever()
            self._rag_chain = self._build_rag_chain(retriever)
        return self._rag_chain
    
    def diagnose(self, symptom: str) -> Tuple[str, List[Document]]:
        """
        根据症状进行诊断
        
        Args:
            symptom: 用户描述的症状
            
        Returns:
            Tuple[str, List[Document]]: 诊断结果和检索到的文档
        """
        try:
            # 获取检索器
            retriever = self._get_retriever()
            
            # 检查是否有相关文档
            retrieved_docs = retriever.invoke(symptom)
            
            # 如果没有找到相关文档，返回提示信息
            if not retrieved_docs:
                return "对不起，该症状不在知识库中记载，请到医院检查", []
            
            # 使用RAG链生成回答
            try:
                answer = self.rag_chain.invoke(symptom)
            except Exception as e:
                error_msg = str(e)
                if "null value" in error_msg.lower() or "choices" in error_msg:
                    # API返回格式问题，使用简单检索结果
                    docs_content = self._format_documents(retrieved_docs)
                    if docs_content:
                        return (
                            f"根据检索到的医学资料，您描述的症状可能与以下疾病相关：\n\n{docs_content}\n\n"
                            "抱歉，大模型暂时无法生成详细诊断建议，请参考以上资料或到医院就诊。",
                            retrieved_docs
                        )
                    else:
                        return "对不起，该症状不在知识库中记载，请到医院检查", []
                raise
            
            # 检查回答是否表示未找到相关信息
            if "对不起" in answer and "不在知识库中" in answer:
                return answer, []
            
            return answer, retrieved_docs
            
        except Exception as e:
            return f"诊断失败: {str(e)}", []
    
    def get_related_diseases(self, symptom: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        获取与症状相关的疾病列表
        
        Args:
            symptom: 症状描述
            top_k: 返回数量
            
        Returns:
            List[Dict]: 相关疾病列表
        """
        docs_with_scores = self.vector_store_manager.similarity_search_with_score(
            symptom, k=top_k
        )
        
        results = []
        for doc, score in docs_with_scores:
            # 相似度过低则跳过
            if score > 2.0:  # 可调整阈值
                continue
            results.append({
                "content": doc.page_content,
                "score": score,
                "source": doc.metadata.get("source", "未知")
            })
        
        return results
    
    def rebuild_chain(self):
        """重建RAG链（当配置改变时）"""
        self._rag_chain = None
    
    def update_config(self, config: AppConfig):
        """更新配置"""
        self.config = config
        self.vector_store_manager.config = config
        self._llm = None
        self._rag_chain = None


def load_medical_documents(json_path: str) -> List[Document]:
    """
    从JSON文件加载医疗文档
    
    Args:
        json_path: JSON文件路径
        
    Returns:
        List[Document]: 文档列表
    """
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"知识库文件不存在: {json_path}")
    
    documents = []
    
    with open(json_path, 'r', encoding='utf-8') as f:
        # 处理JSONL格式
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                
                # 构建文档内容
                content_parts = []
                
                if name := data.get("name"):
                    content_parts.append(f"疾病名称：{name}")
                if desc := data.get("desc"):
                    content_parts.append(f"疾病描述：{desc}")
                if category := data.get("category"):
                    content_parts.append(f"疾病分类：{', '.join(category) if isinstance(category, list) else category}")
                if symptom := data.get("symptom"):
                    symptoms_str = ', '.join(symptom) if isinstance(symptom, list) else symptom
                    content_parts.append(f"主要症状：{symptoms_str}")
                if cause := data.get("cause"):
                    content_parts.append(f"病因：{cause}")
                if prevent := data.get("prevent"):
                    content_parts.append(f"预防措施：{prevent}")
                if cure_way := data.get("cure_way"):
                    cure_str = ', '.join(cure_way) if isinstance(cure_way, list) else cure_way
                    content_parts.append(f"治疗方法：{cure_str}")
                if cure_department := data.get("cure_department"):
                    dept_str = ', '.join(cure_department) if isinstance(cure_department, list) else cure_department
                    content_parts.append(f"就诊科室：{dept_str}")
                if check := data.get("check"):
                    check_str = ', '.join(check) if isinstance(check, list) else check
                    content_parts.append(f"检查项目：{check_str}")
                if drug_detail := data.get("drug_detail"):
                    drug_str = ', '.join(drug_detail[:5]) if isinstance(drug_detail, list) and len(drug_detail) > 5 else (', '.join(drug_detail) if isinstance(drug_detail, list) else drug_detail)
                    content_parts.append(f"相关药物：{drug_str}")
                if acompany := data.get("acompany"):
                    acompany_str = ', '.join(acompany) if isinstance(acompany, list) else acompany
                    content_parts.append(f"可能伴随：{acompany_str}")
                
                content = "\n".join(content_parts)
                
                if content:
                    doc = Document(
                        page_content=content,
                        metadata={
                            "source": data.get("name", "未知疾病"),
                            "disease_name": data.get("name", ""),
                            "category": data.get("category", [])
                        }
                    )
                    documents.append(doc)
                    
            except json.JSONDecodeError:
                continue
    
    return documents
