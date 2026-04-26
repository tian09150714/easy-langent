# 医疗RAG诊断系统

该系统是基于LangChain的医疗知识库问答系统，支持通过症状描述进行智能诊断和建议。主要是用于langchain的学习。用的知识库为网络提供的免费json文件。**注意：** 该系统只是演示系统，请不要做为真实的医疗辅助系统来用。知识库信息并没有进行严密的医学认证！！！

# 作者信息

学员：道法自然

频道：easy-langent

群组：1

## 功能特性

- 🔍 **智能诊断**: 根据用户描述的症状，从医学知识库中检索相关信息并提供诊断
- 🏥 **治疗方案**: 提供针对性的治疗方案和注意事项
- ⚙️ **灵活配置**: 支持配置不同的大模型、向量数据库和检索策略
- 💾 **持久化存储**: 支持FAISS和Chroma两种向量数据库
- 🎨 **友好界面**: 基于Streamlit的现代化Web界面

## 项目结构

```
medical_rag/
├── app.py                     # Streamlit Web应用主入口
├── config_manager.py          # 配置管理模块
├── medical_rag.py             # RAG核心组件
├── vector_store_manager.py    # 向量数据库管理
├── build_knowledge_base.py    # 知识库构建脚本
├── requirements.txt          # 依赖列表
├── config.yaml               # 配置文件
├── .env.example             # 环境变量示例
└── README.md                # 本文档
```

## 快速开始

### 0. 下载医学知识库文件

链接：https://pan.quark.cn/s/89e7981209c3
提取码：qGKt
解压获得medical.json
放在和medical_rag同级目录下

### 1. 安装依赖

```bash
cd medical_rag
pip install -r requirements.txt
```

### 2. 配置环境

复制环境变量示例文件并编辑:

```bash
cp .env.example .env
```

编辑`.env`文件，配置大模型以及向量数据库。下面是一个例子:

```env
MOTSCOPE_OPENAI_API_KEY=xxxxxxxxxx"
MOTSCOPE_OPENAI_API_BASE="https://api-inference.modelscope.cn/v1"
MODELSCOPE_MODEL = "Qwen/Qwen3.5-35B-A3B"


LOCAL_OPENAI_API_KEY="ollama"
LOCAL_OPENAI_API_BASE="http://localhost:11434/v1"
LOCAL_MODEL = "gemma4:e2b"


EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
EMBEDDING_TYPE = "qwen"

PERSIST_DIR = "./vector_store"
VECTOR_STORE_TYPE = "faiss"
```

### 3. 构建知识库

首次运行需要构建向量数据库:

```bash
# 命令行构建
python build_knowledge_base.py --json_path ../medical.json

# 或在Web界面中点击"重建向量库"按钮
```

可选参数:

- `--vector_store_type`: 向量库类型 (faiss/chroma)
- `--persist_dir`: 向量库存储路径
- `--embedding_provider`: Embedding提供商 (huggingface/openai)
- `--embedding_model`: Embedding模型
- `--device`: 运行设备 (cpu/cuda)

### 4. 启动应用

```bash
streamlit run app.py
```

应用将在浏览器中打开，默认地址: http://localhost:8501

## 使用说明

### 诊断流程

1. **输入症状**: 在文本框中描述您的症状
2. **开始诊断**: 点击"开始诊断"按钮
3. **查看结果**: 系统将从知识库中检索相关信息并给出诊断建议

### 配置调整

在侧边栏中可以调整以下配置:

#### 大模型配置

- **提供商**: ModelScope / OpenAI / Ollama
- **模型选择**: 根据提供商显示可用模型
- **Temperature**: 控制回答的创造性 (0.0-1.0)
- **最大Token数**: 回答的最大长度

#### 向量数据库配置

- **类型**: FAISS (默认) / Chroma
- **存储路径**: 向量库持久化目录
- **Embedding**: 模型和设备选择

#### 检索配置

- **检索策略**:
  - `similarity`: 纯相似度检索
  - `mmr`: 多样性检索，兼顾相关性和多样性
  - `similarity_score_threshold`: 带阈值的相似度检索
- **返回数量**: 检索返回的文档数量

## 注意事项

⚠️ **免责声明**:

- 本系统仅供参考，不作为医疗诊断或治疗的依据
- 如有身体不适，请及时就医
- 系统回答基于知识库内容，可能不完整或存在偏差

## 常见问题

### Q: 首次运行报"向量库不存在"错误

A: 这是正常的，需要先构建向量库。在侧边栏点击"重建向量库"按钮，或运行:

```bash
python build_knowledge_base.py --json_path ../medical.json
```

### Q: 诊断结果说"不在知识库中"

A: 说明您的症状描述与知识库中的疾病没有匹配到。可以尝试:

1. 使用更具体的症状描述
2. 增加检索的返回数量 (k值)
3. 检查知识库文件路径是否正确

### Q: 如何切换向量数据库?

A: 在侧边栏的"向量库配置"中选择类型，然后重建向量库。

### Q: 点击开始诊断长时间没反应

A： 可能是大模型配置不对，请点击左侧的测试大模型连接来测试大模型配置是否正确。

## 开发说明

### 配置管理

系统支持两种配置方式:

1. **环境变量**: 在`.env`文件中配置
2. **Web界面**: 在侧边栏实时调整

配置保存后会立即生效，无需重启应用。

### 自定义知识库

如果需要使用其他医学知识库，只需修改`medical.json`文件的路径或内容。每条记录应包含:

- `name`: 疾病名称
- `desc`: 疾病描述
- `symptom`: 症状列表
- `cure_way`: 治疗方法
- 等其他可选字段

## License

MIT License
