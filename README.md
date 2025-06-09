# 🤖 智能文档处理系统 (Intelligent Document Processing System)

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)
![AI](https://img.shields.io/badge/AI-Powered-purple.svg)

**基于 FastAPI 和 AI 技术的智能 OCR 图片分析与文档向量化检索系统**

[🚀 快速开始](#-快速开始) • [📖 功能特性](#-功能特性) • [🛠️ 配置说明](#️-配置说明) • [📁 项目结构](#-项目结构)

</div>

-------

## 🌟 项目亮点

- **🔥 智能流式输出**: 实时显示AI回答，URL自动转换为图片展示
- **🎯 多模态查询**: 支持纯文本、纯图片、文本+图片组合查询
- **⚡ 异步处理**: 高性能异步架构，支持并发查询
- **🧠 RAG技术**: 检索增强生成，结合知识库提供精准答案
- **🎨 现代界面**: 响应式设计，支持拖拽上传、进度展示
- **📊 向量检索**: 基于BGE-M3模型的语义搜索

## 📖 功能特性

### 🔍 智能查询分析
- **图片OCR分析**: 使用千问视觉模型分析图片中的错误信息
- **文档向量检索**: 在知识库中智能搜索相关解决方案  
- **流式答案生成**: 实时显示AI回答过程，URL自动转图片
- **多模态输入**: 支持文字描述、图片上传或两者结合

### 📚 文档处理系统
- **多格式支持**: TXT、DOCX、PDF文档自动解析
- **智能分割**: 自定义分隔符分割文档内容
- **向量化索引**: 创建高效的向量数据库索引
- **文档管理**: 查看处理历史、删除文档及索引

### 🎛️ 系统特性
- **实时进度**: 详细的处理步骤和进度展示
- **错误处理**: 完善的异常处理和用户提示
- **会话管理**: 支持查询取消和状态管理
- **安全可靠**: 文件验证和类型检查

## 🚀 快速开始

### 环境要求

- Python 3.8+ 或 Docker
- CUDA GPU（推荐，用于向量处理）
- 至少 4GB 内存

### 部署方式

#### 🐳 Docker 部署（推荐）

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd intelligent-document-processing
   ```

2. **配置API密钥**
   ```bash
   # 复制环境变量示例文件
   cp docker.env.example .env
   
   # 编辑 .env 文件，设置你的API密钥
   nano .env
   ```
   
   在 `.env` 文件中修改：
   ```bash
   DASHSCOPE_API_KEY=your-real-api-key-here
   ```

3. **启动服务**
   ```bash
   # 使用 docker-compose 启动
   docker-compose up -d
   
   # 查看日志
   docker-compose logs -f
   ```

4. **访问系统**
   
   打开浏览器访问：`http://localhost:8000`

5. **管理服务**
   ```bash
   # 停止服务
   docker-compose down
   
   # 重启服务
   docker-compose restart
   
   # 查看状态
   docker-compose ps
   ```

#### 🔧 手动部署

1. **克隆项目**
   ```bash
   git clone <your-repo-url>
   cd intelligent-document-processing
   ```

2. **安装依赖**
   ```bash
   pip install -r requirements.txt
   ```

3. **配置API密钥（⚠️ 必需步骤）**
   
   编辑 `app.py` 文件，设置您的 DashScope API Key：
   
   📍 **位置**: `app.py` 第 22 行
   
   ```python
   # 找到这行代码并修改:
   dashscope.api_key = ""  # 替换这个示例密钥
   
   # 改为你的真实API密钥:
   dashscope.api_key = "your-real-api-key-here"
   ```
   
   **📝 获取密钥**: 访问 [DashScope控制台](https://dashscope.console.aliyun.com/) 获取API Key

4. **启动服务**
   ```bash
   # 开发模式
   python main.py
   
   # 或使用 uvicorn
   uvicorn app:app --host 0.0.0.0 --port 8000 --reload
   ```

5. **访问系统**
   
   打开浏览器访问：`http://localhost:8000`

## 🛠️ 配置说明

### 🔑 API配置（重要！）

**必须配置DashScope API密钥才能使用系统**

📍 **配置位置**: `app.py` 文件第 22 行

```python
# 文件: app.py
# 行号: 22
# 修改后:
dashscope.api_key = "your-dashscope-api-key-here"  # 替换为你的真实API密钥
```

**获取API密钥步骤:**
1. 访问 [阿里云DashScope控制台](https://dashscope.console.aliyun.com/)
2. 注册/登录阿里云账号
3. 开通DashScope服务
4. 在控制台获取API Key
5. 将API Key替换到 `app.py` 第22行

### 🤖 模型配置（可选）
在 `rag_service.py` 中修改AI模型参数：

```python
# 文件: rag_service.py
# 类: RAGService.__init__()
self.model_name = "qwen-max"      # 可选: qwen-plus, qwen-turbo, qwen-max
self.temperature = 0.3            # 生成温度：0.0-1.0 (0=确定性，1=创造性)
self.max_context_length = 6000    # 最大上下文长度
self.max_tokens = 2000           # 最大生成token数
```

### 📂 存储路径配置（可选）
在 `app.py` 中修改文件存储路径：

```python
# 文件: app.py
# 行号: 约25-30行
UPLOAD_DIR = "uploads"              # 用户上传文件存储目录
VECTOR_DB_BASE_DIR = "vector"       # 向量数据库存储目录

# 可以修改为绝对路径，例如:
# UPLOAD_DIR = "/data/uploads"
# VECTOR_DB_BASE_DIR = "/data/vector"
```

### 🌐 服务器配置（可选）
在 `main.py` 中修改服务器运行参数：

```python
# 文件: main.py
# 函数: if __name__ == "__main__":
uvicorn.run(
    "app:app",
    host="0.0.0.0",        # 绑定IP地址 (0.0.0.0=所有接口, 127.0.0.1=仅本机)
    port=8000,             # 端口号
    reload=True            # 热重载 (生产环境建议设为False)
)
```

### 🔧 向量模型配置（高级）
在 `process.py` 中配置BGE-M3向量模型：

```python
# 文件: process.py
# 如果使用本地模型，修改模型路径:
model_path = "/path/to/your/bge-m3-model"

# 如果使用在线模型 (默认):
model = SentenceTransformer('BAAI/bge-m3')

# GPU/CPU配置:
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## 📁 项目结构

```
intelligent-document-processing/
├── 📄 README.md              # 项目说明文档
├── 📋 requirements.txt       # Python依赖包
├── 🐳 Dockerfile             # Docker镜像构建文件
├── 🐙 docker-compose.yml     # Docker容器编排文件
├── 🔑 docker.env.example     # 环境变量配置示例
├── 🚀 main.py               # 应用启动入口
├── 🏗️ app.py                # FastAPI应用主文件
├── 🔧 process.py            # 文档处理和向量化
├── 🧠 rag_service.py        # RAG服务和AI模型调用
├── 📁 templates/            # 前端模板文件
│   └── 🎨 index.html       # 主界面（响应式设计）
├── 📁 uploads/              # 文件上传目录
├── 📁 vector/               # 向量数据库存储
└── 📁 __pycache__/          # Python缓存文件
```

### 📋 文件详细说明

| 文件 | 功能说明 |
|-----|---------|
| `main.py` | 应用启动入口，配置服务器参数 |
| `app.py` | FastAPI主应用，包含所有API路由和业务逻辑 |
| `process.py` | 文档解析、向量化、检索等核心处理逻辑 |
| `rag_service.py` | RAG服务，处理AI模型调用和流式生成 |
| `templates/index.html` | 前端界面，包含所有UI交互和流式显示逻辑 |

## 🎯 核心功能说明

### 智能查询流程
1. **输入处理**: 接收文字和/或图片输入
2. **图片分析**: 使用千问视觉模型进行OCR识别
3. **向量检索**: 在知识库中搜索相关文档片段
4. **答案生成**: RAG模式生成结构化答案
5. **流式输出**: 实时显示回答，URL自动转换为图片

### 文档处理流程
1. **文件上传**: 支持拖拽和点击上传
2. **格式解析**: 自动识别并解析文档内容
3. **内容分割**: 按分隔符分割为语义片段
4. **向量化**: 使用BGE-M3模型生成向量
5. **索引构建**: 创建FAISS向量数据库

## 🔧 高级配置

### 模型配置优化
```python
# 在 rag_service.py 中调整模型参数
class RAGService:
    def __init__(self):
        self.model_name = "qwen-max"     # 模型选择
        self.temperature = 0.3           # 创造性：0.0-1.0
        self.max_tokens = 2000          # 最大生成长度
        self.top_p = 0.8                # 采样参数
```

### 向量检索优化
```python
# 在 process.py 中调整检索参数
def search_similar_segments(query, top_k=5):
    # top_k: 返回最相似的段落数量
    # 可根据需要调整相似度阈值
```

## 🐛 故障排除

### 常见问题

1. **API密钥错误**
   ```
   错误: API请求失败
   解决: 检查 app.py 中的 dashscope.api_key 配置或 .env 文件中的 DASHSCOPE_API_KEY
   ```

2. **依赖包安装失败**
   ```bash
   # 使用国内镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
   ```

3. **GPU内存不足**
   ```python
   # 在 process.py 中使用CPU版本
   model = SentenceTransformer('BAAI/bge-m3', device='cpu')
   ```

4. **端口占用**
   ```bash
   # 修改 main.py 中的端口
   uvicorn.run("app:app", port=8001)
   ```

### 🐳 Docker 问题

5. **Docker 容器启动失败**
   ```bash
   # 查看详细日志
   docker-compose logs ocr-system
   
   # 重新构建镜像
   docker-compose build --no-cache
   ```

6. **API密钥未生效**
   ```bash
   # 检查 .env 文件是否正确配置
   cat .env
   
   # 重启容器使环境变量生效
   docker-compose restart
   ```

7. **容器内存不足**
   ```yaml
   # 在 docker-compose.yml 中添加内存限制
   services:
     ocr-system:
       deploy:
         resources:
           limits:
             memory: 4G
           reservations:
             memory: 2G
   ```

8. **数据持久化问题**
   ```bash
   # 确保挂载目录权限正确
   sudo chown -R 1000:1000 uploads vector logs
   
   # 查看容器挂载状态
   docker-compose exec ocr-system ls -la /app/
   ```

9. **网络连接问题**
   ```bash
   # 检查容器网络
   docker network ls
   docker-compose exec ocr-system ping google.com
   
   # 重置Docker网络
   docker-compose down
   docker network prune
   docker-compose up -d
   ```

10. **健康检查失败**
    ```bash
    # 查看健康检查状态
    docker-compose ps
    
    # 手动测试健康检查
    docker-compose exec ocr-system python -c "import requests; print(requests.get('http://localhost:8000').status_code)"
    ```

## 📈 性能优化建议

- **GPU加速**: 安装 `torch-gpu` 版本以使用GPU加速
- **内存优化**: 调整向量检索的 `batch_size` 参数
- **并发处理**: 增加 uvicorn 的 `workers` 数量
- **缓存优化**: 启用向量模型缓存以加快启动速度

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

1. Fork 本项目
2. 创建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 打开 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 🙏 致谢

- [FastAPI](https://fastapi.tiangolo.com/) - 现代、快速的Web框架
- [DashScope](https://dashscope.aliyun.com/) - 阿里云AI模型服务
- [BGE-M3](https://huggingface.co/BAAI/bge-m3) - 多语言向量化模型
- [FAISS](https://github.com/facebookresearch/faiss) - 高效相似性搜索库

---

<div align="center">

**如果觉得这个项目对你有帮助，请给个 ⭐ Star 支持一下！**

</div> 
