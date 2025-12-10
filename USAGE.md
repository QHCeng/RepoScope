# RepoScope 使用指南

## 📋 前置要求

- Python 3.9+ 
- Git
- 至少一个 API Key（OpenAI、Google 或其他支持的提供商）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 进入项目目录
cd /Users/kevin/Desktop/COMS6998/RepoScope-main

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖（如果有 requirements.txt）
pip install fastapi uvicorn python-dotenv adalflow transformers faiss-cpu tiktoken requests
```

### 2. 配置环境变量

创建 `.env` 文件在项目根目录：

```bash
touch .env
```

在 `.env` 文件中添加你的 API Key：

```env
# 必需的 API Key（至少需要一个）
OPENAI_API_KEY=your_openai_api_key_here
GOOGLE_API_KEY=your_google_api_key_here

# 可选的配置
PORT=8001
DEEPWIKI_EMBEDDER_TYPE=openai  # 可选: openai, google, ollama
LOG_LEVEL=INFO

# 如果使用 Ollama（本地模型）
OLLAMA_HOST=http://localhost:11434
```

**获取 API Key：**
- OpenAI: https://platform.openai.com/api-keys
- Google: https://makersuite.google.com/app/apikey

### 3. 启动 API 服务

```bash
# 确保虚拟环境已激活
python -m api.main
```

服务将在 `http://localhost:8001` 启动。

### 4. 打开前端界面

启动服务后，在项目根目录找到 `index.html` 文件，直接用浏览器打开即可。

## 🎨 Web Interface (Frontend)

**The easiest way to use RepoScope!**

### 访问前端界面

启动服务器后，有两种方式访问前端界面：

#### 方式 1: 直接打开 HTML 文件（推荐）

1. 在项目根目录找到 `index.html` 文件
2. 用浏览器直接打开该文件（双击或在浏览器中打开）
3. 前端会自动连接到 `http://localhost:8001` 的 API 服务

#### 方式 2: 通过本地服务器访问

如果你有本地 HTTP 服务器（如 Python 的 `http.server`），可以：

```bash
# 在项目根目录运行
python -m http.server 8080
# 然后访问 http://localhost:8080/index.html
```

### 界面功能

前端界面提供了完整的可视化操作界面，包括：

#### 📋 左侧边栏配置

- **Repository Settings（仓库设置）**:
  - 📝 **Repository URL**: 输入 GitHub/GitLab/Bitbucket 仓库 URL
  - 🔧 **Repository Type**: 选择仓库类型（GitHub/GitLab/Bitbucket/Local）
  - 🔑 **Access Token**: 可选，用于私有仓库访问
  - 🤖 **Model Provider**: 选择 AI 模型提供商（OpenAI/Google）
  - 🎯 **Model**: 选择具体的模型（如 gpt-4o, gemini-2.0-flash-exp）
  - 🌐 **Language**: 选择对话语言（中文/English/日本語 等）
  - 🔬 **Deep Research Mode**: 启用深度研究模式（多轮深入分析）

- **Chat History（聊天历史）**:
  - ➕ **+ New Chat 按钮**: 创建新的聊天会话
  - 📚 **聊天历史列表**: 显示所有历史聊天会话
    - 显示会话标题、消息数量、最后更新时间
    - 点击会话可快速切换
    - 鼠标悬停显示删除按钮

#### 💬 主聊天区域

- **聊天标签页**: 顶部显示所有活跃的聊天会话标签
  - 点击标签切换不同会话
  - 点击 × 关闭标签（删除会话）
  
- **消息显示区域**:
  - 实时流式响应显示
  - Markdown 格式渲染（代码块、列表、标题等）
  - 自动滚动到最新消息

- **输入区域**:
  - 文本输入框
  - Send 按钮
  - 支持 Enter 键快速发送（Shift+Enter 换行）

#### ✨ 核心特性

- **💾 自动保存**: 所有聊天会话自动保存到浏览器本地存储
- **🔄 会话管理**: 
  - 创建多个独立的聊天会话
  - 每个会话关联特定的仓库和配置
  - 会话数据持久化保存
- **📊 实时状态**: 显示 API 连接状态（Connected/Disconnected）
- **🎨 现代化 UI**: 美观的界面设计，流畅的用户体验
- **⚡ 实时流式响应**: 消息逐字显示，无需等待完整响应

### 使用步骤

1. **启动后端服务**:
   ```bash
   python -m api.main
   ```

2. **打开前端界面**:
   - 直接打开项目根目录的 `index.html` 文件

3. **配置仓库**:
   - 在左侧边栏输入仓库 URL
   - 选择仓库类型
   - （可选）输入访问令牌（私有仓库）

4. **选择模型和语言**:
   - 选择 AI 模型提供商
   - 选择具体模型
   - 选择对话语言

5. **开始对话**:
   - 在输入框输入问题
   - 点击 Send 或按 Enter 发送
   - 等待 AI 响应（首次查询可能需要几分钟处理仓库）

6. **管理聊天会话**:
   - 点击 "+ New Chat" 创建新会话
   - 在聊天历史中切换不同会话
   - 删除不需要的会话

### 使用技巧

- **首次使用**: 输入仓库 URL 后，系统会自动下载和处理仓库，这可能需要几分钟
- **多仓库管理**: 可以为不同仓库创建不同的聊天会话
- **会话切换**: 点击侧边栏的聊天历史或顶部的标签页快速切换
- **配置保存**: 每个会话会保存其关联的仓库 URL、模型配置和语言设置
- **自动清理**: 空的"New Chat"会话会在加载时自动清理

### Alternative Access Points:

- **Web Interface**: 直接打开 `index.html` 文件（推荐）
- **API Documentation**: http://localhost:8001/docs (Interactive API docs)
- **Health Check**: http://localhost:8001/health
- **Root Endpoint**: http://localhost:8001/ (查看所有可用端点)

## 📖 使用方法

### 方法 1: 通过 API 端点使用

#### 1.1 生成 Wiki 文档

```bash
curl -X POST "http://localhost:8001/api/wiki/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": {
      "owner": "owner_name",
      "repo": "repo_name",
      "type": "github",
      "repoUrl": "https://github.com/owner_name/repo_name"
    },
    "language": "en",
    "provider": "google",
    "model": "gemini-2.5-flash"
  }'
```

#### 1.2 与代码库对话（RAG）

```bash
curl -X POST "http://localhost:8001/chat/completions/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "messages": [
      {
        "role": "user",
        "content": "这个仓库的主要功能是什么？"
      }
    ],
    "provider": "google",
    "model": "gemini-2.5-flash",
    "type": "github",
    "language": "zh"
  }'
```

#### 1.3 获取本地仓库结构

```bash
curl "http://localhost:8001/local_repo/structure?path=/path/to/local/repo"
```

#### 1.4 获取已处理的项目列表

```bash
curl "http://localhost:8001/api/processed_projects"
```

#### 1.5 获取缓存的 Wiki

```bash
curl "http://localhost:8001/api/wiki_cache?owner=owner_name&repo=repo_name&repo_type=github&language=en"
```

### 方法 2: 使用 Python 代码

#### 2.1 生成 Wiki

```python
from api.page import WikiService, RepoInfo, LLMClient

# 初始化 LLM 客户端
llm = LLMClient(
    model="gpt-4o-mini",
    api_key="your_openai_api_key"
)

# 配置仓库信息
repo = RepoInfo(
    owner="owner_name",
    repo="repo_name",
    type="github",
    repoUrl="https://github.com/owner_name/repo_name"
)

# 创建 Wiki 服务
wiki_service = WikiService(
    repo=repo,
    llm=llm,
    language="en"
)

# 生成 Wiki
wiki = wiki_service.generate_wiki(use_cache=True)

# 导出为 Markdown
wiki_service.export_markdown(wiki, "output.md")
```

#### 2.2 使用 RAG 进行代码问答

```python
from api.rag import RAG

# 初始化 RAG
rag = RAG(provider="google", model="gemini-2.5-flash")

# 准备检索器（会自动下载和索引仓库）
rag.prepare_retriever(
    repo_url_or_path="https://github.com/owner/repo",
    type="github"
)

# 提问
result = rag("这个项目的主要功能是什么？", language="zh")
print(result[0].answer)
```

### 方法 3: Deep Research 模式

在聊天请求的消息中添加 `[DEEP RESEARCH]` 标签，系统会进行多轮深入研究：

```bash
curl -X POST "http://localhost:8001/chat/completions/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "messages": [
      {
        "role": "user",
        "content": "[DEEP RESEARCH] 详细分析这个项目的架构设计"
      }
    ],
    "provider": "google",
    "model": "gemini-2.5-flash",
    "type": "github"
  }'
```

## 🔧 高级配置

### 配置文件位置

项目会在以下位置查找配置文件（JSON 格式）：
- `api/config/generator.json` - 模型提供商配置
- `api/config/embedder.json` - 嵌入模型配置
- `api/config/repo.json` - 仓库过滤配置
- `api/config/lang.json` - 语言配置

### 文件过滤

在请求中可以指定包含或排除的文件：

```json
{
  "repo_url": "https://github.com/owner/repo",
  "excluded_dirs": ".venv\nnode_modules\ntests",
  "excluded_files": "*.pyc\n*.log",
  "included_dirs": "src\nlib",
  "included_files": "*.py\n*.js"
}
```

### 支持的仓库类型

- `github` - GitHub 仓库
- `gitlab` - GitLab 仓库
- `bitbucket` - Bitbucket 仓库
- `local` - 本地路径

### 支持的语言

- `en` - English
- `zh` - 中文
- `ja` - 日本語
- `kr` - 한국어
- `es` - Español
- `fr` - Français
- 等等...

## 📝 常见问题

### Q: 如何访问前端界面？
A: 启动后端服务后，直接用浏览器打开项目根目录的 `index.html` 文件即可。前端会自动连接到 `http://localhost:8001` 的 API 服务。

### Q: 前端界面显示 "Disconnected" 怎么办？
A: 
1. 确保后端服务正在运行（`python -m api.main`）
2. 检查服务是否在 `http://localhost:8001` 运行
3. 刷新页面重试
4. 检查浏览器控制台是否有错误信息

### Q: 为什么刷新页面后会出现新的 "New Chat"？
A: 这是正常行为。如果之前没有聊天会话，系统会自动创建一个新的聊天会话。如果已有聊天会话，会加载最近的会话。空的"New Chat"会话会在加载时自动清理。

### Q: 聊天历史保存在哪里？
A: 聊天历史保存在浏览器的 localStorage 中，键名为 `reposcope_chat_sessions`。即使关闭浏览器，聊天记录也会保留。

### Q: 如何删除聊天会话？
A: 
- 方法 1: 在侧边栏的聊天历史中，鼠标悬停在会话上，点击 "Delete" 按钮
- 方法 2: 在顶部的聊天标签页中，点击会话标签右侧的 × 按钮

### Q: 如何查看 API 文档？
A: 启动服务后访问 http://localhost:8001/docs

### Q: 如何更改端口？
A: 在 `.env` 文件中设置 `PORT=8002`，或通过环境变量 `export PORT=8002`。注意：如果更改了端口，需要修改 `index.html` 中的 `API_BASE` 变量。

### Q: 支持私有仓库吗？
A: 支持，在前端界面的 "Access Token" 字段中输入你的 GitHub Personal Access Token，或在 API 请求中提供 `token` 参数。

### Q: 首次查询为什么很慢？
A: 首次查询时，系统需要：
1. 下载仓库（如果尚未下载）
2. 读取所有文件
3. 生成嵌入向量（embeddings）
这个过程可能需要几分钟，取决于仓库大小。后续查询会很快。

### Q: 数据存储在哪里？
A: 
- 仓库克隆: `~/.adalflow/repos/`
- 向量数据库: `~/.adalflow/databases/`
- Wiki 缓存: `~/.adalflow/wikicache/`
- 前端聊天历史: 浏览器 localStorage

### Q: 如何清理缓存？
A: 
- **Wiki 缓存**: 使用 DELETE 端点：
  ```bash
  curl -X DELETE "http://localhost:8001/api/wiki_cache?owner=owner&repo=repo&repo_type=github&language=en"
  ```
- **前端聊天历史**: 在浏览器开发者工具中清除 localStorage，或删除特定键 `reposcope_chat_sessions`
- **仓库和数据库**: 手动删除 `~/.adalflow/` 目录下的相应文件

### Q: 前端界面支持哪些浏览器？
A: 推荐使用现代浏览器（Chrome、Firefox、Safari、Edge 的最新版本）。需要支持：
- ES6+ JavaScript
- localStorage API
- Fetch API
- CSS Grid/Flexbox

### Q: 如何在前端使用 Deep Research 模式？
A: 在左侧边栏勾选 "Deep Research Mode" 复选框，然后在输入框中输入问题。系统会自动在问题前添加 `[DEEP RESEARCH]` 标签，进行多轮深入研究。

## 🎯 使用示例场景

### 场景 1: 使用前端界面快速了解新项目（推荐）

**步骤：**
1. 启动后端服务：`python -m api.main`
2. 打开 `index.html` 文件
3. 在左侧边栏输入仓库 URL（如 `https://github.com/owner/repo`）
4. 选择模型和语言
5. 点击 "+ New Chat" 创建新会话
6. 在输入框输入问题，如："这个项目的主要功能是什么？"
7. 等待 AI 响应（首次查询可能需要几分钟处理仓库）
8. 继续提问了解更多细节

**优势：**
- 可视化界面，操作简单
- 实时流式响应，体验流畅
- 自动保存聊天历史
- 支持多仓库管理

### 场景 2: 代码审查和深入分析

**使用前端界面：**
1. 创建新的聊天会话
2. 输入仓库 URL
3. 启用 "Deep Research Mode"
4. 输入问题，如："详细分析这个项目的架构设计"
5. 系统会进行多轮深入研究，提供详细分析

**或使用 API：**
```bash
curl -X POST "http://localhost:8001/chat/completions/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "messages": [{
      "role": "user",
      "content": "[DEEP RESEARCH] 详细分析这个项目的架构设计"
    }],
    "provider": "google",
    "model": "gemini-2.5-flash",
    "type": "github"
  }'
```

### 场景 3: 多项目对比分析

**使用前端界面：**
1. 为第一个项目创建聊天会话 A
2. 输入第一个仓库 URL，进行提问
3. 点击 "+ New Chat" 创建会话 B
4. 输入第二个仓库 URL，进行提问
5. 在聊天历史中切换两个会话，对比分析结果

### 场景 4: 文档生成和导出

**使用 API：**
```bash
# 生成 Wiki 文档
curl -X POST "http://localhost:8001/api/wiki/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "repo": {
      "owner": "owner_name",
      "repo": "repo_name",
      "type": "github",
      "repoUrl": "https://github.com/owner_name/repo_name"
    },
    "language": "en",
    "provider": "google",
    "model": "gemini-2.5-flash"
  }'

# 导出为 Markdown 或 JSON 格式
curl -X POST "http://localhost:8001/export/wiki" \
  -H "Content-Type: application/json" \
  -d '{
    "repo_url": "https://github.com/owner/repo",
    "pages": [...],
    "format": "markdown"
  }'
```

### 场景 5: 团队协作和知识分享

**使用前端界面：**
1. 团队成员各自创建聊天会话
2. 针对同一项目进行不同角度的提问
3. 聊天历史自动保存，可随时回顾
4. 通过切换会话查看不同成员的分析结果

## 🔗 相关资源

- API 文档: http://localhost:8001/docs
- 健康检查: http://localhost:8001/health
- 根端点: http://localhost:8001/ (查看所有可用端点)

