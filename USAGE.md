# RepoScope 使用指南

## 📋 前置要求

- Python 3.9+
- Git
- 至少一个 API Key（OpenAI 或 Google）

## 🔧 环境配置

### 1. 安装依赖

```bash
# 进入项目目录（改成自己的路径）
cd /Users/kevin/Desktop/COMS6998/RepoScope-main

# 创建 conda 环境（推荐，faiss 需要 conda）
conda create -n reposcope python=3.9 -y
conda activate reposcope

# 安装依赖
pip install fastapi uvicorn python-dotenv adalflow transformers tiktoken requests openai watchfiles
conda install -c conda-forge faiss-cpu -y
```

### 2. 配置 API Key

在项目根目录创建 `.env` 文件：

```bash
touch .env
```

在 `.env` 文件中添加你的 API Key（至少需要一个）：

```env
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key_here

# 或 Google API Key
GOOGLE_API_KEY=your_google_api_key_here

# 可选配置
PORT=8001
```

**获取 API Key：**
- OpenAI: https://platform.openai.com/api-keys
- Google: https://makersuite.google.com/app/apikey

## 🚀 运行服务

### 启动后端服务

```bash
# 确保虚拟环境已激活
python -m api.main
```

服务将在 `http://localhost:8001` 启动。

## 💻 使用前端界面

### 打开前端

启动后端服务后，有两种方式打开前端：

#### 方式 1: 直接打开 HTML 文件（推荐）

1. 在项目根目录找到 `index.html` 文件
2. 用浏览器直接打开该文件（双击或在浏览器中打开）
3. 前端会自动连接到 `http://localhost:8001` 的 API 服务

#### 方式 2: 通过本地服务器访问

```bash
# 在项目根目录运行
python -m http.server 8080
# 然后访问 http://localhost:8080/index.html
```

### 使用步骤

1. **配置仓库**：
   - 在左侧边栏输入仓库 URL（如 `https://github.com/owner/repo`）
   - 选择仓库类型（GitHub/GitLab/Bitbucket）
   - （可选）输入访问令牌（私有仓库需要）

2. **选择模型**：
   - 选择 AI 模型提供商（OpenAI/Google）
   - 选择具体模型（如 gpt-4o, gemini-2.0-flash-exp）
   - 选择对话语言（中文/English 等）

3. **开始对话**：
   - 点击 "+ New Chat" 创建新会话
   - 在输入框输入问题
   - 点击 Send 或按 Enter 发送
   - 等待 AI 响应（首次查询可能需要几分钟处理仓库）

4. **管理会话**：
   - 在聊天历史中切换不同会话
   - 删除不需要的会话
   - 所有聊天记录自动保存到浏览器本地存储

## 📝 注意事项

- **首次查询较慢**：首次查询时，系统需要下载和处理仓库，可能需要几分钟
- **端口更改**：如果更改了端口，需要修改 `index.html` 中的 `API_BASE` 变量
- **私有仓库**：需要在 "Access Token" 字段输入 GitHub Personal Access Token

## 🔗 相关链接

- API 文档: http://localhost:8001/docs
- 健康检查: http://localhost:8001/health
