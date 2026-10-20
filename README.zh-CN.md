[English](README.md) | **简体中文**

# 📚 RAG File System（文档问答系统）

**和你自己的文档对话。上传文件、提问、得到带出处的回答。**

> 如果你能把一堆书、笔记和报告丢进同一个应用，然后直接*问*它们问题，会怎么样？这个项目就是做这件事的。

---

## 🎯 这是什么？

一个基于检索增强生成（RAG）的小型**文档问答应用**。你上传文件，应用把它们切成可搜索的片段；你提问时，它先找出最相关的段落，再交给 AI 模型写出答案。

### 功能

| 功能 | 说明 |
|---|---|
| **多种文件格式** | `.txt` `.md` `.pdf` `.docx` `.pptx` `.epub` `.html` `.rtf`，以及源代码文件 |
| **多本书同时提问** | 一次最多跨 5 本书提问，并合并结果（RRF 融合） |
| **流式回答** | 通过 Server-Sent Events 逐字显示回答 |
| **更好的搜索** | 父子分块、可选的混合搜索、可选的 cross-encoder 重排序 |
| **版本历史** | 重新上传同名文件会保留旧版本（每个文件最多 10 个） |
| **聊天会话** | 每次对话都会保存，并可导出 |
| **界面内设置** | 无需改代码，就能切换模型、温度和检索选项 |

---

## 🏗️ 工作原理

```
 上传文件 ──► 加载器 ──► 分块 ──► 向量嵌入 ──► FAISS 索引
                                                  │
 你的问题 ──► 检索 top-k ──► （重排序）──► AI 模型 ──► 回答 + 出处
```

1. **加载：** `document_loader.py` 读取各种格式的文件。
2. **分块：** 文本被切成较大的父块和较小的子块。
3. **嵌入：** `embedding_manager.py` 把片段转成向量，存进 FAISS 索引。
4. **提问：** `query_engine.py` 找出最佳片段，可选地重排序，然后连同你的问题一起发给模型。

---

## 🚀 快速开始

### 你需要准备

| 条件 | 说明 |
|---|---|
| **Python** | 3.10 或更新版本 |
| **API 密钥** | 兼容 OpenAI 的密钥，默认使用 DeepSeek（[platform.deepseek.com](https://platform.deepseek.com)） |
| **磁盘空间** | 嵌入模型需要几 GB |

### 第 1 步：获取代码

```bash
git clone https://github.com/Kylekk29/RAG_File_System.git
cd RAG_File_System
```

### 第 2 步：安装

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux：source venv/bin/activate
pip install -r requirements.txt
```

### 第 3 步：配置

在 `app.py` 旁边新建 `.env` 文件：

```ini
API_KEY=your-api-key
BASE_URL=https://api.deepseek.com/v1
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
```

其他设置（分块大小、`RETRIEVAL_K`、`RERANK_ENABLED`、`HYBRID_SEARCH_ENABLED` 等）及默认值都列在 `config.py` 中。

### 第 4 步：运行

```bash
python app.py
```

打开 **http://127.0.0.1:8000**，上传文件，开始提问。

> ⏱️ 首次运行会下载嵌入模型，所以比较慢；之后几秒就能启动。

---

## 🔌 API

网页只是 FastAPI 后端的一个轻量客户端。

| 接口 | 用途 |
|---|---|
| `POST /api/upload` | 上传并建立文件索引 |
| `GET /api/books` | 列出已建立索引的书 |
| `POST /api/chat/stream` | 向一本书提问（流式） |
| `POST /api/chat/multi/stream` | 向多本书提问（流式） |
| `GET /api/sessions` | 列出已保存的聊天 |
| `GET /api/health` | 健康检查 |

应用运行时，可在 `/docs` 查看交互式文档。

## 📁 项目结构

```
app.py                 FastAPI 应用和路由
config.py              从 .env 读取的设置
document_loader.py     各类文件读取器
embedding_manager.py   分块、嵌入、FAISS
query_engine.py        检索、重排序、回答
version_manager.py     书籍版本历史
session_manager.py     聊天会话
```

运行时文件夹（`vector_dbs/`、`uploads/`、`chat_history/`、`logs/`）会在首次运行时自动创建。

## 📄 许可证

MIT，见 [LICENSE](LICENSE)。
