# 📄 Document Query Assistant

A simple local app that lets you search your `.txt` and `.pdf` documents using AI embeddings. It uses [JinaAI's embedding API](https://jina.ai/), LangChain for text processing, and [ChromaDB](https://www.trychroma.com/) as a vector database.

---

## 🚀 Features

- 🧠 Converts text & PDFs into semantic embeddings using JinaAI.
- 🗂️ Stores embeddings in a local ChromaDB vector store.
- 🔍 Allows you to ask natural language questions and get the most relevant chunks from your documents.
- 📦 Supports `.txt` and `.pdf` files out of the box.

---

## 🛠️ Tech Stack

- Python 3.8+
- [Jina AI Embeddings API](https://jina.ai/)
- LangChain (`TextLoader`, `PyPDFLoader`, `CharacterTextSplitter`)
- ChromaDB (local vector store)
- Requests (for API calls)

---

## ⚙️ How It Works

1. **Select a file** (`.txt` or `.pdf`).
2. The app splits it into small text chunks.
3. Each chunk is embedded via JinaAI’s API.
4. Embeddings and metadata are stored in ChromaDB.
5. You ask a question, and it returns the top matching chunks from the document.

---

## 📦 Installation

```bash
git clone https://github.com/yourusername/document-query-assistant.git
cd document-query-assistant

# Install dependencies
pip install -r requirements.txt
