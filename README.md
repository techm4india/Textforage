# Textforage

React frontend + Node.js API gateway + Python LLM/embeddings + Pinecone for normal chat and RAG (document Q&A).

## Architecture

- **Frontend** (port 3000): React app — Normal Chat and RAG Chat.
- **Node backend** (port 5000): Express API — `/api/chat`, `/api/chat-rag`, `/api/upload-doc`; uses `.env` and proxies to Python; talks to Pinecone.
- **Python backend** (port 8000): Flask — Phi-2 LLM + sentence-transformers embeddings; `/generate`, `/generate-embeddings`.

## Prerequisites

- Node.js 18+
- Python 3.8+ (with venv recommended)
- Pinecone account; index named in `.env` (e.g. `ayush`) with **dimension 384** (all-MiniLM-L6-v2)

## Environment (.env)

Create `.env` in this directory (same folder as this README). **Do not commit `.env`** (it’s in `.gitignore`).

```env
GEMINI_API_KEY=your_gemini_key
PINECONE_API_KEY=your_pinecone_key
PINECONE_INDEX=ayush

PORT=5000
PYTHON_BACKEND_URL=http://localhost:8000
```

- **Pinecone index:** Create an index with dimension **384** (cosine or dot-product). Name must match `PINECONE_INDEX`.

## Run the full stack

Use three terminals.

### 1. Python backend (port 8000)

```bash
cd tf/aiagent/python_backend
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
python model.py
```

### 2. Node backend (port 5000)

```bash
cd tf/backend
npm install
npm start
```

### 3. Frontend (port 3000)

```bash
cd tf/frontend
npm install
npm start
```

Then open **http://localhost:3000**. Use “Normal Chatbot” for chat, “RAG Chatbot” to upload a .txt and ask questions over it.

## Security

- **Never commit or share `.env`.** It contains API keys.
- Rotate keys if they were ever exposed.
- Keep dependencies updated (`npm audit`, `pip list`).
