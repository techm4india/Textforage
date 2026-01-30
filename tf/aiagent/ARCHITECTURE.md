# TextForge RAG Architecture Design
**Senior AI Systems Architect Review**  
**Date:** 2026-01-28  
**Status:** Architectural Specification (Implementation Pending)

---

## 1. System Overview

### 1.1 Current Architecture
```
React Frontend (port 3000)
    ↓ HTTP POST
Node.js Backend (port 5000) - Express API Gateway
    ↓ HTTP POST
Python LLM Backend (port 8000) - Flask + Phi-2
```

### 1.2 Enhanced RAG Architecture
```
React Frontend (port 3000)
    ↓ HTTP
Node.js Backend (port 5000) - API Gateway + Vector DB Client
    ↓ HTTP (for LLM) | Direct (for embeddings)
Python Backend (port 8000) - LLM + Embeddings + Document Processing
    ↑
Pinecone Vector Database (cloud)
```

---

## 2. Component Responsibilities

### 2.1 Node.js Backend (Port 5000)
**Role:** HTTP API Gateway, Orchestration, Vector DB Client

**Responsibilities:**
- ✅ HTTP request/response handling (Express)
- ✅ CORS and middleware
- ✅ **File upload handling** (multer/formidable)
- ✅ **Pinecone client** (vector database operations)
- ✅ **Mode routing** (normal vs RAG chat)
- ✅ **Session/document management** (tracking active documents per session)
- ✅ **Orchestration** (coordinate between Python services and Pinecone)

**New Dependencies:**
- `@pinecone-database/pinecone` - Vector DB client
- `multer` or `formidable` - File upload handling
- `dotenv` - Environment variable loading

**Does NOT handle:**
- ❌ Text extraction (Python has better libraries)
- ❌ Embedding generation (Python has sentence-transformers)
- ❌ LLM inference (Python already has Phi-2)

---

### 2.2 Python Backend (Port 8000)
**Role:** AI/ML Processing Engine

**Responsibilities:**
- ✅ **LLM generation** (Phi-2) - existing
- ✅ **Text extraction** (PDF, DOCX, TXT, etc.) - new
- ✅ **Embedding generation** (sentence-transformers or OpenAI API) - new
- ✅ **Chunking logic** (text splitting with overlap) - new
- ✅ **RAG prompt construction** (inject retrieved context) - new

**New Dependencies:**
- `sentence-transformers` or `openai` - Embedding models
- `PyPDF2` or `pdfplumber` - PDF extraction
- `python-docx` - DOCX extraction
- `langchain.text_splitter` or custom chunking - Text splitting

**Endpoints:**
- `/health` - existing
- `/generate` - **enhanced** (accepts optional `context` parameter)
- `/extract-text` - **new** (document → plain text)
- `/generate-embeddings` - **new** (text → embeddings array)
- `/chunk-text` - **new** (text → chunks with metadata)

---

### 2.3 React Frontend (Port 3000)
**Role:** User Interface

**Responsibilities:**
- ✅ Chat UI - existing
- ✅ **Mode toggle** (Normal / Document-Aware) - new
- ✅ **Document upload UI** - new
- ✅ **Document list/management** - new
- ✅ **Session state** (track active mode, active documents) - new

---

## 3. Request/Response Flows

### 3.1 Flow A: Normal AI Chat (No Documents)

**Request Path:**
```
User types message → React
    ↓ POST /api/chat { message, mode: "normal" }
Node.js
    ↓ POST /generate { message }
Python LLM
    ↓ { reply }
Node.js
    ↓ { reply }
React → Display
```

**Request Body (React → Node):**
```json
{
  "message": "What is machine learning?",
  "mode": "normal"
}
```

**Request Body (Node → Python):**
```json
{
  "message": "What is machine learning?"
}
```

**Response (Python → Node):**
```json
{
  "reply": "Machine learning is..."
}
```

**Response (Node → React):**
```json
{
  "reply": "Machine learning is..."
}
```

**Key Points:**
- No vector DB interaction
- No embedding generation
- Direct LLM call (existing behavior preserved)

---

### 3.2 Flow B: Document-Aware RAG Chat

**Request Path:**
```
User types message → React
    ↓ POST /api/chat { message, mode: "rag", documentIds: [...] }
Node.js
    ├─→ POST /generate-embeddings { text: user_message }
    │   Python → returns embedding vector
    ├─→ Pinecone.query(embedding, filter: { documentIds })
    │   Pinecone → returns top-k chunks with metadata
    └─→ POST /generate { message, context: [chunk1, chunk2, ...] }
        Python LLM → generates reply with context
    ↓ { reply, sources: [...] }
Node.js
    ↓ { reply, sources: [...] }
React → Display (with source citations)
```

**Request Body (React → Node):**
```json
{
  "message": "What does the document say about pricing?",
  "mode": "rag",
  "documentIds": ["doc_123", "doc_456"]
}
```

**Internal Steps (Node.js):**
1. **Query Embedding:**
   ```javascript
   POST http://localhost:8000/generate-embeddings
   { "text": "What does the document say about pricing?" }
   → Returns: { "embedding": [0.123, -0.456, ...] }
   ```

2. **Vector Search:**
   ```javascript
   pinecone.query({
     vector: embedding,
     topK: 5,
     filter: { documentId: { $in: ["doc_123", "doc_456"] } }
   })
   → Returns: { matches: [{ id, score, metadata: { text, documentId, chunkIndex } }] }
   ```

3. **LLM Generation with Context:**
   ```javascript
   POST http://localhost:8000/generate
   {
     "message": "What does the document say about pricing?",
     "context": [
       { "text": "Our pricing model...", "source": "doc_123", "chunk": 5 },
       { "text": "Enterprise plans start at...", "source": "doc_123", "chunk": 12 }
     ]
   }
   ```

**Response (Python → Node):**
```json
{
  "reply": "According to the document, pricing is structured as follows...",
  "sources": [
    { "documentId": "doc_123", "chunkIndex": 5, "score": 0.89 },
    { "documentId": "doc_123", "chunkIndex": 12, "score": 0.85 }
  ]
}
```

**Response (Node → React):**
```json
{
  "reply": "According to the document, pricing is structured as follows...",
  "sources": [
    { "documentId": "doc_123", "chunkIndex": 5, "score": 0.89 },
    { "documentId": "doc_123", "chunkIndex": 12, "score": 0.85 }
  ],
  "mode": "rag"
}
```

**Key Points:**
- Query is embedded (Python)
- Vector search happens in Node (Pinecone client)
- Retrieved chunks are sent to Python LLM as context
- Python constructs RAG prompt with context
- Sources are returned for citation

---

### 3.3 Flow C: Document Upload & Ingestion

**Request Path:**
```
User uploads file → React (FormData)
    ↓ POST /api/documents/upload (multipart/form-data)
Node.js (multer)
    ├─→ Save file temporarily
    └─→ POST /extract-text { file_path }
        Python
        ├─→ Extract text (PDF/DOCX/TXT)
        └─→ POST /chunk-text { text, chunkSize, overlap }
            Python → returns chunks array
    ├─→ POST /generate-embeddings { texts: [chunk1, chunk2, ...] }
    │   Python → returns embeddings array
    └─→ Pinecone.upsert(vectors with metadata)
        Pinecone → stores vectors
    ↓ { documentId, status: "ingested", chunks: 45 }
Node.js
    ↓ { documentId, status: "ingested", chunks: 45 }
React → Display success, add to document list
```

**Request (React → Node):**
```javascript
FormData:
  - file: <File object>
  - name: "product_spec.pdf"
```

**Internal Steps (Node.js):**
1. **Save uploaded file:**
   ```javascript
   // multer saves to ./uploads/temp_<uuid>.pdf
   ```

2. **Extract Text:**
   ```javascript
   POST http://localhost:8000/extract-text
   { "filePath": "./uploads/temp_abc123.pdf", "mimeType": "application/pdf" }
   → Returns: { "text": "Full extracted text...", "metadata": {...} }
   ```

3. **Chunk Text:**
   ```javascript
   POST http://localhost:8000/chunk-text
   {
     "text": "Full extracted text...",
     "chunkSize": 512,
     "overlap": 50
   }
   → Returns: {
       "chunks": [
         { "text": "Chunk 1...", "index": 0, "start": 0, "end": 512 },
         { "text": "Chunk 2...", "index": 1, "start": 462, "end": 974 }
       ]
     }
   ```

4. **Generate Embeddings (batch):**
   ```javascript
   POST http://localhost:8000/generate-embeddings
   { "texts": ["Chunk 1...", "Chunk 2...", ...] }
   → Returns: {
       "embeddings": [
         [0.123, -0.456, ...],
         [0.789, 0.234, ...]
       ]
     }
   ```

5. **Upsert to Pinecone:**
   ```javascript
   pinecone.upsert({
     vectors: [
       {
         id: "doc_123_chunk_0",
         values: [0.123, -0.456, ...],
         metadata: {
           documentId: "doc_123",
           chunkIndex: 0,
           text: "Chunk 1...",
           fileName: "product_spec.pdf",
           uploadedAt: "2026-01-28T10:00:00Z"
         }
       },
       // ... more chunks
     ]
   })
   ```

**Response (Node → React):**
```json
{
  "documentId": "doc_123",
  "status": "ingested",
  "chunks": 45,
  "fileName": "product_spec.pdf",
  "uploadedAt": "2026-01-28T10:00:00Z"
}
```

**Key Points:**
- File upload handled in Node (multer)
- Text extraction in Python (better libraries)
- Chunking in Python (can use langchain or custom)
- Embedding generation in Python (batch processing)
- Vector storage in Node (Pinecone client)
- Document ID generated in Node (UUID)

---

## 4. API Endpoints Specification

### 4.1 Node.js Backend Endpoints

#### `POST /api/chat`
**Purpose:** Main chat endpoint (supports both modes)

**Request Body:**
```json
{
  "message": "User's question",
  "mode": "normal" | "rag",
  "documentIds": ["doc_123", "doc_456"]  // Optional, required if mode="rag"
}
```

**Response (Normal Mode):**
```json
{
  "reply": "AI response",
  "mode": "normal"
}
```

**Response (RAG Mode):**
```json
{
  "reply": "AI response with context",
  "mode": "rag",
  "sources": [
    {
      "documentId": "doc_123",
      "chunkIndex": 5,
      "score": 0.89,
      "text": "Retrieved chunk text..."
    }
  ]
}
```

---

#### `POST /api/documents/upload`
**Purpose:** Upload and ingest a document

**Request:** `multipart/form-data`
- `file`: File object
- `name`: Optional document name

**Response:**
```json
{
  "documentId": "doc_123",
  "status": "ingested",
  "chunks": 45,
  "fileName": "product_spec.pdf",
  "uploadedAt": "2026-01-28T10:00:00Z"
}
```

---

#### `GET /api/documents`
**Purpose:** List all ingested documents

**Response:**
```json
{
  "documents": [
    {
      "documentId": "doc_123",
      "fileName": "product_spec.pdf",
      "chunks": 45,
      "uploadedAt": "2026-01-28T10:00:00Z"
    }
  ]
}
```

---

#### `DELETE /api/documents/:documentId`
**Purpose:** Delete a document and its vectors from Pinecone

**Response:**
```json
{
  "status": "deleted",
  "documentId": "doc_123"
}
```

---

#### `GET /health`
**Purpose:** Health check (existing)

---

### 4.2 Python Backend Endpoints

#### `POST /generate`
**Purpose:** LLM generation (enhanced to support context)

**Request Body:**
```json
{
  "message": "User's question",
  "context": [  // Optional, for RAG mode
    {
      "text": "Retrieved chunk text...",
      "source": "doc_123",
      "chunkIndex": 5
    }
  ]
}
```

**Response:**
```json
{
  "reply": "AI response",
  "sources": [  // Only if context provided
    { "documentId": "doc_123", "chunkIndex": 5, "score": 0.89 }
  ]
}
```

---

#### `POST /extract-text`
**Purpose:** Extract text from uploaded document

**Request Body:**
```json
{
  "filePath": "./uploads/temp_abc123.pdf",
  "mimeType": "application/pdf"
}
```

**Response:**
```json
{
  "text": "Full extracted text...",
  "metadata": {
    "pageCount": 10,
    "wordCount": 5000,
    "extractionMethod": "PyPDF2"
  }
}
```

---

#### `POST /chunk-text`
**Purpose:** Split text into chunks with overlap

**Request Body:**
```json
{
  "text": "Full text to chunk...",
  "chunkSize": 512,
  "overlap": 50
}
```

**Response:**
```json
{
  "chunks": [
    {
      "text": "Chunk 1 text...",
      "index": 0,
      "start": 0,
      "end": 512
    },
    {
      "text": "Chunk 2 text...",
      "index": 1,
      "start": 462,
      "end": 974
    }
  ]
}
```

---

#### `POST /generate-embeddings`
**Purpose:** Generate embeddings for text(s)

**Request Body (single text):**
```json
{
  "text": "Text to embed"
}
```

**Request Body (batch):**
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"]
}
```

**Response (single):**
```json
{
  "embedding": [0.123, -0.456, 0.789, ...]
}
```

**Response (batch):**
```json
{
  "embeddings": [
    [0.123, -0.456, ...],
    [0.789, 0.234, ...],
    [0.567, -0.123, ...]
  ]
}
```

---

#### `GET /health`
**Purpose:** Health check (existing)

---

## 5. Data Models

### 5.1 Document Metadata (stored in Pinecone)
```javascript
{
  documentId: "doc_123",           // UUID generated by Node
  chunkIndex: 5,                    // Index within document
  text: "Chunk text content...",    // Actual chunk text
  fileName: "product_spec.pdf",     // Original filename
  uploadedAt: "2026-01-28T10:00:00Z", // ISO timestamp
  mimeType: "application/pdf"       // File type
}
```

### 5.2 Pinecone Vector Structure
```javascript
{
  id: "doc_123_chunk_5",            // Composite ID
  values: [0.123, -0.456, ...],     // Embedding vector (384 or 768 dims)
  metadata: {
    documentId: "doc_123",
    chunkIndex: 5,
    text: "Chunk text...",
    fileName: "product_spec.pdf",
    uploadedAt: "2026-01-28T10:00:00Z"
  }
}
```

### 5.3 Session State (Frontend)
```javascript
{
  mode: "normal" | "rag",
  activeDocuments: ["doc_123", "doc_456"],  // For RAG mode
  chatHistory: [
    { sender: "user", text: "...", mode: "normal" },
    { sender: "ai", text: "...", mode: "normal" }
  ]
}
```

---

## 6. Embedding & Vector DB Strategy

### 6.1 Embedding Model Location: **Python Backend**

**Rationale:**
- Python has excellent embedding libraries (`sentence-transformers`, `openai`)
- Can use local models (e.g., `all-MiniLM-L6-v2`) or API-based (OpenAI)
- Batch processing is straightforward
- Keeps ML/AI logic centralized in Python

**Implementation Options:**
1. **Local Model (Recommended for MVP):**
   ```python
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')  # 384 dimensions
   ```

2. **OpenAI API (Alternative):**
   ```python
   import openai
   # Uses text-embedding-ada-002 (1536 dimensions)
   ```

**Embedding Dimensions:**
- `all-MiniLM-L6-v2`: 384 dimensions
- `text-embedding-ada-002`: 1536 dimensions
- **Decision:** Use 384-dim model for faster processing and lower Pinecone costs

---

### 6.2 Pinecone Client Location: **Node.js Backend**

**Rationale:**
- Pinecone has excellent JavaScript SDK (`@pinecone-database/pinecone`)
- Node.js is the API gateway, natural place for external service clients
- Keeps Python focused on ML/AI processing
- Easier to manage API keys and configuration in Node

**Operations:**
- `upsert()` - Store document chunks
- `query()` - Search similar chunks
- `delete()` - Remove document vectors (by metadata filter)
- `describeIndexStats()` - Get index statistics

**Index Configuration:**
- **Index Name:** `aiagent-index` (from .env)
- **Dimensions:** 384 (if using all-MiniLM-L6-v2)
- **Metric:** `cosine` (for semantic similarity)
- **Metadata Fields:** `documentId`, `chunkIndex`, `text`, `fileName`, `uploadedAt`

---

## 7. Mode Switching Logic

### 7.1 Frontend Mode Selection
- **Toggle/Radio buttons:** "Normal Chat" vs "Document Chat"
- **Document Chat requires:** At least one document selected from list
- **State management:** React `useState` for `mode` and `activeDocuments`

### 7.2 Backend Mode Routing (Node.js)
```javascript
if (mode === "normal") {
  // Direct LLM call, no retrieval
  const response = await axios.post("http://localhost:8000/generate", {
    message: userMessage
  });
} else if (mode === "rag") {
  // RAG pipeline: embed → retrieve → generate
  const embedding = await getQueryEmbedding(userMessage);
  const chunks = await pinecone.query(embedding, documentIds);
  const response = await axios.post("http://localhost:8000/generate", {
    message: userMessage,
    context: chunks
  });
}
```

---

## 8. Error Handling & Edge Cases

### 8.1 Document Upload Failures
- **Invalid file type:** Return 400 with allowed types
- **Extraction failure:** Return 500 with error details
- **Pinecone upsert failure:** Retry logic or queue for later

### 8.2 RAG Mode Without Documents
- **No documents selected:** Return 400, prompt user to select documents
- **No matching chunks:** Return response with "No relevant context found, answering from general knowledge"

### 8.3 Vector Search Failures
- **Pinecone unavailable:** Fallback to normal mode with warning
- **Empty results:** Proceed with empty context (LLM answers without grounding)

### 8.4 Python Backend Failures
- **Embedding service down:** Return 503, suggest retry
- **LLM generation timeout:** Return 504, suggest shorter query

---

## 9. Performance Considerations

### 9.1 Embedding Generation
- **Batch processing:** Process all chunks of a document in one API call
- **Caching:** Cache embeddings for identical text (optional optimization)

### 9.2 Vector Search
- **Top-K:** Retrieve top 5 chunks (configurable)
- **Filtering:** Use Pinecone metadata filters to scope search to selected documents

### 9.3 LLM Context Window
- **Context limit:** Phi-2 has ~2048 token context window
- **Chunk selection:** Ensure retrieved chunks + user query fit within limit
- **Truncation:** If needed, prioritize highest-scoring chunks

---

## 10. Security Considerations

### 10.1 File Upload
- **File type validation:** Whitelist allowed MIME types (PDF, DOCX, TXT)
- **File size limits:** Max 10MB per file (configurable)
- **Malware scanning:** Consider virus scanning for production

### 10.2 API Keys
- **Environment variables:** Store Pinecone and OpenAI keys in `.env` (never commit)
- **Key rotation:** Support for key rotation without code changes

### 10.3 Document Access
- **User isolation:** In multi-user scenario, filter by userId in Pinecone metadata
- **Document ownership:** Track which user uploaded which document

---

## 11. Implementation Phases

### Phase 1: Foundation
1. Add `.env` loading to Node.js backend
2. Set up Pinecone client in Node.js
3. Add embedding endpoint to Python backend
4. Enhance `/generate` endpoint to accept `context` parameter

### Phase 2: Document Ingestion
1. Add file upload endpoint in Node.js (multer)
2. Add text extraction endpoint in Python
3. Add chunking endpoint in Python
4. Implement full ingestion pipeline (upload → extract → chunk → embed → upsert)

### Phase 3: RAG Query Flow
1. Implement query embedding in Node.js
2. Implement vector search in Node.js
3. Wire RAG flow: embed → retrieve → generate
4. Add source citations to responses

### Phase 4: Frontend Enhancement
1. Add mode toggle UI
2. Add document upload UI
3. Add document list/management UI
4. Display source citations in chat

### Phase 5: Polish & Optimization
1. Error handling improvements
2. Performance optimization (batch processing, caching)
3. User experience enhancements
4. Testing and validation

---

## 12. Dependencies Summary

### Node.js Backend
```json
{
  "dependencies": {
    "express": "^4.18.2",
    "axios": "^1.6.0",
    "cors": "^2.8.5",
    "@pinecone-database/pinecone": "^1.0.0",
    "multer": "^1.4.5-lts.1",
    "dotenv": "^16.3.1",
    "uuid": "^9.0.0"
  }
}
```

### Python Backend
```txt
flask
transformers
torch
sentence-transformers  # For embeddings
PyPDF2  # or pdfplumber
python-docx
langchain  # Optional, for chunking utilities
```

---

## 13. Configuration

### Node.js `.env`
```env
PORT=5000
PYTHON_BACKEND_URL=http://localhost:8000
PINECONE_API_KEY=pcsk_...
PINECONE_INDEX=aiagent-index
PINECONE_ENVIRONMENT=us-east-1  # or your region
```

### Python `.env` (optional)
```env
HF_KEY=hf_...  # If using Hugging Face models
OPENAI_API_KEY=sk-...  # If using OpenAI embeddings
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

---

## 14. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    React Frontend (3000)                    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Mode Toggle  │  │ Document List │  │  Chat UI     │    │
│  │ Normal / RAG │  │  + Upload     │  │  + History   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│            Node.js Backend - API Gateway (5000)              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Routes:                                              │  │
│  │  • POST /api/chat (mode routing)                     │  │
│  │  • POST /api/documents/upload                        │  │
│  │  • GET  /api/documents                               │  │
│  │  • DELETE /api/documents/:id                         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Services:                                            │  │
│  │  • Pinecone Client (vector DB)                        │  │
│  │  • File Upload Handler (multer)                       │  │
│  │  • Orchestration Logic                                │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Python Backend - ML Engine (8000)                   │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Endpoints:                                           │  │
│  │  • POST /generate (LLM + RAG prompt)                 │  │
│  │  • POST /extract-text                                │  │
│  │  • POST /chunk-text                                  │  │
│  │  • POST /generate-embeddings                         │  │
│  └──────────────────────────────────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Models:                                              │  │
│  │  • Phi-2 (LLM)                                        │  │
│  │  • SentenceTransformer (Embeddings)                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Direct API Calls
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Pinecone Vector Database (Cloud)                │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Index: aiagent-index                                │  │
│  │  • Document chunks as vectors                         │  │
│  │  • Metadata: documentId, chunkIndex, text, etc.       │  │
│  │  • Semantic search via cosine similarity             │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 15. Key Design Decisions

1. **Embeddings in Python:** Better ML ecosystem, easier model management
2. **Pinecone in Node:** Natural fit for API gateway, clean JS SDK
3. **Text Extraction in Python:** Superior document parsing libraries
4. **Mode Routing in Node:** Centralized orchestration logic
5. **Context Injection in Python:** LLM already there, prompt engineering in one place
6. **Preserve Node → Python Pattern:** Maintains existing architecture, minimal disruption

---

**End of Architecture Specification**
