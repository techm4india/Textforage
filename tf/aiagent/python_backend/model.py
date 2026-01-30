"""
TextForge Python backend: LLM (Phi-2) + Embeddings (sentence-transformers) for RAG.

- /generate: LLM generation with optional retrieved context (reduces hallucination when context provided).
- /generate-embeddings: Embedding generation for document chunks and query (single or batch).
- Embeddings are 384-dimensional (all-MiniLM-L6-v2); Pinecone index must use dimension=384.
"""

from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

# ---------------------------------------------------------------------------
# LLM (Phi-2) – used for generation; prompt handling supports RAG context
# ---------------------------------------------------------------------------
MODEL_NAME = "microsoft/phi-2"
print(f"Loading LLM '{MODEL_NAME}' on CPU... This may take some time.")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32  # CPU-friendly, no device_map, no offload
)
model.eval()

# ---------------------------------------------------------------------------
# Embedding model – used for RAG: chunk and query embeddings (semantic retrieval)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIMENSION = 384
_embedding_model = None


def get_embedding_model():
    """Lazy-load embedding model so LLM can start first."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{EMBEDDING_MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"Embedding model ready (dimension={EMBEDDING_DIMENSION}).")
    return _embedding_model


def compute_embeddings(texts):
    """Compute embeddings for a list of texts. Returns list of lists (each inner list is 384 floats)."""
    emb_model = get_embedding_model()
    embeddings = emb_model.encode(texts, convert_to_numpy=True)
    if embeddings.ndim == 1:
        embeddings = [embeddings.tolist()]
    else:
        embeddings = [e.tolist() for e in embeddings]
    return embeddings

def build_prompt(user_message: str, context: str = None) -> str:
    """
    Build prompt template for Phi-2 model.
    
    Supports two modes:
    1. RAG mode: When context is provided, instructs model to answer ONLY from context
    2. Normal mode: Standard assistant behavior when no context
    
    Args:
        user_message: The user's question or message
        context: Optional retrieved context from documents (for RAG mode)
    
    Returns:
        Formatted prompt string
    """
    
    # System instruction - always present
    system_instruction = "You are a helpful AI assistant. Answer clearly and correctly."
    
    if context and context.strip():
        # RAG MODE: Context is provided
        # Instruct model to answer ONLY from the provided context
        system_instruction = (
            "You are a helpful AI assistant. "
            "Answer the user's question based ONLY on the provided context from documents. "
            "If the context does not contain enough information to answer the question, "
            "say so clearly. Do not use information outside the provided context."
        )
        
        # Build RAG prompt structure:
        # 1. System instruction
        # 2. Retrieved context
        # 3. User question
        prompt = (
            f"{system_instruction}\n\n"
            f"{context}\n\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
    else:
        # NORMAL MODE: No context provided
        # Standard assistant behavior
        prompt = (
            f"{system_instruction}\n\n"
            f"User: {user_message}\n"
            f"Assistant:"
        )
    
    return prompt


def generate_reply(user_message: str,
                   context: str = None,
                   max_new_tokens: int = 256,
                   temperature: float = 0.7) -> str:
    """
    Generate reply using Phi-2 model.
    
    Supports both normal chat and RAG modes:
    - Normal mode: Standard assistant response
    - RAG mode: Answer based on provided document context
    
    Args:
        user_message: The user's question or message
        context: Optional retrieved context from documents (for RAG mode)
        max_new_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    
    Returns:
        Generated reply string
    """
    
    # Build prompt based on whether context is provided
    prompt = build_prompt(user_message, context)
    
    # Log mode for debugging
    mode = "RAG" if context and context.strip() else "Normal"
    print(f"🤖 Generating reply in {mode} mode...")
    
    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
    
    # Check input length to avoid overflow
    input_length = inputs["input_ids"].shape[1]
    max_context_length = 2048  # Phi-2 context window
    
    if input_length > max_context_length:
        print(f"⚠️ Warning: Input length ({input_length}) exceeds model context window ({max_context_length})")
        # Truncate if necessary (keep the end, which has the user question)
        max_input_length = max_context_length - max_new_tokens - 50  # Leave room for generation
        if max_input_length > 0:
            inputs["input_ids"] = inputs["input_ids"][:, -max_input_length:]
            inputs["attention_mask"] = inputs["attention_mask"][:, -max_input_length:]
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode full generated text
    full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the assistant's reply (text after "Assistant:")
    if "Assistant:" in full_text:
        reply = full_text.split("Assistant:", 1)[1].strip()
    else:
        # Fallback: return the generated text if "Assistant:" marker not found
        reply = full_text.strip()
    
    return reply


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": MODEL_NAME,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_dimension": EMBEDDING_DIMENSION,
    })


# ---------------------------------------------------------------------------
# RAG: Embedding generation for document chunks (ingestion) and queries (retrieval)
# ---------------------------------------------------------------------------
@app.route("/generate-embeddings", methods=["POST"])
def generate_embeddings():
    """
    Generate embeddings for semantic retrieval (RAG pipeline).

    Request Body (single text):
        { "text": "string" }
    Response:
        { "embedding": [float, ...] }  # length 384

    Request Body (batch):
        { "texts": ["string", ...] }
    Response:
        { "embeddings": [[float, ...], ...] }

    Used by Node backend for:
    - Document ingestion: batch embed chunk texts before storing in Pinecone.
    - Query-time RAG: embed user query before Pinecone similarity search.
    """
    data = request.get_json() or {}

    # Single text
    if "text" in data:
        text = data["text"]
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        if not text.strip():
            return jsonify({"error": "Field 'text' must be a non-empty string"}), 400
        try:
            embeddings = compute_embeddings([text.strip()])
            return jsonify({"embedding": embeddings[0]})
        except Exception as e:
            print(f"❌ Embedding error: {e}")
            return jsonify({"error": "Embedding failed", "details": str(e)}), 500

    # Batch texts
    if "texts" in data:
        texts = data["texts"]
        if not isinstance(texts, list):
            return jsonify({"error": "Field 'texts' must be an array of strings"}), 400
        texts = [t if isinstance(t, str) else str(t) for t in texts if t is not None]
        texts = [t.strip() for t in texts if t.strip()]
        if not texts:
            return jsonify({"error": "Field 'texts' must contain at least one non-empty string"}), 400
        try:
            embeddings = compute_embeddings(texts)
            return jsonify({"embeddings": embeddings})
        except Exception as e:
            print(f"❌ Batch embedding error: {e}")
            return jsonify({"error": "Embedding failed", "details": str(e)}), 500

    return jsonify({"error": "Provide 'text' or 'texts' in request body"}), 400


@app.route("/generate", methods=["POST"])
def generate():
    """
    Generate LLM response with optional RAG context.
    
    Request Body:
        - message: str (required) - User's question or message
        - context: str (optional) - Retrieved document context for RAG mode
    
    Response:
        - reply: str - Generated response
        - sources: list (optional) - Source information if RAG mode was used
    
    Backward Compatibility:
        - Requests without 'context' field work as before (normal chat mode)
        - Requests with empty/null 'context' also use normal mode
    """
    data = request.get_json() or {}
    
    # Extract user message (required)
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Message is required"}), 400
    
    # Extract context (optional - for RAG mode)
    context = data.get("context")
    if context is not None:
        # Normalize context: convert to string, strip whitespace
        context = str(context).strip() if context else None
        # If context is empty string after stripping, treat as None
        if not context:
            context = None
    
    # Determine mode for logging
    mode = "RAG" if context else "Normal"
    print(f"\n💬 /generate request - Mode: {mode}")
    print(f"   Message: {user_message[:100]}...")
    if context:
        print(f"   Context length: {len(context)} chars")
    
    try:
        # Generate reply (with or without context)
        reply = generate_reply(
            user_message=user_message,
            context=context,
            max_new_tokens=256,  # Can be made configurable
            temperature=0.7      # Can be made configurable
        )
        
        # Build response
        response_data = {
            "reply": reply,
        }
        
        # Include sources if provided in request (for RAG mode)
        # The Node backend may pass sources for citation tracking
        if "sources" in data:
            response_data["sources"] = data["sources"]
        
        print(f"✅ Generated reply ({len(reply)} chars)")
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": "Generation failed",
            "details": str(e)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
