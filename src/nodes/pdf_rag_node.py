from pdf_utils import extract_text_from_pdf, chunk_text
from llm_processor import summarize_text, generate_embeddings, get_llm
from qdrant_store import LocalVectorStore  # <- updated
import uuid

def ingest_pdf_to_qdrant(pdf_path: str, vector_store: LocalVectorStore, llm=None):
    """
    Extract text from PDF, chunk it, generate embeddings, and store in LocalVectorStore.
    """
    text = extract_text_from_pdf(pdf_path)
    chunks = chunk_text(text, chunk_size=600, overlap=100)
    embeddings = generate_embeddings(chunks)
    import numpy as np
    embeddings_np = np.array(embeddings)
    print("Embeddings shape:", embeddings_np.shape)
    ids = [str(uuid.uuid4()) for _ in chunks]
    # print(text)
    # Include the chunk text in metadata for retrieval
    metadatas = [{"source": pdf_path, "chunk_index": i, "text": chunk} for i, chunk in enumerate(chunks)]
    # print(metadatas)
    # Upsert into LocalVectorStore
    vector_store.upsert(embeddings=embeddings, metadatas=metadatas, ids=ids)
    print(vector_store.ids)

    # Optionally summarize original document
    if not llm:
        llm = get_llm()
    summary = summarize_text(llm, "\n\n".join(chunks))
    print(f"PDF ingested: {pdf_path}, Chunks: {len(chunks)}")
    print(f"Summary: {summary}")
    return {"summary": summary, "chunks": len(chunks)}


def query_rag(question: str, vector_store: LocalVectorStore, llm=None, top_k=3):
    """
    Generate embedding for question, retrieve top_k chunks from LocalVectorStore, and answer via LLM.
    """
    print(f"Querying RAG for question: {question}")
    if not llm:
        llm = get_llm()

    # Generate embedding for the question
    q_emb = generate_embeddings([question])[0]
    print(q_emb)
    # Query the vector store
    hits = vector_store.query(q_emb, top_k=top_k)

    # Extract chunk text for context
    context_texts = [h["payload"].get("text", "") for h in hits if h["payload"]]

    # Build prompt combining question + retrieved contexts
    context_text = "\n\n".join(context_texts)
    prompt = f"Use the context below to answer the question. Be concise.\n\nContext:\n{context_text}\n\nQuestion: {question}"

    # Get answer from LLM
    answer = llm.call_as_llm(prompt) if hasattr(llm, "call_as_llm") else llm(prompt)
    return {"answer": answer, "hits": hits}
