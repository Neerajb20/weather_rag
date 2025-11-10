from nodes.decision_node import decision_node
from nodes.weather_node import fetch_weather_by_city, normalize_weather
from nodes.pdf_rag_node import query_rag, ingest_pdf_to_qdrant
from qdrant_store import LocalVectorStore  # <- use LocalVectorStore now
from llm_processor import get_llm
from config import VECTOR_COLLECTION

# Use the local FAISS-based vector store 
vector_store = LocalVectorStore(collection_path=f"{VECTOR_COLLECTION}.pkl", dim=1536)
llm = get_llm()

def pipeline_handle(question: str, pdf_path: str = None):
    # 1) Decision
    dec = decision_node(question)
    route = dec["route"]

    if route == "weather":
        # extract city — naive approach: last word or "in <city>"
        import re
        m = re.search(r"in ([A-Za-z\s]+)$", question, re.IGNORECASE)
        if m:
            city = m.group(1).strip()
        else:
            parts = question.split()
            city = parts[-1].strip("?.")
        weather_raw = fetch_weather_by_city(city)
        normalized = normalize_weather(weather_raw)
        # process with LLM for better phrasing / explanation
        answer = get_llm().call_as_llm(f"Summarize this weather info for a clinician: {normalized}") \
                 if hasattr(get_llm(), "call_as_llm") else get_llm()(f"Summarize: {normalized}")
        return {"type": "weather", "answer": normalized, "llm_answer": answer}

    else:
        if not pdf_path:
            raise ValueError("PDF path required for PDF RAG queries.")
        # Ingest PDF into the local vector store
        ingest_pdf_to_qdrant(pdf_path, vector_store, llm=llm)  # pass vector_store instead of Qdrant
        # Query RAG
        rag_resp = query_rag(question, vector_store, llm=llm, top_k=4)
        return {"type": "pdf_rag", **rag_resp}
