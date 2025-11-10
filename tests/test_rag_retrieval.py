from src.qdrant_store import LocalVectorStore
from unittest.mock import MagicMock
import pytest

def test_local_vector_store_query(monkeypatch):
    # Create an instance without calling __init__
    store = LocalVectorStore.__new__(LocalVectorStore)
    # Mock the query method
    store.query = MagicMock(return_value=[{"id": "1", "score": 0.1, "payload": {"text": "sample"}}])

    # Import the function to test
    from src.nodes.pdf_rag_node import query_rag

    # Mock LLM callable
    class DummyLLM:
        def __call__(self, prompt):
            return "Answer using context: sample"

    # Run the RAG query function
    resp = query_rag("What is sample?", store, llm=DummyLLM(), top_k=1)

    # Assertions
    assert "answer" in resp
    assert resp["answer"] == "Answer using context: sample"
    assert resp["hits"][0]["payload"]["text"] == "sample"

