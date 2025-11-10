import streamlit as st
from langgraph_pipeline import pipeline_handle
from qdrant_store import LocalVectorStore  # <- updated
from config import VECTOR_COLLECTION

st.set_page_config(page_title="LangGraph Weather+RAG Demo")
st.title("LangGraph + LangChain: Weather and PDF RAG demo")

# Initialize local vector store (FAISS-based)
vector_store = LocalVectorStore(collection_path=f"{VECTOR_COLLECTION}.pkl", dim=1536)

uploaded = st.file_uploader("Upload a PDF (for RAG)", type=["pdf"])
question = st.text_input("Ask a question (weather or about the PDF)")

if st.button("Ask"):
    if not question:
        st.warning("Enter a question.")
    else:
        pdf_path = None
        if uploaded:
            # Save to temp file
            with open("temp_uploaded.pdf", "wb") as f:
                f.write(uploaded.getbuffer())
            pdf_path = "temp_uploaded.pdf"
        try:
            # Pass the local vector store to your pipeline
            resp = pipeline_handle(question, pdf_path=pdf_path)
            st.json(resp)
        except Exception as e:
            st.error(str(e))
