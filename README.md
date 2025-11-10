# LangGraph + LangChain Weather + PDF RAG Demo

## Objective
Demo an agentic pipeline using LangGraph to route between fetching real-time weather (OpenWeatherMap) and answering questions from PDF documents using RAG (faiss + embeddings). Use LangChain LLMs and LangSmith for evaluation.

## Setup
1. Clone repo.
2. Create virtualenv & install:
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirement.txt
3. Copy `.env` and set your API keys.
4. Run Streamlit demo:
   `streamlit run src/streamlit_app.py`
5. Run tests:
   `pytest -q`


## LangSmith
1. Set LANGSMITH_API_KEY in .env
2. See scripts/langsmith_demo.py for upload & evaluation examples



