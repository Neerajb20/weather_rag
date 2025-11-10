import os
from dotenv import load_dotenv


load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENWEATHERMAP_API_KEY = os.getenv("OPENWEATHERMAP_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
VECTOR_COLLECTION = os.getenv("VECTOR_COLLECTION", "langgraph_rag")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
