from transformers import pipeline
from sentence_transformers import SentenceTransformer
from typing import List

# =============================
# LLM for summarization / text generation
# =============================
# Using a small, free, local model
text_generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small",  # lightweight, free
    tokenizer="google/flan-t5-small"
)

def get_llm():
    """
    Returns a callable that takes a prompt and returns generated text.
    """
    return lambda prompt: text_generator(prompt, max_length=200)[0]['generated_text']

def summarize_text(llm, text: str) -> str:
    """
    Summarize the text using HuggingFace LLM.
    """
    max_tokens = 500
    text_chunks = text.split()  # simple word split
    truncated_text = " ".join(text_chunks[:max_tokens])
    prompt = f"Summarize the following text in a concise manner with key details:\n\n{truncated_text}"
    return llm(prompt)


# =============================
# Embeddings for RAG / vector DB
# =============================
embed_model = SentenceTransformer('all-MiniLM-L6-v2')  # free and lightweight

def generate_embeddings(texts: List[str]):
    """
    Generates embeddings for a list of texts using SentenceTransformer.
    Returns a list of vectors.
    """
    vectors = embed_model.encode(texts)
    return vectors.tolist()
