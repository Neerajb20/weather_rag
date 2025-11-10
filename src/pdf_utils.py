from PyPDF2 import PdfReader
from typing import List

def extract_text_from_pdf(path: str) -> str:
    reader = PdfReader(path)
    out = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            out.append(text)
    return "\n\n".join(out)

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    tokens = text.split()
    chunks = []
    i = 0
    while i < len(tokens):
        chunk = tokens[i:i+chunk_size]
        chunks.append(" ".join(chunk))
        i += chunk_size - overlap
    return chunks
