import os
import pickle
import numpy as np
from typing import List, Dict

class LocalVectorStore:
    def __init__(self, collection_path: str = "faiss_index.pkl", dim: int = 384):
        import faiss
        self.collection_path = collection_path
        self.dim = dim
        self.vectors = np.empty((0, dim), dtype="float32")
        self.metadatas = []
        self.ids = []
        self.index = faiss.IndexFlatL2(dim)

        if os.path.exists(self.collection_path):
            self._load()

    def convert_384_to_1536(self, vec_384: np.ndarray):
        """
        Convert a single 384-dim vector into 1536-dim by repeating it 4 times.
        """
        if vec_384.shape[0] != 384:
            raise ValueError("Input must be 384-dimensional")
        return np.tile(vec_384, 4)

    def _save(self):
        """
        Save FAISS index, vectors, ids, and metadata to disk.
        """
        with open(self.collection_path, "wb") as f:
            pickle.dump({
                "index": self.index,
                "ids": self.ids,
                "metadatas": self.metadatas,
                "vectors": self.vectors
            }, f)

    def _load(self):
        with open(self.collection_path, "rb") as f:
            data = pickle.load(f)
            self.index = data["index"]
            self.ids = data["ids"]
            self.metadatas = data["metadatas"]
            self.vectors = data.get("vectors", np.empty((0, self.dim), dtype="float32"))

    def upsert(self, embeddings: List[List[float]], metadatas: List[Dict], ids: List[str]):
        import faiss
        import numpy as np

        embeddings_np = np.array(embeddings, dtype="float32")
        # Convert 384-dim → 1536-dim
        embeddings_np = np.tile(embeddings_np, (1, 4))

        if embeddings_np.shape[1] != self.dim:
            raise ValueError(f"Embeddings must be of shape (n, {self.dim}), but got {embeddings_np.shape[1]}")

        if self.vectors.size == 0:
            self.vectors = embeddings_np
        else:
            self.vectors = np.vstack([self.vectors, embeddings_np])

        self.metadatas.extend(metadatas)
        self.ids.extend(ids)

        self.index = faiss.IndexFlatL2(self.dim)
        self.index.add(self.vectors)

        self._save()

    def query(self, embedding: List[float], top_k: int = 5):
        import faiss
        import numpy as np

        embedding_np = np.array([embedding], dtype="float32")
        embedding_np = np.tile(embedding_np, (1, 4))
        if embedding_np.shape[1] != self.dim:
            raise ValueError(f"Query embedding must have dimension {self.dim}")

        distances, indices = self.index.search(embedding_np, top_k)
        hits = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < len(self.ids):
                hits.append({
                    "id": self.ids[idx],
                    "score": float(dist),
                    "payload": self.metadatas[idx]
                })
        return hits
