from lib.search_utils import CACHE, load_movies

import numpy as np
import os
from numpy import float64, ndarray
from sentence_transformers import SentenceTransformer


class SemanticSearch:
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.embeddings: ndarray | None = None
        self.documents: list[dict] | None = None
        self.docmap: dict[int, dict] = {}
        self.embeddings_path = os.path.join(CACHE, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> float64:
        if len(text.split()) == 0:
            raise ValueError("Text empty or contains only whitespace")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list[dict]) -> ndarray:
        self.documents = documents
        doc_texts = []
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
            doc_texts.append(f"{doc["title"]}: {doc["description"]}")
        self.embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        self.save()
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> ndarray:
        self.documents = documents
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.load()
        if self.embeddings.any():
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.embeddings_path, "wb") as f:
            np.save(f, self.embeddings)

    def load(self) -> None:
        with open(self.embeddings_path, "rb") as f:
            self.embeddings = np.load(f)

def verify_model() -> None:
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")

def embed_text(text: str) -> float64:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")
