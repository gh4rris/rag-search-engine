from lib.search_utils import MODEL_NAME, CACHE, MAX_CHUNK_SIZE, SENTENCE_OVERLAP, load_movies

import os
import re
import numpy as np
from numpy import ndarray
from sentence_transformers import SentenceTransformer



class SemanticSearch:
    def __init__(self, model_name: str=MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.embeddings: ndarray[ndarray] | None = None # array of embedded documents
        self.documents: list[dict] | None = None # list of documents
        self.docmap: dict[int, dict] = {} # mapping document IDs to document objects
        self.embeddings_path = os.path.join(CACHE, "movie_embeddings.npy")

    def generate_embedding(self, text: str) -> ndarray:
        if not text.strip():
            raise ValueError("Text empty or contains only whitespace")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list[dict]) -> ndarray[ndarray]:
        self.documents = documents
        doc_texts = []
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
            doc_texts.append(f"{doc["title"]}: {doc["description"]}")
        self.embeddings = self.model.encode(doc_texts, show_progress_bar=True)
        self.save()
        return self.embeddings
    
    def load_or_create_embeddings(self, documents: list[dict]) -> ndarray[ndarray]:
        self.documents = documents
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
        if os.path.exists(self.embeddings_path):
            self.load()
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        return self.build_embeddings(documents)
    
    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_embedding = self.generate_embedding(query)
        score_docs = []
        for doc_embedding, doc in zip(self.embeddings, self.documents):
            score = cosine_similarity(query_embedding, doc_embedding)
            score_docs.append((score, doc))
        score_docs.sort(key=lambda x: x[0], reverse=True)
        results = []
        for score, doc in score_docs[:limit]:
            results.append({"score": score,
                            "title": doc["title"],
                            "description": doc["description"]})
        return results
        

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

def embed_text(text: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings() -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(movies)
    print(f"Number of docs: {len(movies)}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query: str) -> None:
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")

def cosine_similarity(vec1: ndarray, vec2: ndarray) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0
    return dot_product / (norm1 * norm2)

def search_command(query: str, limit: int) -> None:
    semantic_search = SemanticSearch()
    movies = load_movies()
    semantic_search.load_or_create_embeddings(movies)
    results = semantic_search.search(query, limit)
    for i, result in enumerate(results, 1):
        print(f"{i}. {result["title"]} ({result["score"]:.2f})")
        print(f"{result["description"]}\n")

def chunk_text(text: str, chunk_size: int, overlap: int) -> None:
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk_words = words[i:i + chunk_size]
        if chunks and len(chunk_words) <= overlap:
            break
        chunks.append(" ".join(chunk_words))
        i += chunk_size - overlap
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")

def semantic_chunk(text: str, chunk_size: int=MAX_CHUNK_SIZE, overlap: int=SENTENCE_OVERLAP) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    i = 0
    while i < len(sentences):
        chunk_sentence = sentences[i: i + chunk_size]
        if chunks and len(chunk_sentence) <= overlap:
            break
        chunks.append(" ".join(chunk_sentence))
        i += chunk_size - overlap
    return chunks

def semantic_chunk_text(text: str, chunk_size: int, overlap: int) -> None:
    chunks = semantic_chunk(text, chunk_size, overlap)
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")