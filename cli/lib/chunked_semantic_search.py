from lib.semantic_search import SemanticSearch, semantic_chunk, cosine_similarity
from lib.search_utils import MODEL_NAME, CACHE, SCORE_PRECISION, DOCUMENT_PREVIEW_LENGTH, load_movies

import os
import json
import numpy as np
from numpy import ndarray


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str=MODEL_NAME) -> None:
        super().__init__(model_name)
        self.chunk_embeddings: ndarray[ndarray] | None = None # array of embedded chunks
        self.chunk_metadata: list[dict] | None = None # list of chunk metadata
        self.chunk_embeddings_path = os.path.join(CACHE, "chunk_embeddings.npy")
        self.metadata_path = os.path.join(CACHE, "chunk_metadata.json")

    def build_chunk_embeddings(self, documents: list[dict]) -> ndarray[ndarray]:
        self.documents = documents
        all_chunks = []
        metadata = []
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
            if not doc["description"].strip():
                continue
            description_chunks = semantic_chunk(doc["description"])
            for chunk in description_chunks:
                all_chunks.append(chunk)
                metadata.append(
                    {
                        "movie_idx": self.documents.index(doc),
                        "chunk_idx": description_chunks.index(chunk),
                        "total_chunks": len(description_chunks)
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = metadata
        self.save_chunks(all_chunks)
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> ndarray[ndarray]:
        self.documents = documents
        for doc in self.documents:
            self.docmap[doc["id"]] = doc
        if os.path.exists(self.chunk_embeddings_path) and os.path.exists(self.metadata_path):
            self.load_chunks()
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)
        
    def save_chunks(self, all_chunks: list[str]) -> None:
        with open(self.chunk_embeddings_path, "wb") as f:
            np.save(f, self.chunk_embeddings)
        with open(self.metadata_path, "w") as f:
            json.dump(
                {
                    "chunks": self.chunk_metadata,
                    "total_chunks": len(all_chunks)
                },
                f,
                indent=2
            )
    
    def load_chunks(self) -> None:
        with open(self.chunk_embeddings_path, "rb") as f:
            self.chunk_embeddings = np.load(f)
        with open(self.metadata_path, "r") as f:
            data = json.load(f)
            self.chunk_metadata = data["chunks"]

    def search_chunks(self, query: str, limit: int) -> list[dict]:
        query_embedding = self.generate_embedding(query)
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(query_embedding, chunk_embedding)
            chunk_scores.append(
                {
                    "chunk_idx": i,
                    "movie_idx": self.chunk_metadata[i]["movie_idx"],
                    "score": score
                }
            )
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx, score = chunk_score["movie_idx"], chunk_score["score"]
            if movie_idx not in movie_scores or score > movie_scores.get(movie_idx, 0):
                movie_scores[movie_idx] = score
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for movie_idx, score in sorted_movies[:limit]:
            document = self.documents[movie_idx]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"][:DOCUMENT_PREVIEW_LENGTH],
                    "score": round(score, SCORE_PRECISION)
                }
            )
        return results


def embed_chunks() -> None:
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    chunk_embeddings = chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int) -> list[dict]:
    chunked_semantic_search = ChunkedSemanticSearch()
    movies = load_movies()
    chunked_semantic_search.load_or_create_chunk_embeddings(movies)
    return chunked_semantic_search.search_chunks(query, limit)
