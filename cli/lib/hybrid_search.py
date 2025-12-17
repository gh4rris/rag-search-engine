from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch

import os


class HybridSearch:
    def __init__(self, documents: list[dict]) -> None:
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(self.idx.index_path):
            self.idx.build()
            self.idx.save()
        
    def _bm25_search(self, query: str, limit: int) -> list[tuple]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int) -> None:
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")
    
    def rrf_search(self, query:str, k: int, limit: int) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalize_command(scores: list[float]) -> None:
    if not scores:
        return
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        normalized_scores = [1.0 for _ in scores]
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    for score in normalized_scores:
        print(f"* {score:.4f}")