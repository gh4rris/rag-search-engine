from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.search_utils import LIMIT_MULTIPLIER, load_movies

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
        
    def _bm25_search(self, query: str, limit: int) -> list[dict]:
        self.idx.load()
        return self.idx.bm25_search(query, limit)
    
    def weighted_search(self, query: str, alpha: float, limit: int) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(query, limit * LIMIT_MULTIPLIER)
        bm25_normalized = normalize_results(bm25_results)
        semantic_normalized = normalize_results(semantic_results)
        document_scores = {}
        for result in bm25_normalized:
            doc_id = result["id"]
            document_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": result["score"],
                "semantic_score": 0,
                "hybrid_score": 0
            }
        
        for result in semantic_normalized:
            doc_id = result["id"]
            if doc_id not in document_scores:
                document = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": 0,
                "semantic_score": result["score"],
                "hybrid_score": 0
            }
            else:
                doc = document_scores[doc_id]
                doc["semantic_score"] = result["score"]
                doc["hybrid_score"] = (alpha * doc["bm25_score"]) + ((1 - alpha) * doc["semantic_score"])

        sorted_docs = sorted(document_scores.items(), key=lambda x: x[1]["hybrid_score"], reverse=True)

        results = []
        for id, doc in sorted_docs[:limit]:
            results.append(
                {
                    "id": id,
                    "title": doc["title"],
                    "document": doc["document"],
                    "bm25_score": doc["bm25_score"],
                    "semantic_score": doc["semantic_score"],
                    "hybrid_score": doc["hybrid_score"]
                }
            )
        return results
    
    def rrf_search(self, query:str, k: int, limit: int) -> None:
        raise NotImplementedError("RRF hybrid search is not implemented yet.")
    

def normalize_command(scores: list[float]) -> list[float]:
    if not scores:
        return []
    min_score, max_score = min(scores), max(scores)
    if min_score == max_score:
        normalized_scores = [1.0 for _ in scores]
    else:
        normalized_scores = [(score - min_score) / (max_score - min_score) for score in scores]
    return normalized_scores

def normalize_results(results: list[dict]) -> list[dict]:
    scores = [result["score"] for result in results]
    normalized_scores = normalize_command(scores)
    for i, result in enumerate(results):
        result["score"] = normalized_scores[i]
    return results

def weighted_command(query: str, alpha: float, limit: int) -> list[tuple]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)