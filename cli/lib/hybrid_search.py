from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search import ChunkedSemanticSearch
from lib.search_utils import LIMIT_MULTIPLIER, SEARCH_MULTIPLIER, RESULT_LIMIT, load_movies
from lib.query_enhancement import enhance_query
from lib.reranking import rerank_results

import os
from typing import Optional


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
        document_scores = combine_scores(bm25_normalized, semantic_normalized, alpha)
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
    
    def rrf_search(self, query:str, k: int, rerank_method: Optional[str]=None, limit: int=RESULT_LIMIT) -> list[dict]:
        bm25_results = self._bm25_search(query, limit * LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(query, limit * LIMIT_MULTIPLIER)
        document_ranks = combine_rrf(bm25_results, semantic_results, k)
        sorted_docs = sorted(document_ranks.items(), key=lambda x: x[1]["rrf_score"], reverse=True)
        search_limit = limit * SEARCH_MULTIPLIER if rerank_method else limit

        results = []
        for id, doc in sorted_docs[:search_limit]:
            results.append(
                {
                    "id": id,
                    "title": doc["title"],
                    "document": doc["document"],
                    "bm25_rank": doc["bm25_rank"],
                    "semantic_rank": doc["semantic_rank"],
                    "rrf_score": doc["rrf_score"]
                }
            )
        return results
    

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

def combine_scores(bm25_results: list[dict], semantic_results: list[dict], alpha: float) -> dict[int, dict]:
    document_scores = {}
    for result in bm25_results:
            doc_id = result["id"]
            document_scores[doc_id] = {
                "title": result["title"],
                "document": result["document"],
                "bm25_score": result["score"],
                "semantic_score": 0,
                "hybrid_score": 0
            }
        
    for result in semantic_results:
        doc_id = result["id"]
        if doc_id not in document_scores:
                document_scores[doc_id] = {
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

    return document_scores

def combine_rrf(bm25_results: list[dict], semantic_results: list[dict], k: int) -> dict[int, dict]:
    document_ranks = {}

    for i, doc in enumerate(bm25_results, 1):
        doc_id = doc["id"]
        document_ranks[doc_id] = {
            "id": doc_id,
            "title": doc["title"],
            "document": doc["document"],
            "bm25_rank": i,
            "semantic_rank": None,
            "rrf_score": 0
        }
    
    for i, doc in enumerate(semantic_results, 1):
        doc_id = doc["id"]
        if doc_id not in document_ranks:
            document_ranks[doc_id] = {
                "id": doc_id,
                "title": doc["title"],
                "document": doc["document"],
                "bm25_rank": None,
                "semantic_rank": i,
                "rrf_score": 0
            }
        else:
            document_ranks[doc_id]["semantic_rank"] = i
            bm25_rrf = 1 / (k + document_ranks[doc_id]["bm25_rank"])
            semantic_rrf = 1 / (k + i)
            document_ranks[doc_id]["rrf_score"] = bm25_rrf + semantic_rrf
    
    return document_ranks


def weighted_command(query: str, alpha: float, limit: int) -> list[dict]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    return hybrid_search.weighted_search(query, alpha, limit)

def rrf_command(query: str, k: int, enhance: Optional[str], rerank_method: Optional[str], limit: int) -> list[dict]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)
    
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)

    results = hybrid_search.rrf_search(query, k, rerank_method, limit)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)
    
    reranked_results = rerank_results(query, rerank_method, results)

    return {
        "enhanced_query": enhanced_query,
        "original_results": results,
        "reranked_results": reranked_results[:limit]
    }