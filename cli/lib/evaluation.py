from lib.search_utils import load_movies, load_golden_dataset, RRF_K
from lib.hybrid_search import HybridSearch


def evaluate_command(limit: int) -> dict[str, dict]:
    movies = load_movies()
    golden_dataset = load_golden_dataset()

    hybrid_search = HybridSearch(movies)
    results = {}
    for test_case in golden_dataset:
        case_results = hybrid_search.rrf_search(test_case["query"], RRF_K, limit=limit)
        case_results.sort(key=lambda x: x["rrf_score"], reverse=True)
        titles = [doc["title"] for doc in case_results]
        relevant_docs = [title for title in titles if title in test_case["relevant_docs"]]
        results[test_case["query"]] = {
            "precision": len(relevant_docs) / limit,
            "recall": len(relevant_docs) / len(test_case["relevant_docs"]),
            "retrieved": titles,
            "relevant": relevant_docs
        }
    return results

