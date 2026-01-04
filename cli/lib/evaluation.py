from lib.search_utils import load_movies, load_golden_dataset, RRF_K, LLM_MODEL
from lib.hybrid_search import HybridSearch

import os
import json
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def evaluate_results(query: str, documents: list[dict]) -> list[dict]:
    doc_list = [f"{doc["id"]}: {doc.get("title", "")} - {doc.get("document", "")[:200]}" for doc in documents]
    prompt = f"""
    Rate how relevant each result is to this query on a 0-3 scale:

    Query: "{query}"

    Results:
    {"\n".join(doc_list)}

    Scale:
    - 3: Highly relevant
    - 2: Relevant
    - 1: Marginally relevant
    - 0: Not relevant

    Do NOT give any numbers out than 0, 1, 2, or 3.

    Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

    [2, 0, 3, 2, 0, 1]
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    response_text = (response.text or "").strip()
    evaluation_scores = json.loads(response_text)
    
    results = []
    for doc, score in zip(documents, evaluation_scores):
        results.append({**doc, "llm_score": score})

    return results

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

