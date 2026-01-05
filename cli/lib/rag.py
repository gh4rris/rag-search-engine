from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, RRF_K, LLM_MODEL, RESULT_LIMIT

import os
from typing import Any
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def augment_generation(query: str, documents: list[dict], limit: int=RESULT_LIMIT) -> str:
    doc_list = [f"- {doc.get("title", "")}: {doc.get("document", "")[:200]}" for doc in documents[:limit]]
    prompt = f"""
    Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {"\n".join(doc_list)}

    Provide a comprehensive answer that addresses the query:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return response.text

def rag_command(query: str) -> dict[str, Any]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, RRF_K)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    response = augment_generation(query, results)

    return {
        "results": results,
        "response": response
    }