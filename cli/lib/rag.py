from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, RRF_K, LLM_MODEL

import os
from typing import Any
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def generate_answer(query: str, documents: list[dict]) -> str:
    doc_list = [f"- {doc.get("title", "")}: {doc.get("document", "")}" for doc in documents]
    prompt = f"""
    Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    Query: {query}

    Documents:
    {"\n\n".join(doc_list)}

    Provide a comprehensive answer that addresses the query:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return (response.text or "").strip()

def generate_summarization(query: str, documents: list[dict]) -> str:
    doc_list = [f"- {doc.get("title", "")}: {doc.get("document", "")}" for doc in documents]
    prompt = f"""
    Provide information useful to this query by synthesizing information from multiple search results in detail.
    The goal is to provide comprehensive information so that users know what their options are.
    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
    This should be tailored to Hoopla users. Hoopla is a movie streaming service.
    Query: {query}
    Search Results:
    {"\n\n".join(doc_list)}
    Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return (response.text or "").strip()

def generate_citations(query: str, documents: list[dict]) -> str:
    doc_list = [f"{i}. {doc.get("title", "")}: {doc.get("document", "")}" for i, doc in enumerate(documents, 1)]
    prompt = f"""
    Answer the question or provide information based on the provided documents.

    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

    If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

    Query: {query}

    Documents:
    {"\n\n".join(doc_list)}

    Instructions:
    - Provide a comprehensive answer that addresses the query
    - Cite sources using [1], [2], etc. format when referencing information
    - If sources disagree, mention the different viewpoints
    - If the answer isn't in the documents, say "I don't have enough information"
    - Be direct and informative

    Answer:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return (response.text or "").strip()

def generate_question_answer(query: str, documents: list[dict]) -> str:
    doc_list = [f"{i}. {doc.get("title", "")}: {doc.get("document", "")}" for i, doc in enumerate(documents, 1)]
    prompt = f"""
    Answer the following question based on the provided documents.

    Question: {query}

    Documents:
    {"\n\n".join(doc_list)}

    General instructions:
    - Answer directly and concisely
    - Use only information from the documents
    - If the answer isn't in the documents, say "I don't have enough information"
    - Cite sources when possible

    Guidance on types of questions:
    - Factual questions: Provide a direct answer
    - Analytical questions: Compare and contrast information from the documents
    - Opinion-based questions: Acknowledge subjectivity and provide a balanced view

    Answer:
    """

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    return (response.text or "").strip()

def rag_command(query: str, limit: int) -> dict[str, Any]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, RRF_K, limit)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    response = generate_answer(query, results[:limit])

    return {
        "results": results[:limit],
        "response": response
    }

def summarize_command(query: str, limit: int) -> dict[str, Any]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, RRF_K, limit)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    response = generate_summarization(query, results[:limit])

    return {
        "results": results[:limit],
        "response": response
    }

def citations_command(query: str, limit: int) -> dict[str, Any]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, RRF_K, limit)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    response = generate_citations(query, results[:limit])

    return {
        "results": results[:limit],
        "response": response
    }

def question_comand(query: str, limit: int) -> dict[str, Any]:
    movies = load_movies()
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, RRF_K, limit)
    results.sort(key=lambda x: x["rrf_score"], reverse=True)

    response = generate_question_answer(query, results[:limit])

    return {
        "results": results[:limit],
        "response": response
    }