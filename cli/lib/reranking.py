from lib.search_utils import LLM_MODEL, CROSS_ENCODER_MODEL

import os
import time
import json
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def rerank_individual(query: str, documents: list[dict]) -> list[dict]:
    scored_docs = []
    for doc in documents:
        prompt = f"""
        Rate how well this movie matches the search query.

        Query: "{query}"
        Movie: {doc.get("title", "")} - {doc.get("document", "")}

        Consider:
        - Direct relevance to query
        - User intent (what they're looking for)
        - Content appropriateness

        Rate 0-10 (10 = perfect match).
        Give me ONLY the number in your response, no other text or explanation.

        Score:
        """

        response = client.models.generate_content(
            model=LLM_MODEL,
            contents=prompt
        )
        score = int((response.text or "").strip())
        scored_docs.append({**doc, "individual_score": score})
        time.sleep(6)
    scored_docs.sort(key=lambda x: x["individual_score"], reverse=True)
    return scored_docs

def rerank_batch(query: str, documents: list[dict]) -> list[dict]:
    doc_map = {doc["id"]: doc for doc in documents}
    doc_list = [f"{doc["id"]}: {doc.get("title", "")} - {doc.get("document", "")[:200]}" for doc in documents]
    doc_list_str = "\n".join(doc_list)
    prompt = f"""
    Rank these movies by relevance to the search query.

    Query: "{query}"

    Movies:
    {doc_list_str}

    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

    [75, 12, 34, 2, 1]
    """
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt
    )
    response_text = (response.text or "").strip()
    ranked_ids = json.loads(response_text)
    
    reranked = []
    for rank, id in enumerate(ranked_ids, 1):
        if id in doc_map:
            reranked.append({**doc_map[id], "batch_rank": rank})
    return reranked

def rerank_cross_encoder(query: str, documents: list[dict]) -> list[dict]:
    pairs = [[query, f"{doc.get("title", "")} - {doc.get("document", "")}"] for doc in documents]
    cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)
    scores = cross_encoder.predict(pairs)
    scored_docs = []
    for doc, score in zip(documents, scores):
        scored_docs.append({**doc, "cross_encoder_score": score})
    scored_docs.sort(key=lambda x: x["cross_encoder_score"], reverse=True)
    return scored_docs


def rerank_results(query: str, rerank_method: str, documents: list[dict]) -> list[dict]:
    match rerank_method:
        case "individual":
            return rerank_individual(query, documents)
        case "batch":
            return rerank_batch(query, documents)
        case "cross_encoder":
            return rerank_cross_encoder(query, documents)
        case _:
            return documents