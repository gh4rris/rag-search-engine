from lib.hybrid_search import HybridSearch
from lib.search_utils import load_movies, RRF_K, LLM_MODEL

import os
import mimetypes
from typing import Any
from dotenv import load_dotenv
from google import genai


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)


def describe_command(image_path: str, query: str) -> dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    mime, _ = mimetypes.guess_type(image_path)
    mime = mime or "image/jpeg"

    with open(image_path, "rb") as f:
        image = f.read()
    
    system_prompt = """
    Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
    - Synthesize visual and textual information
    - Focus on movie-specific details (actors, scenes, style, etc.)
    - Return only the rewritten query, without any additional commentary
    """
    parts = [
        system_prompt,
        genai.types.Part.from_bytes(data=image, mime_type=mime),
        query.strip()
    ]

    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=parts
    )

    return {
        "response": response.text.strip() or "",
        "tokens": response.usage_metadata.total_token_count if response.usage_metadata else None
    }