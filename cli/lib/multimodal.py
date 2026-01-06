from lib.search_utils import LLM_MODEL, MULTIMODAL_MODEL

import os
import mimetypes
from typing import Any
from numpy import ndarray
from dotenv import load_dotenv
from google import genai
from PIL import Image
from sentence_transformers import SentenceTransformer


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

class MultiModalSearch:
    def __init__(self, model_name: str=MULTIMODAL_MODEL):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, path: str) -> ndarray:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Image file not found: {path}")
        image = Image.open(path)
        return self.model.encode([image])[0]
    

def verify_image_embedding(image_path: str) -> int:
    multimodal = MultiModalSearch()
    embedded_image = multimodal.embed_image(image_path)
    return embedded_image.shape[0]


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