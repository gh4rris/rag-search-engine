import os
from dotenv import load_dotenv
from google import genai
from typing import Optional


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash"


def spell_check(query: str) -> str:
    prompt = f"""Fix any spelling errors in this movie search query.
    Only correct obvious typos. Don't change correctly spelled words
    and ignore case.
    Query: "{query}"
    If no errors, return the original query.
    Corrected:"""

    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    corrected = (response.text or "").strip().strip('"')
    return corrected if corrected else query

def enhance_query(query: str, method: Optional[str]=None) -> str:
    match method:
        case "spell":
            return spell_check(query)
        case _:
            return query