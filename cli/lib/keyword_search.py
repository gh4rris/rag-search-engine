import string

from lib.search_utils import RESULT_LIMIT, load_data


def search_movies(query: str, limit: int=RESULT_LIMIT) -> list[dict]:
    movies = load_data()
    results = []
    for movie in movies["movies"]:
        tokenized_query = tokenize_text(query)
        tokenized_title = tokenize_text(movie["title"])
        if has_matching_token(tokenized_query, tokenized_title):
            results.append(movie)
        if len(results) >= limit:
            break
    return results


def has_matching_token(tokenized_query: list[str], tokenized_title: list[str]) -> bool:
    for q_token in tokenized_query:
        for t_token in tokenized_title:
            if q_token in t_token:
                return True
    return False


def tokenize_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()