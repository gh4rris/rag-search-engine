from lib.search_utils import RESULT_LIMIT, tokenize_text
from lib.inverted_index import InvertedIndex


def search_movies(query: str, limit: int=RESULT_LIMIT) -> list[dict]:
    movies = InvertedIndex()
    movies.load()
    tokenized_query = tokenize_text(query)
    seen, results = set(), []
    for token in tokenized_query:
        matching_ids = movies.get_documents(token)
        for id in matching_ids:
            if id in seen:
                continue
            seen.add(id)
            results.append(movies.docmap[id])
            if len(results) >= limit:
                return results
    return results


