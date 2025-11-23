from lib.keyword_search import tokenize_text
from lib.search_utils import load_data, CACHE

import os
import pickle


class Inverted_Index:
    def __init__(self) -> None:
        self.index: dict[str, set] = {}
        self.docmap: dict[int, dict] = {}
        self.index_path = os.path.join(CACHE, "index.pkl")
        self.docmap_path = os.path.join(CACHE, "docmap.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)
        for token in tokenized_text:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index[term.lower()]
        return sorted(list(doc_ids))
    
    def build(self) -> None:
        movies = load_data()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)