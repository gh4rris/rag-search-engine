from lib.search_utils import load_movies, tokenize_text, CACHE

import os
import pickle
import math
from collections import Counter


class InvertedIndex:
    def __init__(self) -> None:
        self.index: dict[str, set] = {}
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: dict[int, Counter] = {}
        self.index_path = os.path.join(CACHE, "index.pkl")
        self.docmap_path = os.path.join(CACHE, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE, "term_frequencies.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokenized_text)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))
    
    def get_tf(self, doc_id: int, term: str) -> int:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term must be a single token")
        counter = self.term_frequencies.get(doc_id, Counter())
        return counter[token[0]]
    
    def get_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term must be a single token")
        matches = self.index[token[0]]
        return math.log((len(self.docmap) + 1) / (len(matches) + 1))
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term must be a single token")
        matches = self.index[token[0]]
        return math.log((len(self.docmap) - len(matches) + 0.5) / (len(matches) + 0.5) + 1)
    
    def build(self) -> None:
        movies = load_movies()
        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie
    
    def save(self) -> None:
        os.makedirs(CACHE, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)


def build_command() -> None:
    inverted_index = InvertedIndex()
    inverted_index.build()
    inverted_index.save()

def tf_command(doc_id: int, term: str) -> int:
    movies = InvertedIndex()
    movies.load()
    return movies.get_tf(doc_id, term)

def idf_command(term: str) -> float:
    movies = InvertedIndex()
    movies.load()
    return movies.get_idf(term)

def tfidf_command(doc_id: int, term: str) -> float:
    movies = InvertedIndex()
    movies.load()
    tf = movies.get_tf(doc_id, term)
    idf = movies.get_idf(term)
    return tf * idf

def bm25_idf_command(term: str) -> float:
    movies = InvertedIndex()
    movies.load()
    return movies.get_bm25_idf(term)