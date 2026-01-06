from lib.search_utils import load_movies, tokenize_text, CACHE, BM25_K1, BM25_B, SCORE_PRECISION

import os
import pickle
import math
from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set] = defaultdict(set) # mapping tokens to sets of document IDs 
        self.docmap: dict[int, dict] = {} # mapping document IDs to document objects
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter) # mapping document IDs to token counter
        self.doc_lengths: dict[int, int] = {} # mapping document IDs to token length
        self.index_path = os.path.join(CACHE, "index.pkl")
        self.docmap_path = os.path.join(CACHE, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE, "term_frequencies.pkl")
        self.doc_lengths_path = os.path.join(CACHE, "doc_lengths.pkl")

    def __add_document(self, doc_id: int, text: str) -> None:
        tokenized_text = tokenize_text(text)
        for token in set(tokenized_text):
            self.index[token].add(doc_id)
        self.term_frequencies[doc_id].update(tokenized_text)
        self.doc_lengths[doc_id] = len(tokenized_text)

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths or len(self.doc_lengths) == 0:
            return 0
        total = 0
        for length in self.doc_lengths.values():
            total += length
        return total / len(self.doc_lengths)

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
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        doc_length = self.doc_lengths.get(doc_id, 0)
        avg_doc_length = self.__get_avg_doc_length()
        length_norm = (1 - b) + (b * (doc_length / avg_doc_length)) if avg_doc_length > 0 else 1
        tf = self.get_tf(doc_id, term)
        return (tf * (k1 + 1) / (tf + k1 * length_norm))
    
    def get_bm25_idf(self, term: str) -> float:
        token = tokenize_text(term)
        if len(token) != 1:
            raise ValueError("Term must be a single token")
        matches = self.index[token[0]]
        return math.log((len(self.docmap) - len(matches) + 0.5) / (len(matches) + 0.5) + 1)
    
    def bm25(self, doc_id: int, term: str) -> float:
        tf = self.get_bm25_tf(doc_id, term)
        idf = self.get_bm25_idf(term)
        return tf * idf
    
    def bm25_search(self, query: str, limit: int) -> list[dict]:
        tokenized_query = tokenize_text(query)
        bm25_scores = {}
        for id in self.docmap:
            score = 0
            for token in tokenized_query:
                score += self.bm25(id, token)
            bm25_scores[id] = score
        sorted_scores = sorted(bm25_scores.items(), key=lambda x: x[1], reverse=True)
        results = []
        for id, score in sorted_scores[:limit]:
            document = self.docmap[id]
            results.append(
                {
                    "id": document["id"],
                    "title": document["title"],
                    "document": document["description"],
                    "score": round(score, SCORE_PRECISION)
                }
            )
        return results

    
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
        with open(self.doc_lengths_path, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f)
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)
        with open(self.doc_lengths_path, "rb") as f:
            self.doc_lengths = pickle.load(f)


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

def bm25_tf_command(doc_id: int, term: str, k1: float, b: float) -> float:
    movies = InvertedIndex()
    movies.load()
    return movies.get_bm25_tf(doc_id, term, k1, b)

def bm25_idf_command(term: str) -> float:
    movies = InvertedIndex()
    movies.load()
    return movies.get_bm25_idf(term)

def bm25_search(query: str, limit: int) -> list[tuple]:
    movies = InvertedIndex()
    movies.load()
    return movies.bm25_search(query, limit)
