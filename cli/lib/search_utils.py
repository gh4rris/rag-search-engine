import os
import json
import string
from nltk.stem import PorterStemmer

RESULT_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
CHUNK_SIZE = 200
OVERLAP = 0

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA = os.path.join(ROOT_PATH, "data", "movies.json")
STOP_WORDS = os.path.join(ROOT_PATH, "data", "stopwords.txt")
CACHE = os.path.join(ROOT_PATH, "cache")


def load_movies() -> list[dict]:
    with open(DATA, "r") as f:
        movies = json.load(f)
    return movies["movies"]


def get_stop_words() -> list[str]:
    with open(STOP_WORDS, "r") as f:
        return f.read().splitlines()
    
    
def tokenize_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    stop_words = get_stop_words()
    filtered_words = [word for word in text.split() if word not in stop_words]
    stemmer = PorterStemmer()
    return [stemmer.stem(token) for token in filtered_words]