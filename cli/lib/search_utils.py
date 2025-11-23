import os
import json

RESULT_LIMIT = 5

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA = os.path.join(ROOT_PATH, "data", "movies.json")
STOP_WORDS = os.path.join(ROOT_PATH, "data", "stopwords.txt")
CACHE = os.path.join(ROOT_PATH, "cache")


def load_data() -> list[dict]:
    with open(DATA, "r") as f:
        movies = json.load(f)
    return movies["movies"]


def get_stop_words() -> list[str]:
    with open(STOP_WORDS, "r") as f:
        return f.read().splitlines()