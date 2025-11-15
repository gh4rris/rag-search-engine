import os
import json

RESULT_LIMIT = 5

ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT_PATH, "data", "movies.json")


def load_data():
    with open(DATA_PATH, "r") as f:
        movies = json.load(f)
    return movies