#! usr/bin/env python3
from lib.search_utils import RESULT_LIMIT, RRF_K, load_golden_dataset
from lib.hybrid_search import rrf_command

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=RESULT_LIMIT, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_golden_dataset()
    for test_case in golden_dataset:
        results = rrf_command(test_case["query"], RRF_K, limit=limit)
        titles = [doc["title"] for doc in results["results"]]
        relevant_docs = [doc for doc in titles if doc in test_case["relevant_docs"]]
        precision = len(relevant_docs) / len(titles)
        print(f"k={limit}")
        print(f"- Query: {test_case["query"]}")
        print(f"\t- Precision@{limit}: {precision:.4f}")
        print(f"\t- Retrieved: {",".join(titles)}")
        print(f"\t- Relevant: {",".join(relevant_docs)}")

if __name__ == "__main__":
    main()