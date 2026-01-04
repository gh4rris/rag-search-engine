#! usr/bin/env python3
from lib.search_utils import RESULT_LIMIT
from lib.evaluation import evaluate_command

import argparse

def main() -> None:
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=RESULT_LIMIT, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    results = evaluate_command(args.limit)

    print(f"k={args.limit}\n")
    for query, result in results.items():
        precision, recall = result["precision"], result["recall"]
        f1score = (2 * (precision * recall)) / (precision + recall)
        print(f"- Query: {query}")
        print(f"\t- Precision@{args.limit}: {precision:.4f}")
        print(f"\t- Recall@{args.limit}: {recall:.4f}")
        print(f"\t- F1 Score: {f1score:.4f}")
        print(f"\t- Retrieved: {", ".join(result["retrieved"])}")
        print(f"\t- Relevant: {", ".join(result["relevant"])}")

if __name__ == "__main__":
    main()