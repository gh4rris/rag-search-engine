#! usr/bin/env python3

from lib.hybrid_search import normalize_command, weighted_command, rrf_command
from lib.search_utils import ALPHA, RESULT_LIMIT, RRF_K

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    weighted_search_parser = subparser.add_parser("weighted-search", help="Weighted search")
    weighted_search_parser.add_argument("query", type=str, help="Search query")
    weighted_search_parser.add_argument("--alpha", type=float, nargs="?", default=ALPHA, help="Weighted parameter")
    weighted_search_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Set the result limit")

    rrf_search_parser = subparser.add_parser("rrf-search", help="Reciprocal Rank Fusion Search")
    rrf_search_parser.add_argument("query", type=str, help="Search query")
    rrf_search_parser.add_argument("-k", type=int, nargs="?", default=RRF_K, help="K parameter")
    rrf_search_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Set the result limit")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch"], help="Reranking method")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalized_scores = normalize_command(args.scores)
            for score in normalized_scores:
                print(f"* {score:.4f}")
        case "weighted-search":
            results = weighted_command(args.query, args.alpha, args.limit)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result["title"]}")
                print(f"Hybrid Score: {result["hybrid_score"]:.3f}")
                print(f"BM25: {result["bm25_score"]:.3f}, Semantic: {result["semantic_score"]:.3f}")
                print(f"{result["document"]}...\n")
        case "rrf-search":
            result = rrf_command(args.query, args.k, args.enhance, args.rerank_method, args.limit)
            if args.enhance:
                print(f"Enhanced query ({args.enhance}): '{args.query}' -> '{result["enhanced_query"]}'\n")
            for i, result in enumerate(result["results"], 1):
                print(f"{i}. {result["title"]}")
                if result.get("batch_rank"):
                    print(f"Rerank Rank: {result["batch_rank"]}")
                print(f"RRF Score: {result["rrf_score"]:.3f}")
                print(f"BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}")
                print(f"{result["document"]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()