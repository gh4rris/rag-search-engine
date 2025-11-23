#!/usr/bin/env python3

from lib.keyword_search import search_movies
from lib.inverted_index import Inverted_Index

import argparse



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Builds inverted index of search tokens to movie ids")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies(args.query)
            for i, result in enumerate(results):
                print(f"{i + 1}: {result}\n")
        case "build":
            inverted_index = Inverted_Index()
            inverted_index.build()
            inverted_index.save()
            doc_ids = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {doc_ids[0]}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

