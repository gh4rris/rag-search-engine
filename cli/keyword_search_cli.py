#!/usr/bin/env python3

from lib.keyword_search import search_movies
from lib.inverted_index import build_command, tf_command, idf_command, tfidf_command, bm25_idf_command

import argparse



def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("build", help="Builds inverted index of search tokens to movie ids")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    tf_parser = subparsers.add_parser("tf", help="Get the term frequency for given term and movie")
    tf_parser.add_argument("id", type=int, help="Movie ID")
    tf_parser.add_argument("term", type=str, help="Term to search")

    idf_parser = subparsers.add_parser("idf", help="Get the IDF score for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF score for")

    tfidf_parser = subparsers.add_parser("tfidf", help="Get the TF-IDF score for a given term and movie")
    tfidf_parser.add_argument("id", type=int, help="Movie ID")
    tfidf_parser.add_argument("term", type=str, help="Term to get TF-IDF for")

    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    args = parser.parse_args()

    match args.command:
        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully.")
        case "search":
            print(f"Searching for: {args.query}")
            results = search_movies(args.query)
            for i, result in enumerate(results, 1):
                print(f"{i}: {result}\n")
        case "tf":
            print(f"Term frequency for '{args.term}' in document '{args.id}':")
            count = tf_command(args.id, args.term)
            print(count)
        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
        case "tfidf":
            tfidf = tfidf_command(args.id, args.term)
            print(f"TF-IDF score of '{args.term}' in document '{args.id}': {tfidf:.2f}")
        case "bm25idf":
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()

