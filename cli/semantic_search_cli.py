#!/usr/bin/env python3

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_text
from lib.search_utils import RESULT_LIMIT, CHUNK_SIZE, OVERLAP

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print model informaition")

    embed_text_parser = subparsers.add_parser("embed_text", help="Embed given text into vector")
    embed_text_parser.add_argument("text", type=str, help="Text to embed into vector")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the dataset")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Embed given query into vector")
    embed_query_parser.add_argument("query", type=str, help="Query to embed into vector")

    search_parser = subparsers.add_parser("search", help="Search movies using semantic search")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Set the result limit")

    chunk_parser = subparsers.add_parser("chunk", help="Split long text into smaller pieces for embedding")
    chunk_parser.add_argument("text", help="text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=CHUNK_SIZE, help="chunk size to split text")
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=OVERLAP, help="Previous chunks share the last overlap words with the following chunk")

    args = parser.parse_args()

    match args.command:
        case "verify":
            print("Loading model...")
            verify_model()
        case "embed_text":
            print("Embedding text...")
            embed_text(args.text)
        case "verify_embeddings":
            print("verifying embeddings...")
            verify_embeddings()
        case "embedquery":
            print("Embeding query...")
            embed_query_text(args.query)
        case "search":
            search_command(args.query, args.limit)
        case "chunk":
            print(f"Chunking {len(args.text)} characters")
            chunk_text(args.text, args.chunk_size, args.overlap)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()