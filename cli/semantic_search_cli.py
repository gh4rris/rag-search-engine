#!/usr/bin/env python3

from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text, search_command, chunk_text, semantic_chunk_text
from lib.chunked_semantic_search import embed_chunks, search_chunked_command
from lib.search_utils import RESULT_LIMIT, CHUNK_SIZE, WORD_OVERLAP, SENTENCE_OVERLAP, MAX_CHUNK_SIZE

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
    chunk_parser.add_argument("--overlap", type=int, nargs="?", default=WORD_OVERLAP, help="Previous chunks share the last overlap words with the following chunk")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Splits text on sentence boundaries for embedding")
    semantic_chunk_parser.add_argument("text", help="text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs="?", default=MAX_CHUNK_SIZE, help="Maximum chunk size")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs="?", default=SENTENCE_OVERLAP, help="Previous chunks share the last overlap sentences with the following chunk")

    subparsers.add_parser("embed_chunks", help="Load or build chunk embeddings")

    search_chunked = subparsers.add_parser("search_chunked", help="Search movies using chunked semantic search")
    search_chunked.add_argument("query", type=str, help="Search query")
    search_chunked.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Set the result limit")

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
        case "semantic_chunk":
            print(f"Semantically chunking {len(args.text)} characters")
            semantic_chunk_text(args.text, args.max_chunk_size, args.overlap)
        case "embed_chunks":
            embed_chunks()
        case "search_chunked":
            results = search_chunked_command(args.query, args.limit)
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
                print(f"    {result["document"]}...")
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()