#!/usr/bin/env python3

from lib.semantic_search import verify_model, embed_text, verify_embeddings

import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    subparsers.add_parser("verify", help="Print model informaition")

    embed_parser = subparsers.add_parser("embed_text", help="Embed given text into vector")
    embed_parser.add_argument("text", type=str, help="Text to embed into vector")

    subparsers.add_parser("verify_embeddings", help="Verify embeddings for the dataset")

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
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()