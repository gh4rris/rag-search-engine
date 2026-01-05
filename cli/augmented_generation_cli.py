from lib.rag import rag_command

import argparse


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_result = rag_command(args.query)
            print("Search Results:")
            for result in rag_result["results"]:
                print(f"\t- {result["title"]}")
            
            print(f"\nRAG Response:")
            print(rag_result["response"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()