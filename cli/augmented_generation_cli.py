from lib.rag import rag_command, summarize_command, citations_command, question_comand
from lib.search_utils import RESULT_LIMIT

import argparse


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    rag_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Result limit")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize query")
    summarize_parser.add_argument("query", type=str, help="Query to summarize")
    summarize_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Result limit")

    citations_parser = subparsers.add_parser("citations", help="Add citations")
    citations_parser.add_argument("query", type=str, help="Query to answer with citations")
    citations_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Result limit")

    question_parser = subparsers.add_parser("question", help="Answer question")
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument("--limit", type=int, nargs="?", default=RESULT_LIMIT, help="Result limit")

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_result = rag_command(args.query, args.limit)
            print("Search Results:")
            for result in rag_result["results"]:
                print(f"\t- {result["title"]}")
            
            print(f"\nRAG Response:")
            print(rag_result["response"])
        case "summarize":
            summarize_result = summarize_command(args.query, args.limit)
            print("Search Results:")
            for result in summarize_result["results"]:
                print(f"\t- {result["title"]}")
            
            print(f"\nLLM Summary:")
            print(summarize_result["response"])
        case "citations":
            citations_result = citations_command(args.query, args.limit)
            print("Search Results:")
            for result in citations_result["results"]:
                print(f"\t- {result["title"]}")
            
            print(f"\nLLM Answer:")
            print(citations_result["response"])
        case "question":
            question_result = question_comand(args.question, args.limit)
            print("Search Results:")
            for result in question_result["results"]:
                print(f"\t- {result["title"]}")
            
            print(f"\nAnswer:")
            print(question_result["response"])
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()