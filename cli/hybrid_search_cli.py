#! usr/bin/env python3

from lib.hybrid_search import normalize_command

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparser.add_parser("normalize", help="Normalize scores")
    normalize_parser.add_argument("scores", type=float, nargs="+", help="Scores to normalize")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.scores)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()