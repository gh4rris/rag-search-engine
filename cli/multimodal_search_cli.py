from lib.multimodal import verify_image_embedding, image_search_command
from lib.search_utils import DOCUMENT_PREVIEW_LENGTH

import argparse


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding", help="Verify dimensions of an image")
    verify_parser.add_argument("image", type=str, help="Path to image file")

    image_search_parser = subparser.add_parser("image_search", help="Search movies using image path")
    image_search_parser.add_argument("image_path", type=str, help="Path location of image")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            dimensions = verify_image_embedding(args.image)
            print(f"Embedding shape: {dimensions} dimensions")
        case "image_search":
            results = image_search_command(args.image_path)
            for i, result in enumerate(results, 1):
                print(f"{i}. {result["title"]} (similarity: {result["similarity_score"]:.3f})")
                print(f"{result["document"][:DOCUMENT_PREVIEW_LENGTH]}...\n")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()