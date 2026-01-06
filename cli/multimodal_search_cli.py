from lib.multimodal import verify_image_embedding

import argparse


def main():
    parser = argparse.ArgumentParser(description="Multimodal search CLI")
    subparser = parser.add_subparsers(dest="command", help="Available commands")

    verify_parser = subparser.add_parser("verify_image_embedding", help="Verify dimensions of an image")
    verify_parser.add_argument("image", type=str, help="Path to image file")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            dimensions = verify_image_embedding(args.image)
            print(f"Embedding shape: {dimensions} dimensions")
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()