from lib.multimodal import describe_command

import argparse
import mimetypes


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="Path to image file")
    parser.add_argument("--query", type=str, help="Query to rewrite based on the image")

    args = parser.parse_args()

    result = describe_command(args.image, args.query)
    
    print(f"Rewritten query: {result["response"]}")
    if result["tokens"]:
        print(f"Total tokens: {result["tokens"]}")


if __name__ == "__main__":
    main()