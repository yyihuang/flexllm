from datasets import load_dataset
from transformers import AutoTokenizer
import json, argparse, os

def main(model_name, max_length, output_file):
    # Load a sample dataset; adjust the dataset name and split as needed.
    dataset = load_dataset("simplescaling/s1K_tokenized", split="train")  # using a subset for speed

    # Load a pre-trained tokenizer.
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Function to tokenize text and add a token count.
    def tokenize_count(example):
        # Tokenize the text field
        tokens = tokenizer.tokenize(example["text"])
        # Save the number of tokens to a new field
        example["token_count"] = len(tokens)
        return example

    # Apply the function to each example in the dataset.
    tokenized_dataset = dataset.map(tokenize_count)

    # Filter entries with token_count less than max_length.
    filtered_dataset = tokenized_dataset.filter(lambda example: example["token_count"] < max_length)

    # Extract the original text field from the filtered examples.
    text_list = filtered_dataset["text"]

    # Save the text list to a JSON file.
    with open(output_file, "w") as f:
        json.dump(text_list, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build S1 training trace")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-70B-Instruct", help="Model name")
    parser.add_argument("-m", "--max-length", type=int, default=8192, help="Max dataset size")
    parser.add_argument("-o", "--output_file", type=str, default="./traces/s1.json", help="Output file name")
    args = parser.parse_args()

    # Change directory to that holding this script
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    os.makedirs("./traces", exist_ok=True)

    main(args.model_name, args.max_length, args.output_file)
