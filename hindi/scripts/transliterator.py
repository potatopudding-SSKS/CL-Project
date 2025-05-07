from ai4bharat.transliteration import XlitEngine
import csv
import ast
import joblib
import os
import argparse


# Function to extract features from each word (same as in crf.py)
def word2features(sent, i):
    word = sent[i]

    # Basic features
    features = {
        "bias": 1.0,
        "word.lower()": word.lower(),
        "word[-3:]": word[-3:] if len(word) > 2 else word,
        "word[-2:]": word[-2:] if len(word) > 1 else word,
        "word.isupper()": word.isupper(),
        "word.istitle()": word.istitle(),
        "word.isdigit()": word.isdigit(),
        "word.length": len(word),
        "has_hyphen": "-" in word,
        "has_special_chars": any(c for c in word if not c.isalnum() and c != "-"),
    }

    # Check if word contains Tamil Unicode characters
    contains_tamil = False
    for char in word:
        char_code = ord(char)
        if 0x0B80 <= char_code <= 0x0BFF:  # Unicode range for Tamil
            contains_tamil = True
            break

    features["contains_tamil"] = contains_tamil

    # Features based on character patterns
    # English words typically use Latin characters
    features["has_latin"] = any(c.isalpha() and ord(c) < 128 for c in word)

    # Add position features
    if i > 0:
        word1 = sent[i - 1]
        features.update(
            {
                "-1:word.lower()": word1.lower(),
                "-1:word.istitle()": word1.istitle(),
                "-1:word.isupper()": word1.isupper(),
                "-1:word.length": len(word1),
                "-1:contains_tamil": any(0x0B80 <= ord(c) <= 0x0BFF for c in word1),
            }
        )
    else:
        features["BOS"] = True  # Beginning of sentence

    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update(
            {
                "+1:word.lower()": word1.lower(),
                "+1:word.istitle()": word1.istitle(),
                "+1:word.isupper()": word1.isupper(),
                "+1:word.length": len(word1),
                "+1:contains_tamil": any(0x0B80 <= ord(c) <= 0x0BFF for c in word1),
            }
        )
    else:
        features["EOS"] = True  # End of sentence

    return features


# Function to convert sentences to features
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


# Function to read data from input file
def read_data(file_path, has_labels=True):
    """
    Read data from input file.
    Args:
        file_path: Path to input file
        has_labels: Whether the input file has labels (default: True)
    Returns:
        sentences: List of lists of words
        labels: List of lists of labels (if has_labels is True)
        raw_texts: List of original texts
    """
    sentences = []
    labels = []
    raw_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        if has_labels:
            # For CSV files with text,labels format
            reader = csv.reader(f)
            for row in reader:
                if len(row) == 2:
                    text, label_str = row
                    # Skip header if it exists
                    if text == "text" or label_str == "labels":
                        continue

                    try:
                        # Convert string representation of list to actual list
                        label_list = ast.literal_eval(label_str)

                        # Split the text into words
                        words = text.split()

                        # Only process if number of words matches number of labels
                        if len(words) == len(label_list):
                            sentences.append(words)
                            raw_texts.append(text)
                            # Convert integers to strings for CRF
                            string_labels = [str(label) for label in label_list]
                            labels.append(string_labels)
                    except (SyntaxError, ValueError) as e:
                        print(f"Error parsing row: {row}")
                        print(f"Error details: {e}")
                        continue
        else:
            # For text files with one comment per line
            lines = f.readlines()
            for text in lines:
                text = text.strip()
                # Skip empty lines
                if not text:
                    continue

                # Split the text into words
                words = text.split()
                if words:
                    sentences.append(words)
                    raw_texts.append(text)

    if has_labels:
        return sentences, labels, raw_texts
    else:
        return sentences, raw_texts


# Function to tag data with the model
def tag_data(model, sentences):
    # Convert sentences to feature representation
    X = [sent2features(s) for s in sentences]

    # Use the model to predict tags
    y_pred = model.predict(X)

    # Convert string labels back to integers
    y_pred_int = [[int(tag) for tag in seq] for seq in y_pred]

    return y_pred_int


# Function to save results to CSV
def save_results(raw_texts, predicted_labels, output_file):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for text, labels in zip(raw_texts, predicted_labels):
            writer.writerow([text, str(labels)])

    print(f"Results saved to {output_file}")


# New function to process tagged output with transliteration
def process_tagged_output(
    input_csv, transliteration_engine, output_txt="pipeline-io\output.txt"
):
    print(f"Reading tagged data from {input_csv}...")

    # Process file and write output
    with open(input_csv, "r", encoding="utf-8") as infile, open(
        output_txt, "w", encoding="utf-8"
    ) as outfile:
        reader = csv.reader(infile)
        for row in reader:
            if len(row) != 2:
                continue

            text, labels_str = row
            try:
                # Parse labels and split text into words
                labels = ast.literal_eval(labels_str)
                words = text.split()

                if len(words) != len(labels):
                    print(f"Warning: Mismatch in words and labels length for: {text}")
                    continue

                # Process each word based on its label
                processed_words = []
                for word, label in zip(words, labels):
                    if label == 1:  # Transliterated Tamil
                        try:
                            result = transliteration_engine.translit_word(word, topk=1)
                            # Handle different return types from transliteration engine
                            if isinstance(result, list) and result:
                                transliterated = result[0]
                            elif isinstance(result, dict) and "hi" in result:
                                transliterated = result["hi"][0]
                            else:
                                transliterated = word
                            processed_words.append(transliterated)
                            print(f"Transliterated: {word} -> {transliterated}")
                        except Exception as e:
                            print(f"Error transliterating '{word}': {e}")
                            processed_words.append(word)
                    else:
                        processed_words.append(word)

                # Write the processed sentence to output file
                processed_sentence = " ".join(processed_words)
                outfile.write(processed_sentence + "\n")

            except (SyntaxError, ValueError) as e:
                print(f"Error processing row: {row}")
                print(f"Error details: {e}")
                continue

    print(f"Transliteration completed. Results saved to {output_txt}")


def main():
    parser = argparse.ArgumentParser(
        description="Tag text data using a CRF model and transliterate tagged words"
    )
    parser.add_argument(
        "--input", default="pipeline-io/input.txt", help="Input text file with comments"
    )
    parser.add_argument(
        "--model", default="analytics/crf_model.pkl", help="Path to trained CRF model"
    )
    parser.add_argument(
        "--output",
        default="analytics/crf-tagged.csv",
        help="Output CSV file for tagged data",
    )
    parser.add_argument(
        "--processed-output",
        default="pipeline-io/output.txt",
        help="Output TXT file for processed text",
    )

    args = parser.parse_args()

    # Check if model exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        return

    # Check if input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input file not found at {args.input}")
        return

    # Load the CRF model
    print(f"Loading CRF model from {args.model}...")
    model = joblib.load(args.model)

    # Read data - note we're setting has_labels=False since input.txt doesn't have labels
    print(f"Reading data from {args.input}...")
    sentences, raw_texts = read_data(args.input, has_labels=False)
    print(f"Loaded {len(sentences)} sentences")

    # Tag the data
    print("Tagging data...")
    predicted_labels = tag_data(model, sentences)

    # Save results
    print("Saving results...")
    save_results(raw_texts, predicted_labels, args.output)

    # Initialize the transliteration engine
    print("Initializing transliteration engine...")
    try:
        e = XlitEngine("hi", beam_width=10)
        print("Transliteration engine initialized successfully.")

        # Ensure the output directory exists
        output_dir = os.path.dirname(args.processed_output)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Process the tagged output with transliteration
        process_tagged_output(args.output, e, args.processed_output)
    except Exception as e:
        print(f"Error initializing transliteration engine or processing output: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()


# torch = 2.5.0
# fairseq = git+https://github.com/One-sixth/fairseq.git@44800430a728c2216fd1cf1e8daa672f50dfacba
# python = 3.12.x
# pip = 23.3.1
