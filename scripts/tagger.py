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


# Function to read data from CSV file
def read_data(file_path, has_labels=True):
    sentences = []
    labels = []
    raw_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if has_labels and len(row) == 2:
                text = row[0]
                # Parse the label list from string format
                try:
                    label_str = row[1].strip()
                    # Skip header row if exists
                    if label_str == "labels":
                        continue

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
            elif not has_labels and len(row) >= 1:
                text = row[0]
                # Skip header row if it exists
                if text == "text" or text.startswith("text"):
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


def main():
    parser = argparse.ArgumentParser(description="Tag text data using a CRF model")
    parser.add_argument(
        "--input", default="test.csv", help="Input CSV file with text data"
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
        "--has-labels",
        action="store_true",
        help="Specify if input file already has labels (for evaluation)",
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

    # Read data
    print(f"Reading data from {args.input}...")
    if args.has_labels:
        sentences, labels, raw_texts = read_data(args.input, has_labels=True)
        print(f"Loaded {len(sentences)} sentences with labels")
    else:
        sentences, raw_texts = read_data(args.input, has_labels=False)
        print(f"Loaded {len(sentences)} sentences without labels")

    # Tag the data
    print("Tagging data...")
    predicted_labels = tag_data(model, sentences)

    # Save results
    print("Saving results...")
    save_results(raw_texts, predicted_labels, args.output)

    # If input had labels, evaluate performance
    if args.has_labels:
        from sklearn.metrics import accuracy_score

        # Flatten the labels for evaluation
        y_true_flat = [int(tag) for sublist in labels for tag in sublist]
        y_pred_flat = [tag for sublist in predicted_labels for tag in sublist]

        accuracy = accuracy_score(y_true_flat, y_pred_flat)
        print(f"\nEvaluation on provided data:")
        print(f"Accuracy: {accuracy:.4f}")

        # Map numeric labels to readable names
        class_names = ["Tamil", "Transliterated Tamil", "English", "Disfluency"]
        print("\nTag interpretation:")
        for i, name in enumerate(class_names):
            print(f"{i}: {name}")


if __name__ == "__main__":
    main()
