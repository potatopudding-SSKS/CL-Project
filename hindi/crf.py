import csv
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
    precision_recall_curve,
    average_precision_score,
)
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import flat_classification_report
import joblib
import os
from collections import Counter
from sklearn.model_selection import learning_curve


# Function to read data from CSV files
def read_data(file_path):
    sentences = []
    labels = []
    raw_texts = []

    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 2:
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

    return sentences, labels, raw_texts


# Function to extract features from each word
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

    # Check if word contains Hindi Unicode characters
    contains_hindi = False
    for char in word:
        char_code = ord(char)
        if 0x0900 <= char_code <= 0x097F:  # Unicode range for Hindi
            contains_hindi = True
            break

    features["contains_hindi"] = contains_hindi

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
                "-1:contains_hindi": any(0x0B80 <= ord(c) <= 0x0BFF for c in word1),
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
                "+1:contains_hindi": any(0x0B80 <= ord(c) <= 0x0BFF for c in word1),
            }
        )
    else:
        features["EOS"] = True  # End of sentence

    return features


# Function to convert sentences to features and labels to numerical tags
def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]


def sent2labels(labels):
    return labels


# Convert data to format required by CRF
def prepare_data(sentences, labels):
    X = [sent2features(s) for s in sentences]
    y = [sent2labels(l) for l in labels]
    return X, y


# Train the CRF model
def train_model(X_train, y_train):
    print("Setting up CRF parameters...")
    crf = sklearn_crfsuite.CRF(
        algorithm="lbfgs",
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True,
    )
    print("Starting CRF training...")
    crf.fit(X_train, y_train)
    return crf


# Evaluate model performance
def evaluate_model(crf, X_test, y_test):
    print("Making predictions...")
    y_pred = crf.predict(X_test)

    # Flatten the prediction and test data for evaluation
    y_test_flat = [int(tag) for sublist in y_test for tag in sublist]
    y_pred_flat = [int(tag) for sublist in y_pred for tag in sublist]

    # Calculate various metrics
    print("Calculating metrics...")
    accuracy = accuracy_score(y_test_flat, y_pred_flat)
    report = classification_report(y_test_flat, y_pred_flat, output_dict=True)
    conf_matrix = confusion_matrix(y_test_flat, y_pred_flat)

    # Calculate precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_flat, y_pred_flat, average="weighted"
    )

    # Get the most informative features
    print("Getting feature importance...")
    state_features = crf.state_features_
    sorted_features = sorted(
        state_features.items(), key=lambda x: abs(x[1]), reverse=True
    )
    top_features = sorted_features[:20]  # Get top 20 features

    # Convert predictions back to integers for output
    y_pred_int = [[int(tag) for tag in seq] for seq in y_pred]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "report": report,
        "confusion_matrix": conf_matrix,
        "y_pred": y_pred_int,
        "top_features": top_features,
    }


# Plot confusion matrix
def plot_confusion_matrix(conf_matrix, classes, title="Confusion Matrix"):
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs("analytics", exist_ok=True)
    plt.savefig("analytics/confusion_matrix.png")
    plt.close()


# Plot feature importance
def plot_feature_importance(top_features, title="Top 20 Features by Importance"):
    features = [f[0][0] for f in top_features]
    importances = [abs(f[1]) for f in top_features]

    plt.figure(figsize=(12, 8))
    plt.barh(range(len(features)), importances, align="center")
    plt.yticks(range(len(features)), features)
    plt.xlabel("Absolute Weight")
    plt.title(title)
    plt.tight_layout()
    os.makedirs("analytics", exist_ok=True)
    plt.savefig("analytics/feature_importance.png")
    plt.close()


# Plot class distribution
def plot_class_distribution(y_train, y_test, class_names):
    train_counts = Counter([int(tag) for sublist in y_train for tag in sublist])
    test_counts = Counter([int(tag) for sublist in y_test for tag in sublist])

    classes = sorted(list(set(train_counts.keys()) | set(test_counts.keys())))
    train_counts_arr = [train_counts.get(c, 0) for c in classes]
    test_counts_arr = [test_counts.get(c, 0) for c in classes]

    plt.figure(figsize=(12, 6))
    x = np.arange(len(classes))
    width = 0.35

    plt.bar(x - width / 2, train_counts_arr, width, label="Training Data")
    plt.bar(x + width / 2, test_counts_arr, width, label="Test Data")

    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.title("Distribution of Classes in Training and Test Data")
    plt.xticks(x, [class_names[i] for i in classes])
    plt.legend()
    plt.tight_layout()
    os.makedirs("analytics", exist_ok=True)
    plt.savefig("analytics/class_distribution.png")
    plt.close()


# Plot error analysis
def plot_error_analysis(test_sentences, y_test, y_pred):
    error_by_length = {}
    total_by_length = {}

    for i, (sentence, true_labels, pred_labels) in enumerate(
        zip(test_sentences, y_test, y_pred)
    ):
        sent_len = len(sentence)
        errors = sum(1 for true, pred in zip(true_labels, pred_labels) if true != pred)

        if sent_len not in total_by_length:
            total_by_length[sent_len] = 0
            error_by_length[sent_len] = 0

        total_by_length[sent_len] += 1
        if errors > 0:
            error_by_length[sent_len] += 1

    lengths = sorted(total_by_length.keys())
    error_rates = [
        error_by_length.get(l, 0) / total_by_length[l] * 100 for l in lengths
    ]

    plt.figure(figsize=(14, 7))
    plt.bar(lengths, error_rates)
    plt.xlabel("Sentence Length (words)")
    plt.ylabel("Error Rate (%)")
    plt.title("Error Rate by Sentence Length")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    os.makedirs("analytics", exist_ok=True)
    plt.savefig("analytics/error_by_length.png")
    plt.close()


# Plot precision-recall curves
def plot_precision_recall(y_test, y_pred_prob, class_names):
    y_test_flat = [int(tag) for sublist in y_test for tag in sublist]

    if y_pred_prob:
        plt.figure(figsize=(10, 8))
        for i, class_name in enumerate(class_names):
            y_bin = [1 if y == i else 0 for y in y_test_flat]
            y_score = [prob[i] for prob in y_pred_prob]

            precision, recall, _ = precision_recall_curve(y_bin, y_score)
            avg_precision = average_precision_score(y_bin, y_score)

            plt.plot(
                recall,
                precision,
                lw=2,
                label=f"Class {class_name} (AP = {avg_precision:.2f})",
            )

        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curves")
        plt.legend(loc="best")
        plt.grid(True)
        os.makedirs("analytics", exist_ok=True)
        plt.savefig("analytics/precision_recall_curves.png")
        plt.close()


# Generate output CSV with predictions
def generate_output(raw_texts, test_sentences, y_pred):
    output_rows = []

    for i, (text, pred) in enumerate(zip(raw_texts, y_pred)):
        if i < len(test_sentences):
            output_rows.append([text, str(pred)])

    os.makedirs("analytics", exist_ok=True)
    with open("analytics/crf-tagged.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(output_rows)


def main():
    os.makedirs("analytics", exist_ok=True)

    train_file = "train.csv"
    test_file = "test.csv"

    print("Reading training data...")
    train_sentences, train_labels, train_texts = read_data(train_file)
    print(f"Loaded {len(train_sentences)} training sentences")

    print("Reading test data...")
    test_sentences, test_labels, test_texts = read_data(test_file)
    print(f"Loaded {len(test_sentences)} test sentences")

    print("Preparing data...")
    X_train, y_train = prepare_data(train_sentences, train_labels)
    X_test, y_test = prepare_data(test_sentences, test_labels)

    print("Training CRF model...")
    crf = train_model(X_train, y_train)

    print("Saving model...")
    joblib.dump(crf, "analytics/crf_model.pkl")

    print("Evaluating model...")
    results = evaluate_model(crf, X_test, y_test)

    with open("analytics/model_metrics.txt", "w") as f:
        f.write("CRF Model Performance Metrics\n")
        f.write("============================\n\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Precision: {results['precision']:.4f}\n")
        f.write(f"Recall: {results['recall']:.4f}\n")
        f.write(f"F1 Score: {results['f1']:.4f}\n\n")

        f.write("Classification Report:\n")
        report = results["report"]
        for label in sorted(report.keys()):
            if label not in ["accuracy", "macro avg", "weighted avg"]:
                f.write(f"Class {label}:\n")
                f.write(f"  Precision: {report[label]['precision']:.4f}\n")
                f.write(f"  Recall: {report[label]['recall']:.4f}\n")
                f.write(f"  F1-score: {report[label]['f1-score']:.4f}\n")
                f.write(f"  Support: {report[label]['support']}\n\n")

    print("\nModel Performance:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1 Score: {results['f1']:.4f}")

    print("\nClassification Report:")
    report = results["report"]
    for label in sorted(report.keys()):
        if label not in ["accuracy", "macro avg", "weighted avg"]:
            print(f"Class {label}:")
            print(f"  Precision: {report[label]['precision']:.4f}")
            print(f"  Recall: {report[label]['recall']:.4f}")
            print(f"  F1-score: {report[label]['f1-score']:.4f}")
            print(f"  Support: {report[label]['support']}")

    print("\nPlotting confusion matrix...")
    class_names = ["Hindi", "Transliterated Hindi", "English", "Disfluency"]
    plot_confusion_matrix(results["confusion_matrix"], class_names)

    print("Plotting feature importance...")
    plot_feature_importance(results["top_features"])

    print("Plotting class distribution...")
    plot_class_distribution(y_train, y_test, class_names)

    print("Plotting error analysis...")
    plot_error_analysis(test_sentences, y_test, results["y_pred"])

    try:
        if hasattr(crf, "predict_marginals"):
            print("Plotting precision-recall curves...")
            y_pred_prob = [crf.predict_marginals([x])[0] for x in X_test]
            plot_precision_recall(y_test, y_pred_prob, class_names)
        else:
            print(
                "Skipping precision-recall curves (model doesn't provide probabilities)"
            )
    except Exception as e:
        print(f"Error generating precision-recall curves: {e}")

    print("Generating output CSV...")
    generate_output(test_texts, test_sentences, results["y_pred"])

    print("\nProcessing complete!")
    print("Model saved as 'analytics/crf_model.pkl'")
    print("Model metrics saved as 'analytics/model_metrics.txt'")
    print("Confusion matrix saved as 'analytics/confusion_matrix.png'")
    print("Feature importance plot saved as 'analytics/feature_importance.png'")
    print("Class distribution saved as 'analytics/class_distribution.png'")
    print("Error analysis saved as 'analytics/error_by_length.png'")
    print("Tagged test data saved as 'analytics/crf-tagged.csv'")


if __name__ == "__main__":
    main()
