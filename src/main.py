"""Train and evaluate a reproducible SMS spam classifier."""

from pathlib import Path
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "spam.csv"
FIGURE_PATH = PROJECT_ROOT / "reports" / "figures" / "confusion-matrix.svg"


def preprocess_text(text: str) -> str:
    """Normalize an English SMS and remove common stop words."""
    normalized = text.lower()
    normalized = re.sub(r"https?://\S+|www\.\S+", " ", normalized)
    normalized = re.sub(r"[^a-z\s]", " ", normalized)
    tokens = (
        token
        for token in normalized.split()
        if token not in ENGLISH_STOP_WORDS
    )
    return " ".join(tokens)


def load_dataset(path: Path) -> pd.DataFrame:
    """Load the UCI-format CSV and return the two modeled columns."""
    data = pd.read_csv(path, encoding="latin-1", usecols=["v1", "v2"])
    data.columns = ["label", "message"]
    data["target"] = data["label"].map({"ham": 0, "spam": 1})

    if data["target"].isna().any():
        raise ValueError("The dataset contains labels other than 'ham' and 'spam'.")

    data["clean_message"] = data["message"].astype(str).map(preprocess_text)
    return data


def classify_message(
    message: str,
    vectorizer: TfidfVectorizer,
    model: MultinomialNB,
) -> tuple[str, float]:
    """Classify one message and return its label and model confidence."""
    features = vectorizer.transform([preprocess_text(message)])
    prediction = int(model.predict(features)[0])
    probabilities = model.predict_proba(features)[0]
    return ("spam" if prediction == 1 else "ham", float(probabilities[prediction]))


def main() -> None:
    data = load_dataset(DATA_PATH)

    x_train, x_test, y_train, y_test = train_test_split(
        data["clean_message"],
        data["target"],
        test_size=0.2,
        random_state=42,
        stratify=data["target"],
    )

    vectorizer = TfidfVectorizer(max_features=3_000)
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)

    model = MultinomialNB()
    model.fit(x_train_tfidf, y_train)
    predictions = model.predict(x_test_tfidf)

    accuracy = accuracy_score(y_test, predictions)
    print(f"Dataset: {len(data):,} messages")
    print(f"Train/test split: {len(x_train):,}/{len(x_test):,}")
    print(f"Accuracy: {accuracy:.4f}\n")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["ham", "spam"],
            digits=4,
        )
    )

    matrix = confusion_matrix(y_test, predictions)
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["ham", "spam"],
        yticklabels=["ham", "spam"],
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("SMS spam classifier — confusion matrix")
    plt.tight_layout()
    plt.savefig(FIGURE_PATH, dpi=160, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to: {FIGURE_PATH.relative_to(PROJECT_ROOT)}")

    examples = [
        "WINNER! Claim your cash prize now by calling this number.",
        "Sorry, I'll call later. I'm in a meeting right now.",
    ]
    for message in examples:
        label, confidence = classify_message(message, vectorizer, model)
        print(f"{label.upper():4} ({confidence:.1%}) — {message}")


if __name__ == "__main__":
    main()
