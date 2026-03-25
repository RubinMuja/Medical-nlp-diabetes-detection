import pickle
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"
    results_dir.mkdir(exist_ok=True)

    train_path = data_dir / "train_diabetes_notes.csv"
    test_path = data_dir / "test_diabetes_notes.csv"

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")

    X_train = train["journal_note"]
    y_train = train["diabetes_label"]

    X_test = test["journal_note"]
    y_test = test["diabetes_label"]

    vectorizer = TfidfVectorizer(
        max_features=5000,
        stop_words="english"
    )

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF train shape: {X_train_tfidf.shape}")
    print(f"TF-IDF test shape: {X_test_tfidf.shape}")

    logistic_model = LogisticRegression(max_iter=1000)
    logistic_model.fit(X_train_tfidf, y_train)

    naive_bayes_model = MultinomialNB()
    naive_bayes_model.fit(X_train_tfidf, y_train)

    with open(results_dir / "tfidf_vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    with open(results_dir / "logistic_model.pkl", "wb") as f:
        pickle.dump(logistic_model, f)

    with open(results_dir / "naive_bayes_model.pkl", "wb") as f:
        pickle.dump(naive_bayes_model, f)

    print("\nSaved models and vectorizer to results/")


if __name__ == "__main__":
    main()