import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def evaluate_model(model_name: str, model, X_test_tfidf, y_test) -> None:
    y_pred = model.predict(X_test_tfidf)

    print(f"\n{'=' * 60}")
    print(f"{model_name} Evaluation")
    print(f"{'=' * 60}")

    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred))

    print("Confusion Matrix:\n")
    print(confusion_matrix(y_test, y_pred))

    print("\nAccuracy:")
    print(accuracy_score(y_test, y_pred))


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    results_dir = project_root / "results"

    test = pd.read_csv(data_dir / "test_diabetes_notes.csv")

    with open(results_dir / "tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open(results_dir / "logistic_model.pkl", "rb") as f:
        logistic_model = pickle.load(f)

    with open(results_dir / "naive_bayes_model.pkl", "rb") as f:
        naive_bayes_model = pickle.load(f)

    X_test = test["journal_note"]
    y_test = test["diabetes_label"]

    X_test_tfidf = vectorizer.transform(X_test)

    evaluate_model("Logistic Regression", logistic_model, X_test_tfidf, y_test)
    evaluate_model("Naive Bayes", naive_bayes_model, X_test_tfidf, y_test)


if __name__ == "__main__":
    main()