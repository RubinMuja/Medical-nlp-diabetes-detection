import pandas as pd
from datasets import load_dataset
from pathlib import Path


def create_diabetes_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    diabetes_mask = (
        df["code"].astype(str).str.startswith("E10")
        | df["code"].astype(str).str.startswith("E11")
    )
    df["diabetes_label"] = diabetes_mask.astype(int)
    return df


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)

    print("Loading dataset...")
    dataset = load_dataset("birgermoell/icd10-clinical-notes")

    train_df = dataset["train"].to_pandas()
    test_df = dataset["test"].to_pandas()

    print(f"Original train shape: {train_df.shape}")
    print(f"Original test shape: {test_df.shape}")

    train_df = create_diabetes_label(train_df)
    test_df = create_diabetes_label(test_df)

    train_final = train_df[["journal_note", "code", "name", "diabetes_label"]].copy()
    test_final = test_df[["journal_note", "code", "name", "diabetes_label"]].copy()

    train_path = data_dir / "train_diabetes_notes.csv"
    test_path = data_dir / "test_diabetes_notes.csv"

    train_final.to_csv(train_path, index=False)
    test_final.to_csv(test_path, index=False)

    print("\nSaved processed files:")
    print(train_path)
    print(test_path)

    print("\nTrain label counts:")
    print(train_final["diabetes_label"].value_counts())

    print("\nTest label counts:")
    print(test_final["diabetes_label"].value_counts())


if __name__ == "__main__":
    main()