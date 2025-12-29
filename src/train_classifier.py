import argparse
import ast
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train academic abstract classifier.")
    parser.add_argument("--data-path", type=Path, required=True, help="CSV with 'abstract' and 'field'.")
    parser.add_argument("--model-dir", type=Path, default=Path("models"), help="Directory to store artifacts.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Holdout split size.")
    parser.add_argument("--max-features", type=int, default=5000, help="TF-IDF max features.")
    return parser.parse_args()


def _parse_terms_cell(cell: str) -> list[str]:
    """
    Parse the `terms` column which is stored like "['cs.CV', 'cs.LG']" into
    a list of strings. Falls back to splitting on common delimiters.
    """
    cell = str(cell)
    try:
        parsed = ast.literal_eval(cell)
        if isinstance(parsed, (list, tuple)):
            return [str(t) for t in parsed]
        return [str(parsed)]
    except Exception:
        parts = re.split(r"[|,]", cell)
        return [p.strip() for p in parts if p.strip()]


def _map_arxiv_terms_to_field(terms: list[str]) -> str:
    """
    Map fine-grained ArXiv category codes (e.g. 'cs.AI', 'cs.CV') to a small
    set of coarse fields suitable for the UI: 'AI', 'Healthcare',
    'Business', 'Environmental science', or 'Other'.
    """
    if not terms:
        return "Other"

    cats = set(terms)

    # --- AI & ML ------------------------------------------------------------
    if any(
        t.startswith(prefix)
        for t in cats
        for prefix in ("cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.NE", "cs.RO", "stat.ML")
    ):
        return "AI"

    # --- Healthcare / Medicine / Bio ---------------------------------------
    if any(
        t.startswith(prefix)
        for t in cats
        for prefix in ("q-bio.", "physics.med-ph")
    ):
        return "Healthcare"

    # --- Business / Finance / Econ -----------------------------------------
    if any(
        t.startswith(prefix)
        for t in cats
        for prefix in ("q-fin.", "econ.")
    ):
        return "Business"

    # --- Environmental science / Earth / Climate ---------------------------
    if any(
        t.startswith(prefix)
        for t in cats
        for prefix in ("physics.ao-ph", "physics.geo-ph", "physics.oc", "cs.SD")
    ):
        return "Environmental science"

    return "Other"


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data_path)

    # Support both the generic 'abstract/field' schema and the provided
    # ArXiv-style 'summaries/terms' schema.
    if {"abstract", "field"}.issubset(df.columns):
        text_series = df["abstract"].astype(str)
        label_series = df["field"].astype(str)
    elif {"summaries", "terms"}.issubset(df.columns):
        text_series = df["summaries"].astype(str)
        raw_terms = df["terms"].astype(str).apply(_parse_terms_cell)
        label_series = raw_terms.apply(_map_arxiv_terms_to_field)
    else:
        raise ValueError(
            "CSV must contain 'abstract' and 'field' columns OR 'summaries' and 'terms' columns."
        )

    n_classes = label_series.nunique()
    stratify_labels = label_series if len(df) * args.test_size >= n_classes else None

    X_train, X_test, y_train, y_test = train_test_split(
        text_series,
        label_series,
        test_size=args.test_size,
        stratify=stratify_labels,
        random_state=42,
    )

    pipeline = Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=args.max_features,
                    ngram_range=(1, 2),
                    stop_words="english",
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=200,
                    multi_class="multinomial",
                    solver="lbfgs",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    print(classification_report(y_test, y_pred))

    args.model_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.model_dir / "abstract_classifier.joblib"
    joblib.dump(pipeline, model_path)
    print(f"Saved model to {model_path}")


if __name__ == "__main__":
    main()

