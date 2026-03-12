from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from lime.lime_tabular import LimeTabularExplainer


INPUT_CSV_CANDIDATES = [
    Path("processed") / "heart_filtered_drop_cols_and_rows.csv",
    Path("processed") / "heart_filtered_drop_cols_and_rows",
]

OUT_DIR = Path("processed") / "random-forest-lime"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "num"
RANDOM_STATE = 42

RF_TUNING_TRIES = 4

LIME_NUM_SAMPLES = 500
LIME_NUM_FEATURES = 12
LIME_EXAMPLES = 10
LIME_HTML_LIMIT = 3


def _load_input() -> pd.DataFrame:
    for p in INPUT_CSV_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"""Could not find input file. Tried:
{chr(10).join([str(p) for p in INPUT_CSV_CANDIDATES])}"""
    )


def _coerce_target(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"""Missing target column '{TARGET_COL}'.""")
    df = df.copy()
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="coerce")
    df = df[df[TARGET_COL].notna()].copy()
    df[TARGET_COL] = df[TARGET_COL].round().astype(int)
    df = df[(df[TARGET_COL] >= 0) & (df[TARGET_COL] <= 4)].copy()
    return df


def _build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    num_cols = list(X.columns)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median", add_indicator=True), num_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def _make_rf_candidates(rng: np.random.Generator) -> list[dict]:
    candidates = []
    for _ in range(RF_TUNING_TRIES):
        use_float = bool(rng.integers(0, 2))
        if use_float:
            max_features = float(rng.choice(np.array([0.5, 0.7], dtype=float)))
        else:
            max_features = str(rng.choice(np.array(["sqrt", "log2"], dtype=object)))

        candidates.append(
            {
                "n_estimators": int(rng.integers(400, 1101)),
                "max_depth": rng.choice([None, 8, 10, 12, 14, 16, 20]),
                "min_samples_split": int(rng.choice([2, 4, 6, 8, 10, 12])),
                "min_samples_leaf": int(rng.choice([1, 2, 3, 4, 5, 6, 8])),
                "max_features": max_features,
                "bootstrap": True,
                "class_weight": "balanced_subsample",
            }
        )
    return candidates


def main() -> None:
    df = _load_input()
    df.columns = df.columns.str.strip()
    df = _coerce_target(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    X_tr_raw, X_val_raw, y_tr, y_val = train_test_split(
        X_train_raw,
        y_train,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    preprocessor = _build_preprocessor(X_train_raw)

    rng = np.random.default_rng(RANDOM_STATE)
    candidates = _make_rf_candidates(rng)

    best = None
    best_score = None

    for params in candidates:
        rf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            class_weight=params["class_weight"],
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

        pipe = Pipeline(steps=[("pre", preprocessor), ("model", rf)])
        pipe.fit(X_tr_raw, y_tr)

        val_pred = pipe.predict(X_val_raw)
        val_acc = float(accuracy_score(y_val, val_pred))
        val_mae = float(mean_absolute_error(y_val, val_pred))
        score = val_acc - 0.15 * val_mae

        if best_score is None or score > best_score:
            best_score = score
            best = (pipe, params, val_acc, val_mae)

    if best is None:
        raise RuntimeError("No model candidates were evaluated.")

    pipe, best_params, val_acc, val_mae = best

    pipe.fit(X_train_raw, y_train)

    y_pred = pipe.predict(X_test_raw)
    acc = float(accuracy_score(y_test, y_pred))
    mae = float(mean_absolute_error(y_test, y_pred))

    (OUT_DIR / "best_params.txt").write_text(
        f"""best_params={best_params}
val_accuracy={val_acc:.6f}
val_mae={val_mae:.6f}
""",
        encoding="utf-8",
    )

    metrics_path = OUT_DIR / "metrics.txt"
    metrics_path.write_text(
        f"""test_accuracy={acc:.6f}
test_mae={mae:.6f}
""",
        encoding="utf-8",
    )

    preds_path = OUT_DIR / "predictions_test.csv"
    pd.DataFrame({"y_true": y_test, "y_pred": y_pred}).to_csv(preds_path, index=False)

    X_train_mat = pipe.named_steps["pre"].transform(X_train_raw)
    X_test_mat = pipe.named_steps["pre"].transform(X_test_raw)

    feature_names = list(pipe.named_steps["pre"].get_feature_names_out())

    if not isinstance(X_train_mat, np.ndarray):
        X_train_mat = X_train_mat.toarray()
    if not isinstance(X_test_mat, np.ndarray):
        X_test_mat = X_test_mat.toarray()

    class_names = [str(c) for c in pipe.named_steps["model"].classes_]

    explainer = LimeTabularExplainer(
        training_data=X_train_mat,
        feature_names=feature_names,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        random_state=RANDOM_STATE,
    )

    def predict_proba_fn(x: np.ndarray) -> np.ndarray:
        return pipe.named_steps["model"].predict_proba(x)

    test_errors = np.abs(y_test - y_pred).astype(float)
    order = np.argsort(test_errors)[::-1]
    n = min(LIME_EXAMPLES, len(order))
    idxs = order[:n]

    rows = []
    html_written = 0

    for i in idxs:
        proba = predict_proba_fn(X_test_mat[i].reshape(1, -1))[0]
        label_idx = int(np.argmax(proba))

        exp = explainer.explain_instance(
            X_test_mat[i],
            predict_proba_fn,
            num_features=LIME_NUM_FEATURES,
            num_samples=LIME_NUM_SAMPLES,
            labels=(label_idx,),
        )

        as_list = exp.as_list(label=label_idx)

        payload = {
            "test_index": int(i),
            "y_true": int(y_test[i]),
            "y_pred": int(y_pred[i]),
            "abs_error": float(test_errors[i]),
            "lime_label_idx": int(label_idx),
            "lime_label_name": str(class_names[label_idx]) if 0 <= label_idx < len(class_names) else "",
        }
        for j, (fname, weight) in enumerate(as_list, start=1):
            payload[f"feature_{j}"] = fname
            payload[f"weight_{j}"] = float(weight)
        rows.append(payload)

        if html_written < LIME_HTML_LIMIT:
            html = exp.as_html()
            (OUT_DIR / f"lime_explanation_testidx_{int(i)}.html").write_text(html, encoding="utf-8")
            html_written += 1

    lime_csv = OUT_DIR / "lime_explanations_sample.csv"
    pd.DataFrame(rows).to_csv(lime_csv, index=False)

    print(f"""Saved outputs to: {OUT_DIR}""")
    print(f"""Best params: {best_params}""")
    print(f"""Validation accuracy: {val_acc:.4f} | Validation MAE: {val_mae:.4f}""")
    print(f"""Test accuracy: {acc:.4f}""")
    print(f"""Test MAE: {mae:.4f}""")
    print(f"""Saved LIME CSV: {lime_csv}""")
    print(f"""Saved LIME HTML files: {html_written}""")


if __name__ == "__main__":
    main()
