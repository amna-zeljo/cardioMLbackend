from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

try:
    import shap
except ImportError as e:
    raise ImportError(
        f"""shap is not installed. Install it with:
pip install shap
Original error: {e}"""
    )

try:
    from xgboost import XGBClassifier
except ImportError as e:
    raise ImportError(
        f"""xgboost is not installed. Install it with:
pip install xgboost
Original error: {e}"""
    )

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt


INPUT_CSV_CANDIDATES = [
    #Path("processed") / "heart_filtered_drop_cols_and_rows.csv",
    Path("processed") / "xgboost-fill-missing/heart_imputed_xgb.csv",
]

OUTPUT_DIR = Path("processed/xgboost-shap")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_OUT = OUTPUT_DIR / "heart_binary_predictions_test.csv"

TARGET_COL = "num"
RANDOM_STATE = 42

SHAP_SAMPLE_SIZE = 500


def _load_input() -> pd.DataFrame:
    for p in INPUT_CSV_CANDIDATES:
        if p.exists():
            return pd.read_csv(p)
    raise FileNotFoundError(
        f"""Could not find input file. Tried:
{chr(10).join([str(p) for p in INPUT_CSV_CANDIDATES])}"""
    )


def _coerce_target_binary(df: pd.DataFrame) -> pd.DataFrame:
    if TARGET_COL not in df.columns:
        raise ValueError(f"""Missing target column '{TARGET_COL}'.""")

    out = df.copy()
    out[TARGET_COL] = pd.to_numeric(out[TARGET_COL], errors="coerce")
    out = out[out[TARGET_COL].notna()].copy()
    out[TARGET_COL] = out[TARGET_COL].round().astype(int)
    out = out[(out[TARGET_COL] == 0) | (out[TARGET_COL] == 1)].copy()
    return out


def _build_preprocessor(df: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    cols = [c.strip() for c in df.columns]
    df.columns = cols

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    cat_cols: List[str] = []
    num_cols: List[str] = []

    for c in feature_cols:
        s = df[c]
        if pd.api.types.is_bool_dtype(s) or pd.api.types.is_object_dtype(s) or pd.api.types.is_categorical_dtype(s):
            cat_cols.append(c)
        else:
            num_cols.append(c)

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []
    if len(num_cols) > 0:
        transformers.append(("num", SimpleImputer(strategy="median", add_indicator=True), num_cols))
    if len(cat_cols) > 0:
        transformers.append(
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", ohe),
                    ]
                ),
                cat_cols,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor, num_cols, cat_cols


def _feature_names(preprocessor: ColumnTransformer) -> List[str]:
    names = preprocessor.get_feature_names_out()
    return [str(n) for n in names]


def _train_model(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    model = XGBClassifier(
        n_estimators=700,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train.astype(int))
    return model


def _save_shap_outputs(model: XGBClassifier, X_test_df: pd.DataFrame) -> None:
    out_summary = OUTPUT_DIR / "shap_summary_binary.png"
    out_waterfall = OUTPUT_DIR / "shap_waterfall_binary_row0.png"

    explainer = shap.TreeExplainer(model)

    n = min(SHAP_SAMPLE_SIZE, len(X_test_df))
    Xs = X_test_df.sample(n=n, random_state=RANDOM_STATE) if n > 0 else X_test_df

    sv = explainer(Xs)
    plt.figure()
    try:
        shap.summary_plot(sv.values, Xs, show=False)
    except Exception:
        shap.summary_plot(sv, Xs, show=False)
    plt.tight_layout()
    plt.savefig(out_summary, dpi=200, bbox_inches="tight")
    plt.close()

    if len(X_test_df) > 0:
        x0 = X_test_df.iloc[[0]]
        sv0 = explainer(x0)
        plt.figure()
        try:
            shap.plots.waterfall(sv0[0], show=False)
        except Exception:
            try:
                shap.waterfall_plot(sv0[0], show=False)
            except Exception:
                pass
        plt.tight_layout()
        plt.savefig(out_waterfall, dpi=200, bbox_inches="tight")
        plt.close()


def main() -> None:
    df = _load_input()
    df.columns = df.columns.str.strip()

    df = _coerce_target_binary(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor, _, _ = _build_preprocessor(df)

    X_train_mat = preprocessor.fit_transform(X_train_raw)
    X_test_mat = preprocessor.transform(X_test_raw)

    fnames = _feature_names(preprocessor)
    X_test_df = pd.DataFrame(X_test_mat, columns=fnames, index=X_test_raw.index)

    model = _train_model(X_train_mat, y_train)

    p = model.predict_proba(X_test_mat)[:, 1]
    y_pred = (p >= 0.5).astype(int)

    out = pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "p_disease": p,
        }
    )
    out.to_csv(PRED_OUT, index=False)

    acc = float(np.mean(y_pred == y_test))
    mae = float(np.mean(np.abs(y_pred - y_test)))

    print(f"""Saved predictions: {PRED_OUT}""")
    print(f"""Test accuracy: {acc:.4f}""")
    print(f"""Test MAE: {mae:.4f}""")

    cm = pd.crosstab(pd.Series(y_test, name="true"), pd.Series(y_pred, name="pred"))
    print(f"""\nConfusion matrix:\n{cm}""")

    _save_shap_outputs(model, X_test_df)

    print(f"""\nSaved SHAP plots into: {OUTPUT_DIR}""")


if __name__ == "__main__":
    main()
