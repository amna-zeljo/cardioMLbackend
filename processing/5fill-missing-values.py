from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    import xgboost as xgb
except ImportError as e:
    raise ImportError(
        f"""xgboost is not installed. Install it with:
pip install xgboost
Original error: {e}"""
    )


INPUT_CSV = Path("processed") / "heart_filtered_drop_cols_and_rows.csv"

OUTPUT_DIR = Path("processed/xgboost-fill-missing")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUT_IMPUTED = OUTPUT_DIR / "heart_imputed_xgb.csv"
OUT_EXPL = OUTPUT_DIR / "imputation_explanations_shap.csv"
OUT_SUMMARY = OUTPUT_DIR / "imputation_shap_summary.csv"

TARGET_COL = "num"

RANDOM_STATE = 42
TOP_K = 8

MISSING_TOKENS = ["", " ", "?", "NA", "NaN", "nan", "null", "NULL", "-9", "-9.", "-9.0"]


def _is_integer_like(series: pd.Series) -> bool:
    s = series.dropna()
    if len(s) == 0:
        return False
    s_num = pd.to_numeric(s, errors="coerce")
    if s_num.isna().any():
        return False
    return bool(np.all(np.isclose(s_num.values, np.round(s_num.values))))


def _choose_task(series: pd.Series) -> str:
    s = series.dropna()
    if len(s) == 0:
        return "skip"
    uniq = pd.unique(s)
    if len(uniq) <= 20 and _is_integer_like(series):
        return "classification"
    return "regression"


def _load_df() -> pd.DataFrame:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"""Missing input: {INPUT_CSV}""")
    df = pd.read_csv(INPUT_CSV, dtype=str, keep_default_na=False)
    df.columns = df.columns.str.strip()
    for c in df.columns:
        df[c] = df[c].astype(str).str.strip()
    df = df.replace(MISSING_TOKENS, pd.NA)
    df = df.replace(r"^\s*$", pd.NA, regex=True)
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _prepare_X(df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, List[str]]:
    X = df[feature_cols].copy()
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    med = X.median(numeric_only=True)
    X = X.fillna(med)
    return X, list(X.columns)


def _align_columns(X: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    X2 = X.copy()
    for c in columns:
        if c not in X2.columns:
            X2[c] = 0.0
    extra = [c for c in X2.columns if c not in columns]
    if extra:
        X2 = X2.drop(columns=extra)
    return X2[columns]


def _fit_xgb_and_impute(
    df: pd.DataFrame,
    col: str,
    predictor_cols: List[str],
) -> Tuple[pd.Series, Optional[object], Optional[pd.DataFrame], Optional[pd.DataFrame], Optional[np.ndarray]]:
    train_mask = df[col].notna()
    pred_mask = df[col].isna()

    if int(pred_mask.sum()) == 0:
        return df[col], None, None, None, None

    y_train_raw = df.loc[train_mask, col]
    if y_train_raw.notna().sum() == 0:
        return df[col], None, None, None, None

    task = _choose_task(y_train_raw)
    if task == "skip":
        return df[col], None, None, None, None

    if task == "classification":
        y_train_num = pd.to_numeric(y_train_raw, errors="coerce").dropna()
        if len(y_train_num) == 0:
            return df[col], None, None, None, None

        classes = np.sort(pd.unique(y_train_num.values))
        if len(classes) == 1:
            filled = df[col].where(~pred_mask, classes[0])
            return filled, None, None, None, None

        class_to_idx = {c: i for i, c in enumerate(classes)}
        idx_to_class = {i: c for c, i in class_to_idx.items()}
        y_train = y_train_num.map(class_to_idx).astype(int).values

        X_train, cols = _prepare_X(df.loc[train_mask], predictor_cols)
        X_pred, _ = _prepare_X(df.loc[pred_mask], predictor_cols)
        X_pred = _align_columns(X_pred, cols)

        if len(classes) == 2:
            model = xgb.XGBClassifier(
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
            model.fit(X_train.values, y_train)
            probs = model.predict_proba(X_pred.values)[:, 1]
            pred_idx = (probs >= 0.5).astype(int)
        else:
            model = xgb.XGBClassifier(
                n_estimators=700,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob",
                num_class=int(len(classes)),
                eval_metric="mlogloss",
                tree_method="hist",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
            model.fit(X_train.values, y_train)
            probs = model.predict_proba(X_pred.values)
            pred_idx = np.argmax(probs, axis=1)

        preds = pd.Series([idx_to_class[int(i)] for i in pred_idx], index=df.index[pred_mask])
        filled = df[col].where(~pred_mask, preds)
        return filled, model, X_train, X_pred, pred_idx

    y_train_num = pd.to_numeric(y_train_raw, errors="coerce").dropna()
    if len(y_train_num) == 0:
        return df[col], None, None, None, None

    X_train, cols = _prepare_X(df.loc[train_mask], predictor_cols)
    X_pred, _ = _prepare_X(df.loc[pred_mask], predictor_cols)
    X_pred = _align_columns(X_pred, cols)

    model = xgb.XGBRegressor(
        n_estimators=900,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train.values, y_train_num.values)
    preds = model.predict(X_pred.values)

    if _is_integer_like(y_train_raw):
        preds = np.round(preds)

    preds_s = pd.Series(preds, index=df.index[pred_mask])
    filled = df[col].where(~pred_mask, preds_s)
    return filled, model, X_train, X_pred, None


def _shap_explain_imputations(
    col: str,
    model: object,
    X_pred: pd.DataFrame,
    imputed_values: pd.Series,
    pred_class_idx: Optional[np.ndarray],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if X_pred is None or len(X_pred) == 0:
        explained = pd.DataFrame(
            columns=["row_index", "column_imputed", "imputed_value"]
            + [f"top_feature_{i+1}" for i in range(TOP_K)]
            + [f"top_shap_{i+1}" for i in range(TOP_K)]
        )
        summary = pd.DataFrame(columns=["column_imputed", "feature", "mean_abs_shap_on_imputed_rows"])
        return explained, summary

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_pred)

    feature_names = list(X_pred.columns)
    values = sv.values if hasattr(sv, "values") else np.asarray(sv)

    if values.ndim == 3:
        if pred_class_idx is None:
            shap_matrix = values[:, 0, :]
        else:
            shap_matrix = np.vstack([values[i, int(pred_class_idx[i]), :] for i in range(values.shape[0])])
    else:
        shap_matrix = values

    shap_matrix = np.asarray(shap_matrix)
    if shap_matrix.ndim == 1:
        shap_matrix = shap_matrix.reshape(1, -1)

    if shap_matrix.shape[1] != len(feature_names):
        m = min(shap_matrix.shape[1], len(feature_names))
        shap_matrix = shap_matrix[:, :m]
        feature_names = feature_names[:m]

    abs_mean = np.mean(np.abs(shap_matrix), axis=0)
    abs_mean = np.asarray(abs_mean).reshape(-1)
    m = min(len(abs_mean), len(feature_names))
    abs_mean = abs_mean[:m]
    feature_names = feature_names[:m]

    summary = pd.DataFrame(
        {
            "column_imputed": [col] * len(feature_names),
            "feature": feature_names,
            "mean_abs_shap_on_imputed_rows": abs_mean,
        }
    ).sort_values("mean_abs_shap_on_imputed_rows", ascending=False)

    rows = []
    for i, row_idx in enumerate(X_pred.index):
        contrib = shap_matrix[i, :]
        top_idx = np.argsort(np.abs(contrib))[::-1][:TOP_K]
        top_feats = [feature_names[int(j)] for j in top_idx]
        top_vals = [float(contrib[int(j)]) for j in top_idx]
        payload: Dict[str, Any] = {
            "row_index": int(row_idx) if isinstance(row_idx, (int, np.integer)) else str(row_idx),
            "column_imputed": col,
            "imputed_value": float(imputed_values.loc[row_idx]) if pd.notna(imputed_values.loc[row_idx]) else np.nan,
        }
        for t in range(TOP_K):
            payload[f"top_feature_{t+1}"] = top_feats[t] if t < len(top_feats) else ""
            payload[f"top_shap_{t+1}"] = top_vals[t] if t < len(top_vals) else np.nan
        rows.append(payload)

    explained = pd.DataFrame(rows)
    return explained, summary


def _empty_expl_df() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["row_index", "column_imputed", "imputed_value"]
        + [f"top_feature_{i+1}" for i in range(TOP_K)]
        + [f"top_shap_{i+1}" for i in range(TOP_K)]
    )


def _empty_summary_df() -> pd.DataFrame:
    return pd.DataFrame(columns=["column_imputed", "feature", "mean_abs_shap_on_imputed_rows"])


def main() -> None:
    df = _load_df()

    if TARGET_COL not in df.columns:
        raise ValueError(f"""Missing target column '{TARGET_COL}'.""")

    feature_cols_all = [c for c in df.columns if c not in {TARGET_COL}]
    imputable_cols = [c for c in feature_cols_all if df[c].isna().any()]

    all_expl = []
    all_summary = []
    fill_report = []

    total_filled = 0

    for col in imputable_cols:
        before_missing = int(df[col].isna().sum())
        predictor_cols = [c for c in feature_cols_all if c not in {col}]
        filled, model, X_train, X_pred, pred_class_idx = _fit_xgb_and_impute(df, col, predictor_cols)
        df[col] = filled
        after_missing = int(df[col].isna().sum())

        filled_count = before_missing - after_missing
        total_filled += max(0, int(filled_count))
        fill_report.append((col, before_missing, after_missing, filled_count))

        if model is not None and X_pred is not None and filled_count > 0:
            imputed_values = df.loc[X_pred.index, col]
            explained, summary = _shap_explain_imputations(col, model, X_pred, imputed_values, pred_class_idx)
            all_expl.append(explained)
            all_summary.append(summary)

    df.to_csv(OUT_IMPUTED, index=False)

    if total_filled == 0:
        expl_df = df.copy()
        expl_df.insert(0, "row_index", expl_df.index.astype(int))
        expl_df.insert(1, "column_imputed", "")
        expl_df.insert(2, "imputed_value", np.nan)
        for i in range(TOP_K):
            expl_df[f"top_feature_{i+1}"] = ""
            expl_df[f"top_shap_{i+1}"] = np.nan
        sum_df = _empty_summary_df()
    else:
        expl_df = pd.concat(all_expl, ignore_index=True) if len(all_expl) > 0 else _empty_expl_df()
        sum_df = pd.concat(all_summary, ignore_index=True) if len(all_summary) > 0 else _empty_summary_df()

    expl_df.to_csv(OUT_EXPL, index=False)
    sum_df.to_csv(OUT_SUMMARY, index=False)

    print(f"""Saved imputed dataset: {OUT_IMPUTED}""")
    print(f"""Saved per-imputed-cell SHAP explanations: {OUT_EXPL}""")
    print(f"""Saved per-column SHAP summary: {OUT_SUMMARY}""")

    print(f"""\nFill report (col: missing_before -> missing_after, filled):""")
    for col, b, a, f in sorted(fill_report, key=lambda x: x[3], reverse=True):
        print(f"""{col}: {b} -> {a} (filled {f})""")

    remaining = df.isna().sum().sort_values(ascending=False)
    remaining = remaining[remaining > 0]
    if len(remaining) > 0:
        print(f"""\nRemaining missing counts (after imputation):""")
        for c, v in remaining.items():
            print(f"""{c}: {int(v)}""")
    else:
        print(f"""\nNo missing values remain after imputation.""")


if __name__ == "__main__":
    main()
