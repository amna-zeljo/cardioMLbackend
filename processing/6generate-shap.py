from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path("processed/xgboost-fill-missing")

SUMMARY_CSV = BASE_DIR / "imputation_shap_summary.csv"
EXPL_CSV = BASE_DIR / "imputation_explanations_shap.csv"

CHART_DIR = BASE_DIR / "charts"
CHART_DIR.mkdir(parents=True, exist_ok=True)

TOP_COLS = 8
TOP_FEATURES = 12
LOCAL_EXAMPLES = 20
RANDOM_STATE = 42


def _safe_name(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s))[:180]


def main() -> None:
    if not SUMMARY_CSV.exists():
        raise FileNotFoundError(f"""Missing file: {SUMMARY_CSV}""")
    if not EXPL_CSV.exists():
        raise FileNotFoundError(f"""Missing file: {EXPL_CSV}""")

    summary_df = pd.read_csv(SUMMARY_CSV)
    expl_df = pd.read_csv(EXPL_CSV)

    needed_summary_cols = {"column_imputed", "feature", "mean_abs_shap_on_imputed_rows"}
    if not needed_summary_cols.issubset(set(summary_df.columns)):
        raise ValueError(f"""Summary CSV missing required columns. Found: {list(summary_df.columns)}""")

    needed_expl_cols = {"row_index", "column_imputed", "imputed_value"}
    if not needed_expl_cols.issubset(set(expl_df.columns)):
        raise ValueError(f"""Explanations CSV missing required columns. Found: {list(expl_df.columns)}""")

    counts = (
        expl_df["column_imputed"]
        .astype(str)
        .value_counts(dropna=False)
        .sort_values(ascending=False)
    )

    if len(counts) == 0:
        raise ValueError(f"""No imputation explanations found in {EXPL_CSV}""")

    plt.figure()
    counts.plot(kind="bar")
    plt.xlabel("column_imputed")
    plt.ylabel("imputed_cells_count")
    plt.title("Imputed cells per column")
    plt.tight_layout()
    out_counts = CHART_DIR / "imputed_cells_count_by_column.png"
    plt.savefig(out_counts, dpi=200, bbox_inches="tight")
    plt.close()

    top_cols = list(counts.head(TOP_COLS).index)

    for col in top_cols:
        sub = summary_df[summary_df["column_imputed"].astype(str) == str(col)].copy()
        sub["mean_abs_shap_on_imputed_rows"] = pd.to_numeric(sub["mean_abs_shap_on_imputed_rows"], errors="coerce")
        sub = sub.dropna(subset=["mean_abs_shap_on_imputed_rows"])
        if len(sub) == 0:
            continue

        sub = sub.sort_values("mean_abs_shap_on_imputed_rows", ascending=False).head(TOP_FEATURES)
        sub = sub.sort_values("mean_abs_shap_on_imputed_rows", ascending=True)

        plt.figure()
        plt.barh(sub["feature"].astype(str), sub["mean_abs_shap_on_imputed_rows"].values)
        plt.xlabel("mean(|SHAP|) on imputed rows")
        plt.ylabel("feature")
        plt.title(f"""Global drivers for imputing '{col}'""")
        plt.tight_layout()
        out_path = CHART_DIR / f"""drivers_{_safe_name(col)}.png"""
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        plt.close()

    top_feat_cols = [c for c in expl_df.columns if c.startswith("top_feature_")]
    top_shap_cols = [c for c in expl_df.columns if c.startswith("top_shap_")]

    if len(top_feat_cols) == 0 or len(top_shap_cols) == 0:
        raise ValueError(f"""No top_feature_*/top_shap_* columns found in {EXPL_CSV}""")

    def _idx(c: str) -> int:
        m = re.search(r"(\d+)$", c)
        return int(m.group(1)) if m else 10**9

    top_feat_cols = sorted(top_feat_cols, key=_idx)
    top_shap_cols = sorted(top_shap_cols, key=_idx)

    n = min(LOCAL_EXAMPLES, len(expl_df))
    sample_df = expl_df.sample(n=n, random_state=RANDOM_STATE) if n > 0 else expl_df

    local_dir = CHART_DIR / "local_examples"
    local_dir.mkdir(parents=True, exist_ok=True)

    for _, row in sample_df.iterrows():
        feats = [row.get(c, "") for c in top_feat_cols]
        vals = [row.get(c, np.nan) for c in top_shap_cols]

        pairs = []
        for f, v in zip(feats, vals):
            if pd.isna(v):
                continue
            fs = str(f).strip()
            if fs == "" or fs.lower() == "nan":
                continue
            pairs.append((fs, float(v)))

        if len(pairs) == 0:
            continue

        pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        labels = [p[0] for p in pairs][::-1]
        shap_vals = [p[1] for p in pairs][::-1]

        row_id = _safe_name(row.get("row_index", "row"))
        col = _safe_name(row.get("column_imputed", "col"))
        imp_val = row.get("imputed_value", "")
        imp_val_s = _safe_name(imp_val)

        plt.figure()
        plt.barh(labels, shap_vals)
        plt.xlabel("SHAP contribution")
        plt.ylabel("feature")
        plt.title(f"""Local imputation explanation: {col} (row {row_id}) -> {imp_val}""")
        plt.tight_layout()
        out_local = local_dir / f"""local_{col}_row_{row_id}_val_{imp_val_s}.png"""
        plt.savefig(out_local, dpi=200, bbox_inches="tight")
        plt.close()

    print(f"""Saved: {out_counts}""")
    print(f"""Saved driver charts for columns: {top_cols} -> {CHART_DIR}""")
    print(f"""Saved local example charts (sample {n}) -> {local_dir}""")


if __name__ == "__main__":
    main()
