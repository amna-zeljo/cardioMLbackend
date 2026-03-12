from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_CSV_CANDIDATES = [
    Path("processed") / "heart_filtered_drop_cols_and_rows.csv",
    Path("processed") / "heart_filtered_drop_cols_and_rows",
]

OUT_DIR = Path("processed") / "linear-feature"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "num"
RANDOM_STATE = 42


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


def _plot_top_bars(names: np.ndarray, values: np.ndarray, title: str, out_path: Path, top_n: int = 30) -> None:
    order = np.argsort(np.abs(values))[::-1]
    order = order[: min(top_n, len(order))]
    n = names[order][::-1]
    v = values[order][::-1]
    plt.figure(figsize=(10, 8))
    plt.barh(n, v)
    plt.xlabel("importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


def main() -> None:
    df = _load_input()
    df.columns = df.columns.str.strip()
    df = _coerce_target(df)

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].copy()
    y = df[TARGET_COL].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    cat_cols = [c for c in X.columns if (X[c].dtype == "object") or str(X[c].dtype).startswith("category") or (X[c].dtype == "bool")]
    num_cols = [c for c in X.columns if c not in set(cat_cols)]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scaler", StandardScaler()),
                    ]
                ),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                cat_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    model = LinearRegression()

    pipe = Pipeline(
        steps=[
            ("pre", preprocessor),
            ("model", model),
        ]
    )

    pipe.fit(X_train, y_train)

    y_pred_cont = pipe.predict(X_test)
    y_pred = np.clip(np.rint(y_pred_cont), 0, 4).astype(int)

    acc = float(np.mean(y_pred == y_test))
    mae = float(mean_absolute_error(y_test, y_pred))

    pred_out = OUT_DIR / "linear_predictions_test.csv"
    pd.DataFrame(
        {
            "y_true": y_test,
            "y_pred": y_pred,
            "y_pred_continuous": y_pred_cont,
        }
    ).to_csv(pred_out, index=False)

    X_train_mat = pipe.named_steps["pre"].transform(X_train)
    feature_names = pipe.named_steps["pre"].get_feature_names_out()
    coefs = pipe.named_steps["model"].coef_.astype(float)

    coef_abs = np.abs(coefs)
    coef_out = OUT_DIR / "linear_coefficients.csv"
    pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs,
            "abs_coef": coef_abs,
        }
    ).sort_values("abs_coef", ascending=False).to_csv(coef_out, index=False)

    _plot_top_bars(
        feature_names.astype(str),
        coef_abs,
        f"Linear model feature importance (|coef|) | acc={acc:.4f}, MAE={mae:.4f}",
        OUT_DIR / "coef_importance_top30.png",
        top_n=30,
    )

    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        scoring="neg_mean_absolute_error",
    )

    perm_names = np.array(list(X_test.columns), dtype=str)
    perm_vals = perm.importances_mean.astype(float)

    perm_out = OUT_DIR / "permutation_importance.csv"
    pd.DataFrame(
        {
            "feature": perm_names,
            "importance_mean": perm_vals,
            "importance_std": perm.importances_std.astype(float),
        }
    ).sort_values("importance_mean", ascending=False).to_csv(perm_out, index=False)

    _plot_top_bars(
        perm_names,
        perm_vals,
        "Permutation importance (neg MAE) on test set",
        OUT_DIR / "permutation_importance_top30.png",
        top_n=30,
    )

    cm = pd.crosstab(pd.Series(y_test, name="true"), pd.Series(y_pred, name="pred"))
    cm_out = OUT_DIR / "confusion_matrix.csv"
    cm.to_csv(cm_out)

    report_out = OUT_DIR / "metrics.txt"
    report_out.write_text(
        f"""test_accuracy={acc:.6f}
test_mae={mae:.6f}
predictions_csv={pred_out}
coefficients_csv={coef_out}
permutation_importance_csv={perm_out}
confusion_matrix_csv={cm_out}
""",
        encoding="utf-8",
    )

    try:
        import shap

        X_test_mat = pipe.named_steps["pre"].transform(X_test)
        if not isinstance(X_test_mat, np.ndarray):
            X_test_mat = X_test_mat.toarray()

        background = X_train_mat
        if not isinstance(background, np.ndarray):
            background = background.toarray()

        n_bg = min(300, background.shape[0])
        n_test = min(500, X_test_mat.shape[0])

        rng = np.random.default_rng(RANDOM_STATE)
        bg_idx = rng.choice(background.shape[0], size=n_bg, replace=False) if background.shape[0] > n_bg else np.arange(background.shape[0])
        te_idx = rng.choice(X_test_mat.shape[0], size=n_test, replace=False) if X_test_mat.shape[0] > n_test else np.arange(X_test_mat.shape[0])

        explainer = shap.LinearExplainer(pipe.named_steps["model"], background[bg_idx], feature_names=feature_names)
        sv = explainer(X_test_mat[te_idx])

        plt.figure()
        shap.summary_plot(sv.values, pd.DataFrame(X_test_mat[te_idx], columns=feature_names), show=False)
        plt.tight_layout()
        plt.savefig(OUT_DIR / "shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

    except Exception:
        pass

    print(f"""Saved outputs to: {OUT_DIR}""")
    print(f"""Test accuracy: {acc:.4f}""")
    print(f"""Test MAE: {mae:.4f}""")


if __name__ == "__main__":
    main()
