import pandas as pd
from pathlib import Path

INPUT_CSV = "processed/heart_combined_processed.csv"

OUTPUT_DIR = Path("processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FILTERED_OUT_CSV = OUTPUT_DIR / "heart_filtered_drop_cols_and_rows.csv"

TARGET_COL = "num"
# These values represent the threshold of how many values could be missing from a row
# or column to completely drop it.
# The lower the COL number the more columns it will drop but it will also remove less rows
# since they wont have so many null values
# The lower the ROW number the more rows it will drop. By putting it to 1 you will drop everything with a null
# By playing around I found 0.5 and 0.3 to be good numbers. It drops around 70 rows but still keeps some unbalanced data.
MAX_COL_MISSING_RATE = 0.5
MAX_ROW_MISSING_RATE = 0.3

na_values = ["", " ", "?", "NA", "NaN", "nan", "null", "NULL", "-9", "-9.", "-9.0"]

df = pd.read_csv(INPUT_CSV, na_values=na_values, keep_default_na=True, dtype=str)
df.columns = df.columns.str.strip()
df = df.replace(r"^\s*$", pd.NA, regex=True)

for c in df.columns:
    if df[c].dtype == "object":
        df[c] = df[c].str.strip()

if TARGET_COL not in df.columns:
    raise ValueError(f"""Missing expected column '{TARGET_COL}'. Found columns: {list(df.columns)}""")

for c in df.columns:
    df[c] = pd.to_numeric(df[c], errors="coerce")

before_rows = len(df)

labeled_df = df[df[TARGET_COL].notna()].copy()
dropped_missing_target = before_rows - len(labeled_df)

empty_cols = labeled_df.columns[labeled_df.isna().all()].tolist()

missing_rate = labeled_df.isna().mean()
gt50_cols = missing_rate[missing_rate > MAX_COL_MISSING_RATE].index.tolist()

cols_to_drop = sorted(set(empty_cols + gt50_cols))
filtered_df = labeled_df.drop(columns=cols_to_drop)

feature_cols = [c for c in filtered_df.columns if c != TARGET_COL]
row_missing_rate = (
    filtered_df[feature_cols].isna().mean(axis=1)
    if feature_cols
    else pd.Series([0] * len(filtered_df), index=filtered_df.index)
)

before_row_drop = len(filtered_df)
filtered_df = filtered_df[row_missing_rate <= MAX_ROW_MISSING_RATE].copy()
dropped_sparse_rows = before_row_drop - len(filtered_df)

filtered_df.to_csv(FILTERED_OUT_CSV, index=False)

print(f"""Input rows: {before_rows}""")
print(f"""Dropped rows (missing target '{TARGET_COL}'): {dropped_missing_target}""")
print(f"""Rows after target filter: {len(labeled_df)}""")

print(f"""Dropped columns (100% missing among labeled): {sorted(empty_cols)}""")
print(f"""Dropped columns (> {MAX_COL_MISSING_RATE:.0%} missing among labeled): {sorted(set(gt50_cols))}""")
print(f"""Total dropped columns: {len(cols_to_drop)}""")

print(f"""Dropped rows (>{MAX_ROW_MISSING_RATE:.0%} missing among features): {dropped_sparse_rows}""")
print(f"""Final rows: {len(filtered_df)}""")
print(f"""Final cols: {len(filtered_df.columns)}""")
print(f"""Saved: {FILTERED_OUT_CSV}""")
