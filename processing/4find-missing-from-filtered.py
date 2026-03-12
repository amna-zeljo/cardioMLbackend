import pandas as pd

CSV_PATH = "processed/heart_filtered_drop_cols_and_rows.csv"

df = pd.read_csv(CSV_PATH)

total = len(df)

missing_counts = df.isna().sum().sort_values(ascending=False)
for col, missing in missing_counts.items():
    print(f"""{col}: {int(missing)}/{total}""")

complete_rows = int(df.notna().all(axis=1).sum())
print(f"""\nrows_with_all_fields_present: {complete_rows}/{total}""")
