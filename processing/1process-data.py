from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


DATASET_DIR = Path("dataset")
INPUT_FILES = [
    "cleveland.cleaned.data",
    "hungarian.data",
    #"long-beach-va.data",
    #"switzerland.data",

    #"mixed.data",
]
OUTPUT_CSV = Path("processed") / "heart_combined_processed.csv"


FIELDS_TO_EXTRACT = [
    ("age", 3),
    ("sex", 4),
    ("cp", 9),
    ("trestbps", 10),
    ("chol", 12),
    ("fbs", 16),
    ("restecg", 19),
    ("exang", 38),
    ("thalach", 32),
    ("oldpeak", 40),
    ("slope", 41),
    ("num", 58),

    #("painloc", 5),
    #("painexer", 6),
    #("relrest", 7),
    #("pncaden", 8),

    #("htn", 11),
    #("smoke", 13),
    #("cigs", 14),
    #("years_smoking", 15),
    #("dm", 17),
    #("famhist", 18),

    #("proto", 28),
    #("thaldur", 29),
    #("met", 31),
    #("tpeakbps", 34),
    #("tpeakbpd", 35),
    #("ca", 44),
    #("thal", 51),
]

"""
FIELDS_TO_EXTRACT = [
    ("age", 1),
    ("sex", 2),
    ("cp", 3),
    ("trestbps", 4),
    ("chol", 5),
    ("fbs", 6),
    ("restecg", 7),
    ("thalach", 8),
    ("exang", 9),
    ("oldpeak", 10),
    ("slope", 11),
    ("num", 12),

]
"""


ROW_LENGTH = 76
#ROW_LENGTH = 12 # For fixed cleveland


def _normalize_source_name(filename: str) -> str:
    name = filename
    if name.endswith(".data"):
        name = name[: -len(".data")]
    return (
        name.replace(" ", "_")
        .replace("-", "_")
        .replace(".", "_")
        .lower()
    )


def _parse_value(token: str) -> Optional[Any]:
    t = token.strip()
    if t == "" or t in {"-9", "-9.0", "?", "NA", "NaN", "nan", "null", "NULL"}:
        return None
    try:
        if any(ch in t for ch in [".", "e", "E"]):
            return float(t)
        return int(t)
    except ValueError:
        return t


def _read_tokens(path: Path) -> List[str]:
    text = path.read_text(encoding="utf-8", errors="replace")
    text = text.replace(",", " ")
    tokens: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        tokens.extend(parts)
    return tokens


def _records_from_tokens(tokens: List[str], record_len: int = 76) -> List[List[str]]:
    total = len(tokens)
    n = total // record_len
    records = [tokens[i * record_len : (i + 1) * record_len] for i in range(n)]
    return records


def _extract_fields(record_tokens: List[str]) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    for col, idx_1based in FIELDS_TO_EXTRACT:
        token = record_tokens[idx_1based - 1]
        row[col] = _parse_value(token)

    num_val = row.get("num")
    if num_val is None:
        row["num"] = None
    else:
        try:
            row["num"] = 0 if int(float(num_val)) == 0 else 1
        except (ValueError, TypeError):
            row["num"] = None

    return row


def parse_file(path: Path) -> List[Dict[str, Any]]:
    tokens = _read_tokens(path)
    records = _records_from_tokens(tokens, record_len=ROW_LENGTH)
    rows = [_extract_fields(r) for r in records]
    return rows


def main() -> None:
    all_rows: List[Dict[str, Any]] = []
    for fname in INPUT_FILES:
        path = DATASET_DIR / fname
        if not path.exists():
            raise FileNotFoundError(f"""Missing input file: {path}""")
        rows = parse_file(path)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    columns = [c for c, _ in FIELDS_TO_EXTRACT]
    df = df.reindex(columns=columns)

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)


if __name__ == "__main__":
    main()
