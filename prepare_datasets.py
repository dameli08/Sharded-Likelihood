"""
prepare_datasets.py

Converts the 8 CSV datasets from /home/dameli/8datasets/ into JSONL files
(one plain-text example per line) that compute_sharded_comparison_test.py
can consume directly.

Usage:
    python prepare_datasets.py                      # convert all 8 datasets
    python prepare_datasets.py --dataset mmlu_all   # convert one dataset by key
    python prepare_datasets.py --list               # list available dataset keys

Output is written to ./prepared_data/<dataset_key>.jsonl
"""

import csv
import os
import json
import re
import argparse

# ── Dataset registry ──────────────────────────────────────────────────────────
# Each entry: (short_key, csv_path)
DATASETS_DIR = "/home/dameli/8datasets"
DATASETS = {
    "mmlu_all":             os.path.join(DATASETS_DIR, "mmlu_all.csv"),
    "mmlu_cf_all":          os.path.join(DATASETS_DIR, "mmlu_cf_all.csv"),
    "mmlu_first100":        os.path.join(DATASETS_DIR, "mmlu_first100.csv"),
    "mmlu_pro_all":         os.path.join(DATASETS_DIR, "mmlu_pro_all.csv"),
    "mmlu_redux_all":       os.path.join(DATASETS_DIR, "mmlu_redux_all.csv"),
    "kazmmlu_all":          os.path.join(DATASETS_DIR, "kazmmlu_all.csv"),
    "rummlu_all":           os.path.join(DATASETS_DIR, "rummlu_all.csv"),
    "MMLU_KAZ_Translation": os.path.join(DATASETS_DIR, "MMLU_KAZ_Translation.csv"),
    "MMLU_RUS_Translation": os.path.join(DATASETS_DIR, "MMLU_RUS_Translation.csv"),
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "prepared_data")


# ── Formatters ────────────────────────────────────────────────────────────────

def _choice_lines(row, letters):
    """Build 'A) ... \n B) ...' string for available non-empty choices."""
    lines = []
    for letter in letters:
        col = f"choice_{letter}"
        val = row.get(col, "").strip()
        if val:
            lines.append(f"{letter}) {val}")
    return "\n".join(lines)


def format_standard_mmlu(row):
    """mmlu_all, mmlu_first100, mmlu_cf_all, mmlu_redux_all — 4 choices."""
    q = row["question"].strip()
    choices = _choice_lines(row, ["A", "B", "C", "D"])
    # prefer answer_letter when available; fall back to answer
    ans = row.get("answer_letter", row.get("answer", "")).strip()
    return f"Question: {q}\n{choices}\nAnswer: {ans}"


def format_kazmmlu(row):
    """kazmmlu_all — up to 5 choices (A-E), answer is a letter."""
    q = row["question"].strip()
    choices = _choice_lines(row, ["A", "B", "C", "D", "E"])
    ans = row.get("answer", "").strip()
    return f"Question: {q}\n{choices}\nAnswer: {ans}"


def format_mmlu_pro(row):
    """mmlu_pro_all — up to 10 choices (A-J), answer is a letter."""
    q = row["question"].strip()
    choices = _choice_lines(row, list("ABCDEFGHIJ"))
    ans = row.get("answer", "").strip()
    return f"Question: {q}\n{choices}\nAnswer: {ans}"


def format_rummlu(row):
    """rummlu_all — 4 choices (A-D), answer is a letter."""
    q = row["question"].strip()
    choices = _choice_lines(row, ["A", "B", "C", "D"])
    ans = row.get("answer", "").strip()
    return f"Question: {q}\n{choices}\nAnswer: {ans}"


def _clean_choices_column(raw):
    """
    The KAZ/RUS translation files store choices as a stringified numpy array,
    e.g. "['0' '4' '2' '6']".  Convert it to a readable numbered list.
    """
    # Extract quoted tokens: '...'
    tokens = re.findall(r"'([^']*)'", raw)
    if tokens:
        return "  ".join(f"{chr(65+i)}) {t}" for i, t in enumerate(tokens))
    return raw.strip()


def format_kaz_translation(row):
    """MMLU_KAZ_Translation — question (kaz), choices in single col, numeric answer."""
    q = row["question"].strip()
    choices = _clean_choices_column(row.get("таңдаулар", ""))
    ans = row.get("жауап", "").strip()
    return f"Сұрақ: {q}\n{choices}\nЖауап: {ans}"


def format_rus_translation(row):
    """MMLU_RUS_Translation — question (rus), choices in single col, numeric answer."""
    q = row["question"].strip()
    choices = _clean_choices_column(row.get("выбор", ""))
    ans = row.get("отвечать", "").strip()
    return f"Вопрос: {q}\n{choices}\nОтвет: {ans}"


def detect_formatter(columns):
    """
    Auto-select formatter based on column names present in the CSV header.
    Returns a callable (row -> str).
    """
    cols = set(columns)
    if "таңдаулар" in cols:
        return format_kaz_translation
    if "выбор" in cols:
        return format_rus_translation
    if "choice_F" in cols or "choice_J" in cols:
        return format_mmlu_pro
    if "choice_E" in cols:
        return format_kazmmlu
    if "domain" in cols:          # rummlu has domain column
        return format_rummlu
    # default: standard MMLU (handles mmlu_all, mmlu_cf_all, mmlu_first100, mmlu_redux)
    return format_standard_mmlu


# ── Conversion ────────────────────────────────────────────────────────────────

def convert_csv_to_jsonl(csv_path, jsonl_path, max_examples=None):
    """Read csv_path, format each row as text, write one text per line to jsonl_path."""
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)

    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        formatter = detect_formatter(reader.fieldnames or [])

        count = 0
        with open(jsonl_path, "w", encoding="utf-8") as out:
            for row in reader:
                text = formatter(row).strip()
                if not text:
                    continue
                out.write(json.dumps(text, ensure_ascii=False) + "\n")
                count += 1
                if max_examples and count >= max_examples:
                    break

    return count


def convert_all(max_examples=None):
    for key, csv_path in DATASETS.items():
        jsonl_path = os.path.join(OUTPUT_DIR, f"{key}.jsonl")
        if not os.path.exists(csv_path):
            print(f"[SKIP] {key}: file not found at {csv_path}")
            continue
        n = convert_csv_to_jsonl(csv_path, jsonl_path, max_examples=max_examples)
        print(f"[OK]   {key}: {n} examples → {jsonl_path}")


def convert_one(dataset_key, max_examples=None):
    if dataset_key not in DATASETS:
        print(f"Unknown dataset key '{dataset_key}'. Use --list to see options.")
        return
    csv_path = DATASETS[dataset_key]
    jsonl_path = os.path.join(OUTPUT_DIR, f"{dataset_key}.jsonl")
    if not os.path.exists(csv_path):
        print(f"[SKIP] file not found at {csv_path}")
        return
    n = convert_csv_to_jsonl(csv_path, jsonl_path, max_examples=max_examples)
    print(f"[OK]   {dataset_key}: {n} examples → {jsonl_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Convert 8 CSV datasets to JSONL for Sharded Likelihood.")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Convert only this dataset key (omit to convert all).")
    parser.add_argument("--list", action="store_true",
                        help="List available dataset keys and exit.")
    parser.add_argument("--max_examples", type=int, default=None,
                        help="Cap number of examples per dataset (useful for quick tests).")
    args = parser.parse_args()

    if args.list:
        print("Available dataset keys:")
        for k, p in DATASETS.items():
            exists = "✓" if os.path.exists(p) else "✗"
            print(f"  {exists}  {k:30s}  {p}")
        return

    if args.dataset:
        convert_one(args.dataset, max_examples=args.max_examples)
    else:
        convert_all(max_examples=args.max_examples)


if __name__ == "__main__":
    main()
