"""Audit intent data before fine-tuning the classifier."""

from collections import Counter, defaultdict
from pathlib import Path
import json
import re


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "intent.json"


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text).strip().lower())


def load_rows():
    with DATA_PATH.open(encoding="utf-8") as file:
        rows = json.load(file)

    if not isinstance(rows, list):
        raise ValueError("intent.json must contain a JSON list.")

    for index, row in enumerate(rows):
        if not isinstance(row, dict) or {"text", "intent"} - set(row):
            raise ValueError(f"Row {index} must contain text and intent.")
        if not normalize_text(row["text"]) or not str(row["intent"]).strip():
            raise ValueError(f"Row {index} has empty text or intent.")

    return rows


def audit(rows):
    by_text = defaultdict(set)
    duplicate_rows = Counter()
    label_counts = Counter()

    for row in rows:
        text = normalize_text(row["text"])
        intent = str(row["intent"]).strip()
        by_text[text].add(intent)
        duplicate_rows[(text, intent)] += 1
        label_counts[intent] += 1

    conflicting = {
        text: sorted(intents)
        for text, intents in by_text.items()
        if len(intents) > 1
    }
    duplicates = {
        (text, intent): count
        for (text, intent), count in duplicate_rows.items()
        if count > 1
    }

    print(f"Rows: {len(rows)}")
    print(f"Unique labels: {len(label_counts)}")
    print("Examples per label:")
    for label, count in sorted(label_counts.items()):
        print(f"  {label}: {count}")
    print(f"Exact duplicate rows: {sum(count - 1 for count in duplicates.values())}")
    print(f"Conflicting texts: {len(conflicting)}")

    if duplicates:
        print("\nDuplicate examples:")
        for (text, intent), count in sorted(duplicates.items()):
            print(f"  {count}x [{intent}] {text}")

    if conflicting:
        print("\nConflicting examples:")
        for text, intents in sorted(conflicting.items()):
            print(f"  [{', '.join(intents)}] {text}")


if __name__ == "__main__":
    audit(load_rows())