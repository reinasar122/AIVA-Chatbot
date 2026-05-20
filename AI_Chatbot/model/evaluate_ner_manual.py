from pathlib import Path
import json

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForTokenClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model" / "ner_model"
DATA_PATH = BASE_DIR.parent / "data" / "ner_manual_test.json"
MAX_LENGTH = 64


def load_manual_data():
    with DATA_PATH.open(encoding="utf-8") as file:
        rows = json.load(file)

    for index, row in enumerate(rows):
        missing_columns = {"tokens", "ner_tags"} - set(row)
        if missing_columns:
            raise ValueError(f"Row {index} is missing: {sorted(missing_columns)}")
        if len(row["tokens"]) != len(row["ner_tags"]):
            raise ValueError(f"Row {index} has mismatched tokens and ner_tags.")

    return rows


def extract_entities(labels):
    entities = []
    start = None
    entity_type = None

    for index, label in enumerate(labels + ["O"]):
        if label == "O":
            if entity_type is not None:
                entities.append((entity_type, start, index - 1))
                start = None
                entity_type = None
            continue

        prefix, current_type = label.split("-", 1)
        starts_new_entity = prefix == "B" or entity_type != current_type

        if starts_new_entity:
            if entity_type is not None:
                entities.append((entity_type, start, index - 1))
            start = index
            entity_type = current_type

    return entities


def compute_entity_metrics(predictions, gold_labels):
    predicted_entities = set()
    expected_entities = set()

    for row_index, (predicted_row, gold_row) in enumerate(zip(predictions, gold_labels)):
        for entity in extract_entities(predicted_row):
            predicted_entities.add((row_index, *entity))
        for entity in extract_entities(gold_row):
            expected_entities.add((row_index, *entity))

    true_positive = len(predicted_entities & expected_entities)
    precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positive / len(expected_entities) if expected_entities else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return precision, recall, f1


def predict_word_labels(model, tokenizer, tokens, id2label):
    encoded = tokenizer(
        tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    token_predictions = np.argmax(logits.numpy(), axis=-1)[0]
    word_ids = encoded.word_ids(batch_index=0)
    predictions = []
    previous_word_id = None

    for token_prediction, word_id in zip(token_predictions, word_ids):
        if word_id is None or word_id == previous_word_id:
            previous_word_id = word_id
            continue

        predictions.append(id2label[int(token_prediction)])
        previous_word_id = word_id

    return predictions


def main():
    rows = load_manual_data()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    model.eval()

    id2label = model.config.id2label
    all_predictions = []
    all_gold_labels = []
    flat_predictions = []
    flat_gold_labels = []

    for row in rows:
        predictions = predict_word_labels(model, tokenizer, row["tokens"], id2label)
        gold_labels = row["ner_tags"]

        all_predictions.append(predictions)
        all_gold_labels.append(gold_labels)
        flat_predictions.extend(predictions)
        flat_gold_labels.extend(gold_labels)

    labels = sorted(
        {label for label in flat_gold_labels + flat_predictions if label != "O"}
    )
    non_o_precision, non_o_recall, non_o_f1, _ = precision_recall_fscore_support(
        flat_gold_labels,
        flat_predictions,
        labels=labels,
        average="micro",
        zero_division=0,
    )
    macro_f1 = precision_recall_fscore_support(
        flat_gold_labels,
        flat_predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )[2]
    entity_precision, entity_recall, entity_f1 = compute_entity_metrics(
        all_predictions,
        all_gold_labels,
    )

    print("Manual NER evaluation:")
    print(f"  examples: {len(rows)}")
    print(f"  token_accuracy: {accuracy_score(flat_gold_labels, flat_predictions):.4f}")
    print(f"  non_o_precision: {non_o_precision:.4f}")
    print(f"  non_o_recall: {non_o_recall:.4f}")
    print(f"  non_o_f1: {non_o_f1:.4f}")
    print(f"  macro_f1: {macro_f1:.4f}")
    print(f"  entity_precision: {entity_precision:.4f}")
    print(f"  entity_recall: {entity_recall:.4f}")
    print(f"  entity_f1: {entity_f1:.4f}")

    print("\nMistakes:")
    mistakes_found = False
    for row, gold_labels, predictions in zip(rows, all_gold_labels, all_predictions):
        mismatches = [
            f"{token}: gold={gold}, pred={predicted}"
            for token, gold, predicted in zip(row["tokens"], gold_labels, predictions)
            if gold != predicted
        ]

        if not mismatches:
            continue

        mistakes_found = True
        print(f"\n  Text: {' '.join(row['tokens'])}")
        for mismatch in mismatches:
            print(f"    {mismatch}")

    if not mistakes_found:
        print("  None")


if __name__ == "__main__":
    main()
