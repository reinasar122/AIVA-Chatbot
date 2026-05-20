from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


SEED = 42
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 64

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR.parent / "data"
TRAIN_DATA_PATHS = [
    DATA_DIR / "ner_dataset.json",
    DATA_DIR / "ner_data.json",
]
EVAL_DATA_PATH = DATA_DIR / "ner_manual_test.json"
OUTPUT_DIR = BASE_DIR / "model" / "ner_model"
CHECKPOINT_DIR = BASE_DIR / "model"

set_seed(SEED)


def normalize_label(label):
    if label.startswith("B-ACTION-"):
        return "B-ACTION"
    if label.startswith("I-ACTION-"):
        return "I-ACTION"
    return label


def normalize_tags(tags):
    return [normalize_label(tag) for tag in tags]


def load_ner_file(path):
    df = pd.read_json(path)
    required_columns = {"tokens", "ner_tags"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {path}: {sorted(missing_columns)}")

    bad_rows = [
        index
        for index, row in df.iterrows()
        if len(row["tokens"]) != len(row["ner_tags"])
    ]
    if bad_rows:
        raise ValueError(f"Rows with mismatched tokens/ner_tags in {path}: {bad_rows[:20]}")

    df = df[["tokens", "ner_tags"]].copy()
    df["ner_tags"] = df["ner_tags"].apply(normalize_tags)
    df["source_file"] = path.name
    return df


def drop_duplicate_rows(df, label):
    before = len(df)
    df = df.copy()
    df["_tokens_key"] = df["tokens"].apply(tuple)
    df["_tags_key"] = df["ner_tags"].apply(tuple)
    df = df.drop_duplicates(subset=["_tokens_key", "_tags_key"])
    df = df.drop(columns=["_tokens_key", "_tags_key"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Dropped {removed} exact duplicate {label} rows.")
    return df


def load_train_eval_data():
    train_frames = [load_ner_file(path) for path in TRAIN_DATA_PATHS]
    train_df = pd.concat(train_frames, ignore_index=True)
    eval_df = load_ner_file(EVAL_DATA_PATH)

    train_df = drop_duplicate_rows(train_df, "training")
    eval_df = drop_duplicate_rows(eval_df, "evaluation")

    eval_token_keys = set(eval_df["tokens"].apply(tuple))
    before = len(train_df)
    train_df = train_df[~train_df["tokens"].apply(tuple).isin(eval_token_keys)].reset_index(drop=True)
    leaked_rows = before - len(train_df)
    if leaked_rows:
        print(f"Removed {leaked_rows} training rows with token text found in manual eval.")

    return train_df, eval_df


def label_sort_key(label):
    if label == "O":
        return ("", 0)
    prefix, entity = label.split("-", 1)
    return (entity, 0 if prefix == "B" else 1)


train_df, eval_df = load_train_eval_data()

label_list = sorted(
    {label for frame in (train_df, eval_df) for tags in frame["ner_tags"] for label in tags},
    key=label_sort_key,
)
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

print(f"Training NER model with {len(train_df)} training examples.")
print(f"Evaluating on {len(eval_df)} manual holdout examples.")
print(f"NER model has {len(label_list)} labels.")
print("NER labels:", ", ".join(label_list))

train_source_counts = train_df["source_file"].value_counts().to_dict()
print("Training sources:", train_source_counts)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize_and_align_labels(example):
    tokenized_inputs = tokenizer(
        example["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    word_ids = tokenized_inputs.word_ids()
    labels = example["ner_tags"]

    label_ids = []
    previous_word_idx = None

    for word_idx in word_ids:
        if word_idx is None:
            label_ids.append(-100)
        elif word_idx != previous_word_idx:
            label_ids.append(label2id[labels[word_idx]])
        else:
            label_ids.append(-100)

        previous_word_idx = word_idx

    tokenized_inputs["labels"] = label_ids
    return tokenized_inputs


train_dataset = Dataset.from_pandas(
    train_df.drop(columns=["source_file"]),
    preserve_index=False,
)
eval_dataset = Dataset.from_pandas(
    eval_df.drop(columns=["source_file"]),
    preserve_index=False,
)

tokenized_train_dataset = train_dataset.map(
    tokenize_and_align_labels,
    remove_columns=train_dataset.column_names,
)
tokenized_eval_dataset = eval_dataset.map(
    tokenize_and_align_labels,
    remove_columns=eval_dataset.column_names,
)

train_dataset = tokenized_train_dataset.flatten_indices()
eval_dataset = tokenized_eval_dataset.flatten_indices()

model = AutoModelForTokenClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer)


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


def compute_entity_metrics(predictions, true_labels):
    predicted_entities = set()
    expected_entities = set()

    for row_index, (predicted_row, true_row) in enumerate(zip(predictions, true_labels)):
        for entity in extract_entities(predicted_row):
            predicted_entities.add((row_index, *entity))
        for entity in extract_entities(true_row):
            expected_entities.add((row_index, *entity))

    true_positive = len(predicted_entities & expected_entities)
    precision = true_positive / len(predicted_entities) if predicted_entities else 0.0
    recall = true_positive / len(expected_entities) if expected_entities else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    return precision, recall, f1


def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_predictions = []
    true_labels = []
    flat_predictions = []
    flat_labels = []

    for prediction_row, label_row in zip(predictions, label_ids):
        row_predictions = []
        row_labels = []

        for prediction_id, label_id in zip(prediction_row, label_row):
            if label_id == -100:
                continue

            predicted_label = label_list[prediction_id]
            true_label = label_list[label_id]

            row_predictions.append(predicted_label)
            row_labels.append(true_label)
            flat_predictions.append(predicted_label)
            flat_labels.append(true_label)

        true_predictions.append(row_predictions)
        true_labels.append(row_labels)

    token_accuracy = accuracy_score(flat_labels, flat_predictions)

    entity_labels = [label for label in label_list if label != "O"]
    non_o_precision, non_o_recall, non_o_f1, _ = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        labels=entity_labels,
        average="micro",
        zero_division=0,
    )
    macro_f1 = precision_recall_fscore_support(
        flat_labels,
        flat_predictions,
        labels=entity_labels,
        average="macro",
        zero_division=0,
    )[2]
    entity_precision, entity_recall, entity_f1 = compute_entity_metrics(
        true_predictions,
        true_labels,
    )

    return {
        "token_accuracy": token_accuracy,
        "non_o_precision": non_o_precision,
        "non_o_recall": non_o_recall,
        "non_o_f1": non_o_f1,
        "macro_f1": macro_f1,
        "entity_precision": entity_precision,
        "entity_recall": entity_recall,
        "entity_f1": entity_f1,
    }


training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=25,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="entity_f1",
    greater_is_better=True,
    save_total_limit=2,
    dataloader_pin_memory=False,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
)

trainer.train()

final_metrics = trainer.evaluate()
print("Final NER evaluation:")
for key, value in sorted(final_metrics.items()):
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
