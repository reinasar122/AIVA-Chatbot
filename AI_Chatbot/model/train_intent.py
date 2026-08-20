from pathlib import Path

import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


SEED = 42
MODEL_NAME = "xlm-roberta-base"
MAX_LENGTH = 64

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "intent.json"
OUTPUT_DIR = BASE_DIR / "intent_model"

set_seed(SEED)


def load_intent_data():
    df = pd.read_json(DATA_PATH)

    required_columns = {"text", "intent"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in {DATA_PATH}: {sorted(missing_columns)}")

    df = df.dropna(subset=["text", "intent"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["intent"] = df["intent"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["intent"] != "")]

    conflicting_counts = df.groupby("text")["intent"].nunique()
    conflicting_texts = conflicting_counts[conflicting_counts > 1].index.tolist()
    if conflicting_texts:
        print("Dropping conflicting duplicate texts:")
        for text in conflicting_texts:
            intents = sorted(df.loc[df["text"] == text, "intent"].unique())
            print(f"  - {text!r}: {intents}")
        df = df[~df["text"].isin(conflicting_texts)].copy()

    before = len(df)
    df = df.drop_duplicates(subset=["text", "intent"]).reset_index(drop=True)
    removed = before - len(df)
    if removed:
        print(f"Dropped {removed} exact duplicate training rows.")

    return df


df = load_intent_data()

labels = sorted(df["intent"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

print(f"Training intent model with {len(df)} examples and {len(labels)} labels.")

train_df, eval_df = train_test_split(
    df,
    test_size=0.2,
    random_state=SEED,
    stratify=df["intent"],
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_LENGTH,
    )


def encode_labels(batch):
    return {"labels": [label2id[intent] for intent in batch["intent"]]}


def prepare_dataset(frame):
    dataset = Dataset.from_pandas(frame, preserve_index=False)
    dataset = dataset.map(tokenize, batched=True)
    dataset = dataset.map(encode_labels, batched=True)
    dataset = dataset.remove_columns(["text", "intent"])
    dataset.set_format("torch")
    return dataset


train_dataset = prepare_dataset(train_df)
eval_dataset = prepare_dataset(eval_df)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


def compute_metrics(eval_pred):
    logits, true_labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(true_labels, predictions)
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true_labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "weighted_precision": weighted_precision,
        "weighted_recall": weighted_recall,
        "weighted_f1": weighted_f1,
    }


training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=8,
    weight_decay=0.01,
    warmup_steps=50,
    logging_steps=10,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
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
print("Final intent evaluation:")
for key, value in sorted(final_metrics.items()):
    if isinstance(value, float):
        print(f"  {key}: {value:.4f}")
    else:
        print(f"  {key}: {value}")

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))
