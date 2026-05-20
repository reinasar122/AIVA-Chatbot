from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# -------------------------
# 0. SETTINGS
# -------------------------
model_name = "xlm-roberta-base"

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# -------------------------
# 1. LOAD DATA
# -------------------------
df = pd.read_json("../data/intents.json", orient='records')

# -------------------------
# 2. CREATE LABELS
# -------------------------
labels = df["intent"].unique().tolist()
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}

df["labels"] = df["intent"].map(label2id)

# -------------------------
# 3. SPLIT DATASET
# -------------------------
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.2)

# -------------------------
# 4. TOKENIZER (FIXED)
# -------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(example):
    return tokenizer(
        example["text"],     # ✅ FIXED (not tokens)
        padding=True,
        truncation=True,
        max_length=128
    )

dataset = dataset.map(tokenize_function)

# Remove unused columns
dataset = dataset.remove_columns(["text", "intent"])
dataset.set_format("torch")

# -------------------------
# 5. DATA COLLATOR (FIXED POSITION)
# -------------------------
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# -------------------------
# 6. METRICS
# -------------------------
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )

    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

# -------------------------
# 7. MODEL
# -------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# -------------------------
# 8. TRAINING SETTINGS
# -------------------------
training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=5,
    per_device_train_batch_size=8,
    learning_rate=2e-5,
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    save_total_limit=2,
    logging_dir="./logs",
    dataloader_pin_memory=False   # optional fix for warning
)

# -------------------------
# 9. TRAINER (FIXED)
# -------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    compute_metrics=compute_metrics,
    data_collator=data_collator   # ✅ FIXED comma
)

# -------------------------
# 10. TRAIN
# -------------------------
trainer.train()

# -------------------------
# 11. SAVE MODEL
# -------------------------
trainer.save_model("./results")
tokenizer.save_pretrained("./results")

print("Training complete!")