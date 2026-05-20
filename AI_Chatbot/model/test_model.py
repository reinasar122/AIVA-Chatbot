from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

model_path = "./model/ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

id2label = model.config.id2label

text = "Order Coke 2 bottles"
tokens = text.split()

inputs = tokenizer(tokens, is_split_into_words=True, return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs)

predictions = torch.argmax(outputs.logits, dim=2)

word_ids = inputs.word_ids()

previous_word = None

for i, word_id in enumerate(word_ids):
    if word_id is None or word_id == previous_word:
        continue

    label = id2label[predictions[0][i].item()]
    print(tokens[word_id], "->", label)

    previous_word = word_id