from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

model = AutoModelForSequenceClassification.from_pretrained("./intent_model")
tokenizer = AutoTokenizer.from_pretrained("./intent_model")

text = "Order Coke 2 bottles"

inputs = tokenizer(text, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

pred = torch.argmax(logits, dim=1).item()

print(model.config.id2label[pred])