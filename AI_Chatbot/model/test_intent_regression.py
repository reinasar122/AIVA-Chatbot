from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.nn.functional import softmax

MODEL_DIR = Path(__file__).resolve().parent / "intent_model"

phrases = [
    {"text": "cancel my order", "expected": "cancel_order"},
    {"text": "please stop my order", "expected": "cancel_order"},
    {"text": "add 2 more Coke", "expected": "add_quantity"},
    {"text": "please add one more bottle", "expected": "add_quantity"},
    {"text": "where is my order", "expected": "order_status"},
    {"text": "what is the status of my order", "expected": "order_status"},
    {"text": "how much is Coke", "expected": "ask_price"},
    {"text": "how much for Sprite", "expected": "ask_price"},
    {"text": "do you have Sprite", "expected": "check_availability"},
    {"text": "is Coke available", "expected": "check_availability"},
    {"text": "do you accept GCash", "expected": "payment_method"},
    {"text": "can I pay with card", "expected": "payment_method"},
    {"text": "maya pwede?", "expected": "payment_method"},
    {"text": "deliver to my house in Talamban", "expected": "provide_address"},
    {"text": "Brgy. Palanas lang ihatud", "expected": "provide_address"},
    {"text": "my phone number is 09381234567", "expected": "provide_contact"},
    {"text": "what is the weather today", "expected": "out_of_scope"},
    {"text": "how long does delivery take", "expected": "delivery_time"},
    {"text": "pila ka oras ang delivery", "expected": "delivery_time"},
    {"text": "where is my order now", "expected": "order_status"},
    {"text": "has my order been shipped", "expected": "order_status"},
    {"text": "what products do you sell", "expected": "business_info"},
    {"text": "what drinks do you have", "expected": "business_info"},
    {"text": "is Sprite in stock", "expected": "check_availability"},
    {"text": "naa moy Coke", "expected": "check_availability"},
    {"text": "I want to order two Coke", "expected": "order_item"},
    {"text": "pa order ug Sprite", "expected": "order_item"},
    {"text": "add two more Coke to my order", "expected": "add_quantity"},
    {"text": "make it three bottles", "expected": "add_quantity"},
    {"text": "how much is one Coke", "expected": "ask_price"},
    {"text": "tagpila ang Sprite", "expected": "ask_price"},
    {"text": "can I pay with GCash", "expected": "payment_method"},
    {"text": "is cash on delivery accepted", "expected": "payment_method"},
]


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = softmax(logits, dim=-1)[0].tolist()
    pred = int(torch.argmax(logits, dim=-1).item())
    label = model.config.id2label[pred]
    score = probs[pred]
    return label, score


def main():
    tokenizer, model = load_model()
    print("Regression test phrases:\n")
    correct = 0
    for item in phrases:
        label, score = predict(item["text"], tokenizer, model)
        ok = label == item["expected"]
        if ok:
            correct += 1
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {item['text']}")
        print(f"  expected: {item['expected']}")
        print(f"  predicted: {label} ({score:.3f})\n")
    print(f"Summary: {correct}/{len(phrases)} correct")


if __name__ == "__main__":
    main()
