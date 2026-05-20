from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

model_path = "./model/ner_model"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForTokenClassification.from_pretrained(model_path)

ner = pipeline(
    "token-classification",
    model=model,
    tokenizer=tokenizer,
    aggregation_strategy="simple"
)

# -------------------------
# PRODUCT LIST
# -------------------------
PRODUCTS = [
    "coke",
    "sprite",
    "royal",
    "yakult",
    "coke zero",
    "mountain dew",
    "pepsi",
]

# -------------------------
# PRODUCT MATCHER
# -------------------------
def find_product(text):
    text_lower = text.lower()

    for product in PRODUCTS:
        if product in text_lower:
            return product

    return None

texts = [
    "I want 3 Yakult",
    "Buy 2 Sprite using cash",
    "Order 1 Royal and pay with GCash",
    "Can I get 5 Coke Zero?",
    "Magbayad ko sa C2 Apple na akong gi order?"
]

for text in texts:
    results = ner(text)
    product = find_product(text)

    print("\nInput:", text)
    print("Detected product:", product)
    print("Predictions:")
    for item in results:
        print(item)

results = ner(text)

print("\nInput:", text)
print("\nPredictions:")
for item in results:
    print(item)