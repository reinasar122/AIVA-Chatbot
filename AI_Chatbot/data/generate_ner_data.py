import random
import json

products = [
    ["Coke"], ["Sprite"], ["Royal"],
    ["Coke", "Zero"], ["Royal", "Orange"],
    ["Sprite", "1.5L"]
]

quantities = [
    ["1"], ["2"], ["3"], ["4"], ["5"],
    ["one"], ["two"], ["three"],
    ["isa"], ["duha"], ["tulo"], ["lima"]
]

payments = [
    ["cash"], ["GCash"], ["PayMaya"]
]

locations = [
    ["Lahug"], ["IT", "Park"], ["Ayala", "Center"],
    ["SM", "Seaside"], ["Banilad"], ["Mabolo"],
    ["Poblacion"], ["Talamban"], ["Apas"]
]

templates = [
    ["Order", "{qty}", "{product}"],
    ["I", "want", "{qty}", "{product}"],
    ["Palihog", "{qty}", "{product}"],
    ["Hatagi", "kog", "{qty}", "{product}"],
    ["Order", "{qty}", "{product}", "{payment}"],
    ["Deliver", "{qty}", "{product}", "to", "{location}"],
    ["Pa-deliver", "{qty}", "{product}", "{payment}", "{location}"],
    ["{qty}", "{product}", "{payment}"],
    ["{product}", "{qty}", "{payment}"],
]

misspellings = [
    ["coke"], ["cok"], ["cokee"], ["koke"],
    ["plss"], ["pls"],  
]

# randomly replace product sometimes
def build_sample():
    template = random.choice(templates)

    product = random.choice(products)
    qty = random.choice(quantities)
    payment = random.choice(payments)
    location = random.choice(locations)

    tokens = []
    tags = []

    for word in template:
        if word == "{product}":
            for i, p in enumerate(product):
                tokens.append(p)
                tags.append("B-PRODUCT" if i == 0 else "I-PRODUCT")

        elif word == "{qty}":
            for i, q in enumerate(qty):
                tokens.append(q)
                tags.append("B-QUANTITY" if i == 0 else "I-QUANTITY")

        elif word == "{payment}":
            for i, pay in enumerate(payment):
                tokens.append(pay)
                tags.append("B-PAYMENT" if i == 0 else "I-PAYMENT")

        elif word == "{location}":
            for i, loc in enumerate(location):
                tokens.append(loc)
                tags.append("B-LOCATION" if i == 0 else "I-LOCATION")

        else:
            tokens.append(word)
            tags.append("O")

    return {"tokens": tokens, "ner_tags": tags}


# GENERATE DATASET
dataset = [build_sample() for _ in range(1200)]  # 🔥 1200 samples

# SAVE FILE
with open("ner_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("✅ Generated 1200 NER samples!")