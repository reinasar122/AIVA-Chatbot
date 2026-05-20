import random
import json
from pathlib import Path


random.seed(42)

products = [
    ["Coke"], ["Sprite"], ["Royal"],
    ["Yakult"], ["Pepsi"], ["C2", "Apple"],
    ["Mountain", "Dew"],
    ["Coke", "Zero"], ["Royal", "Orange"],
    ["Sprite", "1.5L"],
    ["c0ke", "zero"], ["sprte"], ["pepsi"]
]

quantities = [
    ["1"], ["2"], ["3"], ["4"], ["5"],
    ["one"], ["two"], ["three"],
    ["isa"], ["duha"], ["tulo"], ["lima"],
    ["usa", "ka"], ["duha", "ka"], ["tulo", "ka"], ["lima", "ka"]
]

payments = [
    ["cash"], ["GCash"], ["PayMaya"], ["Maya"],
    ["COD"], ["Cash", "on", "delivery"]
]

locations = [
    ["Lahug"], ["IT", "Park"], ["Ayala", "Center"],
    ["SM", "Seaside"], ["Banilad"], ["Mabolo"],
    ["Poblacion"], ["Talamban"], ["Apas"],
    ["Mandaue", "City"], ["Gaisano", "Mall"], ["Parkmall"]
]

addresses = [
    ["Unit", "3B", "Tower", "2", "Banilad"],
    ["Unit", "4A", "Tower", "1", "Banilad"],
    ["Room", "12", "Dorm"],
    ["Block", "8", "Lot", "12"],
    ["Purok", "7", "Mabolo"],
    ["Purok", "3", "Talamban"]
]

phones = [
    ["09176543210"],
    ["0998-123-4567"],
    ["+639171234567"],
    ["09", "17", "123", "4567"],
    ["09", "18", "555", "1234"]
]

# ACTION CATEGORIES
actions_add = [
    ["add"], ["pun-an"], ["pun", "an"], ["insert"], ["include"], ["pun", "e"]
]

actions_modify = [
    ["change"], ["modify"], ["edit"], ["update"], ["ilisan"], ["palitan"], ["usba"]
]

actions_cancel = [
    ["cancel"], ["remove"], ["drop"], ["undone"], ["delete"]
]

actions_ship = [
    ["Deliver"], ["Ship"], ["Pa-deliver"], ["Padala"], ["Order"], ["Hatagi"], ["Mag-order"],
    ["deliver"], ["Palit"], ["Pa-order"], ["Send"], ["i-deliver"]
]

delivery_actions = [
    ["deliver"], ["i-deliver"]
]

templates = [
    ["{action}", "{qty}", "{product}"],
    ["I", "want", "{qty}", "{product}"],
    ["{action}", "{qty}", "{product}"],
    ["{action}", "kog", "{qty}", "{product}"],
    ["{action}", "{qty}", "{product}", "{payment}"],
    ["{action}", "{qty}", "{product}", "to", "{location}"],
    ["{action}", "{qty}", "{product}", "{payment}", "{location}"],
    ["{qty}", "{product}", "{payment}"],
    ["{product}", "{qty}", "{payment}"],
    ["Please", "{action}", "{qty}", "{product}"],
    ["Can", "you", "{action}", "{qty}", "{product}"],   
    ["{action}", "{product}", "{qty}"],
    ["{action}", "{product}", "{qty}", "{payment}"],
    ["{action}", "{product}", "to", "{product}"],
    ["{action}", "sa", "{location}"],
    ["{product}", "plz", "{action}", "{location}"],
    ["{payment}", "please", "for", "{qty}", "{product}"],
    ["Contact", "ko", "{phone}"],
    ["Number", "nako", "{phone}"],
    ["My", "contact", "is", "{phone}"],
    ["Sa", "{location}", "ko", "kuhaon"],
    ["Kuhaon", "nako", "sa", "{location}"],
    ["Pickup", "lang", "ko", "sa", "{location}"],
    ["Palihog", "{action}", "{qty}", "{product}", "sa", "{location}"],
    ["Please", "{action}", "{qty}", "{product}", "to", "my", "order"],
    ["{action}", "my", "order", "to", "{qty}", "{product}"],
    ["{action}", "my", "{product}", "order"],
    ["Make", "it", "{qty}", "{product}", "instead"],
    ["Diri", "sa", "{location}"],
    ["{action}", "to", "{address}"],
    ["Sa", "{location}", "lang", "{delivery_action}"],
    ["{payment}", "akong", "bayad", "for", "{qty}", "{product}"],
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
    address = random.choice(addresses)
    phone = random.choice(phones)
    
    # Choose random action category
    action_category = random.choice([actions_add, actions_modify, actions_cancel, actions_ship])
    action = random.choice(action_category)
    delivery_action = random.choice(delivery_actions)
    
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

        elif word == "{address}":
            for i, loc in enumerate(address):
                tokens.append(loc)
                tags.append("B-LOCATION" if i == 0 else "I-LOCATION")

        elif word == "{phone}":
            for i, number in enumerate(phone):
                tokens.append(number)
                tags.append("B-PHONE" if i == 0 else "I-PHONE")

        elif word == "{action}":
            for i, act in enumerate(action):
                tokens.append(act)
                tags.append("B-ACTION" if i == 0 else "I-ACTION")

        elif word == "{delivery_action}":
            for i, act in enumerate(delivery_action):
                tokens.append(act)
                tags.append("B-ACTION" if i == 0 else "I-ACTION")

        else:
            tokens.append(word)
            tags.append("O")

    return {"tokens": tokens, "ner_tags": tags}


# GENERATE DATASET
dataset = [build_sample() for _ in range(1200)]  # 🔥 1200 samples

# SAVE FILE
output_path = Path(__file__).with_name("ner_dataset.json")
with output_path.open("w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2)

print(f"Generated {len(dataset)} NER samples at {output_path}")
