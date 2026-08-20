from functools import lru_cache
from pathlib import Path
import re

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, pipeline

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"
INTENT_MODEL_DIR = MODEL_DIR / "intent_model"
NER_MODEL_DIR = MODEL_DIR / "model" / "ner_model"
EDGE_PUNCTUATION = ".,!?/\\;:()[]{}\"'"
ACTION_PHRASES = (
    ("pun", "an"),
    ("pun", "e"),
    ("pun-an",),
    ("tag", "pila"),
)
ADD_INTENT_WORDS = {
    "add",
    "dugang",
    "include",
    "insert",
    "pun-an",
    "punan",
}
ADD_INTENT_PHRASES = (
    ("pun", "an"),
    ("pun", "e"),
)
AVAILABILITY_INTENT_WORDS = {
    "availabe",
    "availble",
    "available",
    "avaliable",
    "avialable",
    "stock",
}
AVAILABILITY_INTENT_PHRASES = (
    ("naa", "mo"),
    ("naa", "moy"),
    ("naa", "pa"),
    ("naa", "pa", "ba"),
    ("naay",),
    ("do", "you", "have"),
    ("may",),
    ("meron",),
)
BUSINESS_INFO_INTENT_WORDS = {
    "baligya",
    "namaligya",
    "offer",
    "sell",
    "selling",
}
BUSINESS_INFO_INTENT_PHRASES = (
    ("do", "you", "sell"),
)
BUSINESS_HOURS_WORDS = {
    "abli",
    "abri",
    "hours",
    "magabli",
    "managbli",
    "mangabli",
    "moabli",
    "muabli",
    "open",
    "opening",
}
BUSINESS_POLICY_WORDS = {
    "bulk",
    "fee",
    "fees",
    "minimum",
}
BUSINESS_POLICY_PHRASES = (
    ("minimum", "order"),
    ("minimum", "orders"),
    ("delivery", "fee"),
    ("delivery", "fees"),
    ("bulk", "order"),
    ("bulk", "orders"),
)
ADDRESS_START_WORDS = {
    "patungo",
}
ADDRESS_START_PHRASES = (
    ("sa", "may"),
)
ADDRESS_STOP_WORDS = {
    "kuya",
    "po",
}
DELIVERY_AREA_INTENT_PHRASES = (
    ("do", "you", "deliver"),
    ("do", "you", "deliver", "to"),
    ("do", "you", "deliver", "in"),
    ("mo", "deliver"),
    ("mo", "deliver", "sa"),
)
DELIVERY_TIME_WORDS = {
    "arrive",
    "darating",
    "delivery",
    "dugay",
    "fast",
    "kabilis",
    "kanus-a",
    "katagal",
    "kaya",
    "long",
    "maabot",
    "minutes",
    "oras",
    "paspas",
    "soon",
    "take",
    "time",
    "when",
}
PRICE_INTENT_WORDS = {
    "hm",
    "magkano",
    "presyo",
    "tagpila",
}
PRICE_INTENT_PHRASES = (
    ("how", "much"),
    ("pila",),
    ("tag", "pila"),
)
PAYMENT_INTENT_WORDS = {
    "gcash",
    "paymaya",
    "maya",
    "cash",
    "cod",
    "paypal",
    "card",
    "credit",
    "debit",
    "online",
    "bank",
    "bayad",
}
PAYMENT_ACCEPT_WORDS = {
    "pwede",
    "can",
    "may",
    "accept",
    "available",
    "possible",
    "ask",
}
PAYMENT_INTENT_PHRASES = (
    ("cash", "on", "delivery"),
    ("gcash", "pwede"),
    ("paymaya", "pwede"),
    ("maya", "pwede"),
)
ADDRESS_INTENT_WORDS = {
    "deliver",
    "i-deliver",
    "ihatud",
    "hatud",
    "ihatud",
}
ADDRESS_HINT_WORDS = {
    "brgy",
    "barangay",
    "poblacion",
    "balay",
    "house",
    "home",
    "location",
    "diri",
    "dinhi",
    "direksyon",
}
ADDRESS_INTENT_PHRASES = (
    ("deliver", "sa"),
    ("deliver", "to"),
    ("i-deliver", "sa"),
    ("i-deliver", "to"),
    ("deliver", "here"),
    ("ihatud", "sa"),
    ("ihatud", "dinhi"),
)
KNOWN_PRODUCT_WORDS = {
    "apple",
    "c2",
    "coke",
    "dew",
    "gatas",
    "milk",
    "mountain",
    "pepsi",
    "royal",
    "sprite",
    "yakult",
}
MODIFY_INTENT_WORDS = {
    "change",
    "ilisi",
    "ilisan",
    "modify",
    "palitan",
    "replace",
    "switch",
    "update",
    "usba",
    "usbon",
}
NON_ENTITY_WORDS = {
    "bali",
    "tanan",
    "total",
}
LOW_CONFIDENCE_THRESHOLD = 0.25
# When the model confidence is below this threshold, return a fallback intent
# so the app can ask the user to clarify rather than incorrectly act.
OUT_OF_SCOPE_WORDS = {
    "ai",
    "boyfriend",
    "capital",
    "cryptocurrency",
    "hack",
    "hacking",
    "japan",
    "joke",
    "learning",
    "love",
    "messi",
    "nba",
    "physics",
    "president",
    "prisedint",
    "quantum",
    "song",
    "spider-man",
    "swift",
    "website",
    "wifi",
}


class RequestBody(BaseModel):
    text: str = Field(..., min_length=1, max_length=500)


@lru_cache(maxsize=1)
def get_intent_classifier():
    return pipeline(
        "text-classification",
        model=str(INTENT_MODEL_DIR),
        tokenizer=str(INTENT_MODEL_DIR),
    )


@lru_cache(maxsize=1)
def get_ner_model():
    tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(NER_MODEL_DIR)
    model.eval()
    return tokenizer, model


def split_words(text):
    words = []
    spans = []

    for match in re.finditer(r"\S+", text):
        raw_word = match.group()
        leading_trimmed = len(raw_word) - len(raw_word.lstrip(EDGE_PUNCTUATION))
        trailing_trimmed = len(raw_word.rstrip(EDGE_PUNCTUATION))
        word = raw_word[leading_trimmed:trailing_trimmed]

        if not word:
            continue

        words.append(word)
        spans.append((match.start() + leading_trimmed, match.start() + trailing_trimmed))

    return words, spans


def format_intent(prediction):
    top_prediction = prediction[0]
    return {
        "label": top_prediction["label"],
        "score": float(top_prediction["score"]),
    }


def apply_intent_overrides(text, intent):
    words = [word.lower() for word in re.findall(r"[A-Za-z-]+", text)]
    word_set = set(words)

    if is_business_info_question(words, word_set):
        return {
            "label": "business_info",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_delivery_area_question(words):
        return {
            "label": "business_info",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_business_hours_question(word_set):
        return {
            "label": "business_info",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_delivery_time_question(word_set):
        return {
            "label": "delivery_time",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_provide_address_request(words, word_set):
        return {
            "label": "provide_address",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_payment_method_question(words, word_set):
        return {
            "label": "payment_method",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_price_question(words, word_set):
        return {
            "label": "ask_price",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_business_policy_question(words, word_set):
        return {
            "label": "business_info",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if is_availability_question(words, word_set):
        return {
            "label": "check_availability",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if "cancel" in word_set and "order" in word_set:
        return {
            "label": "cancel_order",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    for phrase in ADD_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return {
                    "label": "add_quantity",
                    "score": intent["score"],
                    "source": "rule",
                    "model_label": intent["label"],
                }

    if word_set & ADD_INTENT_WORDS:
        return {
            "label": "add_quantity",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if word_set & MODIFY_INTENT_WORDS:
        return {
            "label": "modify_order",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if word_set & OUT_OF_SCOPE_WORDS:
        return {
            "label": "out_of_scope",
            "score": intent["score"],
            "source": "rule",
            "model_label": intent["label"],
        }

    if intent["score"] < LOW_CONFIDENCE_THRESHOLD:
        return {
            "label": "fallback",
            "score": intent["score"],
            "source": "low_confidence",
            "model_label": intent["label"],
        }

    return intent


def is_business_info_question(words, word_set):
    has_business_word = bool(word_set & BUSINESS_INFO_INTENT_WORDS)
    has_business_phrase = False

    for phrase in BUSINESS_INFO_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                has_business_phrase = True
                break
        if has_business_phrase:
            break

    has_catalog_question = (
        "what" in word_set
        and bool(word_set & {"drinks", "products"})
        and ("have" in word_set or "sell" in word_set)
    )

    return has_business_word or has_business_phrase or has_catalog_question


def is_business_hours_question(word_set):
    return bool(word_set & BUSINESS_HOURS_WORDS)


def is_business_policy_question(words, word_set):
    if word_set & BUSINESS_POLICY_WORDS:
        return True

    for phrase in BUSINESS_POLICY_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return True

    return False


def is_delivery_area_question(words):
    for phrase in DELIVERY_AREA_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return True

    return False


def is_delivery_time_question(word_set):
    time_words = DELIVERY_TIME_WORDS - {"delivery"}
    return bool(word_set & time_words) and (
        "deliver" in word_set
        or "delivery" in word_set
        or "arrive" in word_set
        or "maabot" in word_set
        or "darating" in word_set
    )


def is_price_question(words, word_set):
    if word_set & PRICE_INTENT_WORDS:
        return True

    for phrase in PRICE_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return True

    return False


def is_payment_method_question(words, word_set):
    if not (word_set & PAYMENT_INTENT_WORDS):
        return False

    if word_set & PAYMENT_ACCEPT_WORDS:
        return True

    for phrase in PAYMENT_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return True

    return bool(word_set & PAYMENT_INTENT_WORDS) and not bool(word_set & PRICE_INTENT_WORDS)


def is_provide_address_request(words, word_set):
    if not (word_set & ADDRESS_INTENT_WORDS):
        return False

    if word_set & ADDRESS_HINT_WORDS:
        return True

    if "deliver" in word_set or "i-deliver" in word_set or "ihatud" in word_set or "hatud" in word_set or "ihatud" in word_set:
        return not bool(word_set & KNOWN_PRODUCT_WORDS)

    return False


def is_availability_question(words, word_set):
    has_product = bool(word_set & KNOWN_PRODUCT_WORDS)
    has_availability_word = bool(word_set & AVAILABILITY_INTENT_WORDS)
    has_availability_phrase = False

    for phrase in AVAILABILITY_INTENT_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                has_availability_phrase = True
                break
        if has_availability_phrase:
            break

    return has_product and (has_availability_word or has_availability_phrase)


def predict_word_labels(text):
    words, spans = split_words(text)
    if not words:
        return [], [], []

    tokenizer, model = get_ner_model()
    encoded = tokenizer(
        words,
        is_split_into_words=True,
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**encoded).logits

    prediction_ids = torch.argmax(logits, dim=-1)[0].tolist()
    word_ids = encoded.word_ids(batch_index=0)
    id2label = model.config.id2label
    labels = []
    previous_word_id = None

    for prediction_id, word_id in zip(prediction_ids, word_ids):
        if word_id is None or word_id == previous_word_id:
            previous_word_id = word_id
            continue

        labels.append(id2label[int(prediction_id)])
        previous_word_id = word_id

    words = words[: len(labels)]
    spans = spans[: len(labels)]
    labels = apply_domain_label_overrides(words, labels)

    return words, spans, labels


def apply_domain_label_overrides(words, labels):
    labels = list(labels)
    lowered_words = [word.lower() for word in words]

    for index, word in enumerate(lowered_words):
        if word in KNOWN_PRODUCT_WORDS:
            labels[index] = "B-PRODUCT"
        elif word in NON_ENTITY_WORDS:
            labels[index] = "O"
        elif word.isdigit():
            previous_word = lowered_words[index - 1] if index > 0 else ""
            next_word = lowered_words[index + 1] if index + 1 < len(lowered_words) else ""
            if previous_word in {"bali", "total"} or next_word == "tanan":
                labels[index] = "B-QUANTITY"

    for phrase in ACTION_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(lowered_words) - phrase_length + 1):
            if tuple(lowered_words[index : index + phrase_length]) != phrase:
                continue

            labels[index] = "B-ACTION"
            for offset in range(1, phrase_length):
                labels[index + offset] = "I-ACTION"

    return labels


def extract_entities(words, spans, labels):
    entities = []
    current_type = None
    current_words = []
    start = None
    end = None

    def close_entity():
        if current_type is None:
            return

        entities.append(
            {
                "type": current_type.lower(),
                "text": " ".join(current_words),
                "start": start,
                "end": end,
            }
        )

    for word, span, label in zip(words, spans, labels):
        if label == "O":
            close_entity()
            current_type = None
            current_words = []
            start = None
            end = None
            continue

        prefix, entity_type = label.split("-", 1)
        starts_new_entity = prefix == "B" or current_type != entity_type

        if starts_new_entity:
            close_entity()
            current_type = entity_type
            current_words = [word]
            start, end = span
        else:
            current_words.append(word)
            end = span[1]

    close_entity()

    return entities


def predict_entities(text):
    words, spans, labels = predict_word_labels(text)
    return extract_entities(words, spans, labels)


def add_destination_entity(text, intent, entities):
    words, spans = split_words(text)
    lowered_words = [word.lower() for word in words]
    start_index = None

    for index, word in enumerate(lowered_words):
        if word in ADDRESS_START_WORDS:
            start_index = index + 1
            break

    if start_index is None:
        for phrase in ADDRESS_START_PHRASES:
            phrase_length = len(phrase)
            for index in range(len(lowered_words) - phrase_length + 1):
                if tuple(lowered_words[index : index + phrase_length]) == phrase:
                    start_index = index + phrase_length
                    break
            if start_index is not None:
                break

    if start_index is None or start_index >= len(words):
        return entities

    end_index = len(words)
    for index in range(start_index, len(words)):
        if lowered_words[index] in ADDRESS_STOP_WORDS:
            end_index = index
            break

    if end_index <= start_index:
        return entities

    start, _ = spans[start_index]
    _, end = spans[end_index - 1]
    address_text = " ".join(words[start_index:end_index])

    has_existing_location = any(
        entity["type"] == "location"
        and entity["start"] <= start
        and entity["end"] >= end
        for entity in entities
    )
    if has_existing_location:
        return entities

    filtered_entities = [
        entity
        for entity in entities
        if not (
            entity["type"] == "location"
            and entity["start"] < end
            and start < entity["end"]
        )
    ]

    return filtered_entities + [
        {
            "type": "location",
            "text": address_text,
            "start": start,
            "end": end,
        }
    ]


def add_delivery_area_entity(text, intent, entities):
    if intent["label"] != "business_info":
        return entities

    words, spans = split_words(text)
    lowered_words = [word.lower() for word in words]
    if not is_delivery_area_question(lowered_words):
        return entities

    for index, word in enumerate(lowered_words):
        if word not in {"to", "in", "sa"}:
            continue

        area_index = index + 1
        if area_index >= len(words):
            continue

        area_start, area_end = spans[area_index]
        has_existing_location = any(
            entity["type"] == "location"
            and entity["start"] <= area_start
            and entity["end"] >= area_end
            for entity in entities
        )
        if has_existing_location:
            return entities

        return entities + [
            {
                "type": "location",
                "text": words[area_index],
                "start": area_start,
                "end": area_end,
            }
        ]

    return entities


def clean_contextual_entities(text, intent, entities):
    words = [word.lower() for word in re.findall(r"[A-Za-z-]+", text)]

    if intent["label"] == "delivery_time":
        return []

    if intent["label"] == "business_info" and is_business_hours_question(set(words)):
        return []

    if intent["label"] == "business_info" and is_business_policy_question(words, set(words)):
        return []

    if intent["label"] == "ask_price":
        return [
            entity
            for entity in entities
            if not (
                entity["type"] == "payment"
                and entity["text"].lower() == "delivery"
            )
        ]

    if intent["label"] == "provide_address":
        return [
            entity
            for entity in entities
            if entity["type"] not in {"product", "action"}
        ]

    return entities


def build_secondary_intents(text, intent):
    if intent["label"] == "provide_address":
        return []

    words = [word.lower() for word in re.findall(r"[A-Za-z-]+", text)]
    if has_destination_phrase(words):
        return [
            {
                "label": "provide_address",
                "source": "rule",
            }
        ]

    return []


def has_destination_phrase(words):
    if any(word in ADDRESS_START_WORDS for word in words):
        return True

    for phrase in ADDRESS_START_PHRASES:
        phrase_length = len(phrase)
        for index in range(len(words) - phrase_length + 1):
            if tuple(words[index : index + phrase_length]) == phrase:
                return True

    return False


def group_slots(entities):
    slots = {}

    for entity in entities:
        slots.setdefault(entity["type"], []).append(entity["text"])

    return slots


def build_product_quantity_items(entities):
    items = []
    pending_quantity = None

    for entity in sorted(entities, key=lambda item: item["start"]):
        if entity["type"] == "quantity":
            if items and "quantity" not in items[-1]:
                items[-1]["quantity"] = entity["text"]
            else:
                pending_quantity = entity["text"]

        elif entity["type"] == "product":
            item = {"product": entity["text"]}
            if pending_quantity is not None:
                item["quantity"] = pending_quantity
                pending_quantity = None
            items.append(item)

    return items


def build_modification(intent, entities):
    if intent["label"] != "modify_order":
        return None

    items = build_product_quantity_items(entities)
    if len(items) < 2:
        return None

    return {
        "from": items[0],
        "to": items[1],
    }


def build_addition(intent, entities):
    if intent["label"] != "add_quantity":
        return None

    products = [entity["text"] for entity in entities if entity["type"] == "product"]
    quantities = [
        entity["text"]
        for entity in entities
        if entity["type"] == "quantity"
    ]

    if not products or not quantities:
        return None

    addition = {
        "product": products[0],
    }

    if len(quantities) == 1:
        addition["add_quantity"] = quantities[0]
    else:
        addition["current_quantity"] = quantities[0]
        addition["add_quantity"] = quantities[1]

    if len(quantities) >= 3:
        addition["total_quantity"] = quantities[2]

    return addition


def build_answer(text, intent, entities):
    label = intent["label"]
    words = [word.lower() for word in re.findall(r"[A-Za-z-]+", text)]
    products = [entity["text"] for entity in entities if entity["type"] == "product"]
    quantities = [entity["text"] for entity in entities if entity["type"] == "quantity"]
    locations = [entity["text"] for entity in entities if entity["type"] == "location"]

    if label == "greeting":
        return "Hello! I can help with products, orders, delivery, and payment."
    if label == "help":
        return "You can ask about products, prices, availability, orders, delivery, addresses, or payment."
    if label == "thanks":
        return "You are welcome!"
    if label == "order_item":
        if products and quantities:
            return f"You want to order {quantities[0]} {products[0]}. Please confirm before I place it."
        if products:
            return f"You want to order {', '.join(products)}. What quantity would you like?"
        return "What product and quantity would you like to order?"
    if label == "add_quantity":
        if products and quantities:
            return f"You want to add {quantities[-1]} more {products[0]}. Please confirm before I update the order."
        return "What product and quantity would you like to add?"
    if label == "ask_price":
        product_text = ", ".join(products) if products else "that product"
        return f"I do not have a price listed for {product_text} yet."
    if label == "check_availability":
        product_text = ", ".join(products) if products else "that product"
        return f"I cannot confirm whether {product_text} is in stock because live inventory is not connected yet."
    if label == "business_info":
        if is_delivery_area_question(words):
            return "I can help with delivery to your area, but the delivery-area list is not connected yet."
        return "We sell drinks and accept orders for delivery. Ask me about a specific product or service."
    if label == "payment_method":
        return "We can accept cash, GCash, Maya, card, or cash on delivery."
    if label == "delivery_time":
        return "Delivery time depends on your location. Please provide your area so I can check it."
    if label == "provide_address":
        location_text = ", ".join(locations)
        return f"I noted your delivery location: {location_text}." if location_text else "Please provide your complete delivery address."
    if label == "provide_contact":
        return "I noted your contact details."
    if label == "order_status":
        return "I cannot check order status yet because order tracking is not connected."
    if label == "cancel_order":
        return "I can help cancel your order. Please provide your order number."
    if label == "modify_order":
        return "I can help change your order. Please tell me the product you want to replace."
    if label == "confirmation":
        return "Thanks for confirming."
    if label == "rejection":
        return "Understood. I will not make that change."
    if label == "complaint":
        return "I am sorry about that. Please describe the problem so I can help."
    if label == "out_of_scope":
        return "I can help with products, orders, delivery, addresses, and payment questions."
    return "I did not fully understand that. Please rephrase your question."


def should_return_entities(intent):
    return intent["label"] not in {"out_of_scope", "fallback"}


@app.get("/", include_in_schema=False)
def chat_interface():
    return FileResponse(BASE_DIR / "web" / "index.html")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "intent_model_ready": INTENT_MODEL_DIR.is_dir(),
        "ner_model_ready": NER_MODEL_DIR.is_dir(),
    }


@app.api_route("/predict", methods=["GET", "POST"])
def predict(
    text: str | None = Query(default=None, description="Text to classify"),
    body: RequestBody | None = None,
):
    if body is not None:
        text_value = body.text
    else:
        text_value = text

    if text_value is None:
        raise HTTPException(status_code=400, detail="text must be provided")

    text_value = text_value.strip()
    if not text_value:
        raise HTTPException(status_code=400, detail="text must not be blank")

    intent_prediction = get_intent_classifier()(text_value)
    intent = apply_intent_overrides(text_value, format_intent(intent_prediction))
    entities = predict_entities(text_value) if should_return_entities(intent) else []
    entities = add_destination_entity(text_value, intent, entities)
    entities = add_delivery_area_entity(text_value, intent, entities)
    entities = clean_contextual_entities(text_value, intent, entities)
    secondary_intents = build_secondary_intents(text_value, intent)
    modification = build_modification(intent, entities)
    addition = build_addition(intent, entities)

    response = {
        "text": text_value,
        "intent": intent,
        "entities": entities,
        "slots": group_slots(entities),
        "answer": build_answer(text_value, intent, entities),
    }

    if intent["label"] == "fallback":
        response["clarification"] = "I didn’t catch that—can you please rephrase?"

    if secondary_intents:
        response["secondary_intents"] = secondary_intents

    if secondary_intents:
        response["secondary_intents"] = secondary_intents

    if modification is not None:
        response["modification"] = modification

    if addition is not None:
        response["addition"] = addition

    return response
