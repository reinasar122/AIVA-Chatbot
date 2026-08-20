import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from fastapi.testclient import TestClient
from api.main import app

cases = [
    {
        "text": "maya pwede?",
        "expected_intent": "payment_method",
        "description": "payment method phrase should be captured by API rules",
    },
    {
        "text": "Brgy. Palanas lang ihatud",
        "expected_intent": "provide_address",
        "description": "address phrase with barangay should map to provide_address",
    },
    {
        "text": "Upper Palanas lang hatud",
        "expected_intent": "provide_address",
        "description": "address phrase should keep provide_address even when product-like entity appears",
    },
    {
        "text": "can I pay with card",
        "expected_intent": "payment_method",
        "description": "general payment request should remain payment_method",
    },
    {
        "text": "deliver to my house in Talamban",
        "expected_intent": "provide_address",
        "description": "known address phrase should map to provide_address",
    },
    {
        "text": "what drinks do you have",
        "expected_intent": "business_info",
        "description": "catalog question should map to business_info",
    },
    {
        "text": "naa moy Coke",
        "expected_intent": "check_availability",
        "description": "Bisaya availability phrase should map to check_availability",
    },
    {
        "text": "tagpila ang Sprite",
        "expected_intent": "ask_price",
        "description": "Bisaya price phrase should map to ask_price",
    },
    {
        "text": "add two more Coke to my order",
        "expected_intent": "add_quantity",
        "description": "add phrase should map to add_quantity",
    },
]

client = TestClient(app)


def run():
    total = len(cases)
    passed = 0

    print("API regression test cases:\n")
    for case in cases:
        response = client.post("/predict", json={"text": case["text"]})
        if response.status_code != 200:
            print(f"FAIL: {case['text']}")
            print(f"  error: HTTP {response.status_code}")
            continue

        payload = response.json()
        actual = payload.get("intent", {}).get("label")
        if actual == case["expected_intent"]:
            print(f"PASS: {case['text']}")
            passed += 1
        else:
            print(f"FAIL: {case['text']}")
            print(f"  expected: {case['expected_intent']}")
            print(f"  actual:   {actual}")
            print(f"  details:  {case['description']}")
            print(f"  response: {payload}\n")

    get_response = client.get("/predict", params={"text": "maya pwede?"})
    if get_response.status_code == 200:
        print("PASS: GET /predict with text query")
        passed += 1
    else:
        print("FAIL: GET /predict with text query")
        print(f"  error: HTTP {get_response.status_code}")
        print(f"  response: {get_response.text}")

    total += 1
    print(f"Summary: {passed}/{total} correct")
    return passed == total


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
