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

    print(f"Summary: {passed}/{total} correct")
    return passed == total


if __name__ == "__main__":
    success = run()
    raise SystemExit(0 if success else 1)
