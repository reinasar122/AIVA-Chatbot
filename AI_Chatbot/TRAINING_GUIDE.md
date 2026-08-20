# AIVA Chatbot Training Guide

The chatbot becomes reliable from clear examples and measured tests, not from
more epochs alone. Every intent example should express one user goal.

## 1. Install the environment

Your computer currently has the Windows Store `python.exe` alias, but not a
real Python installation. Install Python 3.12 from PowerShell:

```powershell
winget install --id Python.Python.3.12 --exact --source winget
```

Close and reopen VS Code after installation. Then confirm that the required
commands work:

```powershell
python --version
py --version
python -m pip --version
```

From the `AI_Chatbot` folder, create and activate a virtual environment:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If `py` is also unavailable, install Python 3.11 or newer and enable the
Python launcher during installation.

Required tools:

- Python 3.11 or newer
- pip, included with Python
- Git, useful for source control but not needed to train
- PowerShell, included with Windows

Optional VS Code tools:

- Microsoft Python extension
- Microsoft Pylance extension

The Python and Pylance extensions improve editing, autocomplete, and error
checking, but they do not install Python itself.

## 2. Audit and clean the data

Run the audit before every training run:

```powershell
python .\model\audit_intents.py
```

Resolve every conflicting text. Do not label a sentence such as `Order Coke,
how much total?` as only `order_item`; split it into two messages or choose the
primary intent consistently. Keep exact duplicates to a minimum.

Use the existing intent names consistently:

- `order_item`: wants to buy or order a product
- `add_quantity`: changes or increases an existing order
- `business_info`: asks about store products, hours, delivery area, fees, or policy
- `check_availability`: asks whether a specific product is in stock
- `ask_price`: asks for a price or total
- `payment_method`: asks how payment can be made
- `provide_address` and `provide_contact`: supplies customer details
- `out_of_scope`: unrelated questions

Add at least 30 varied examples per intent before trusting a label. Include
English, Cebuano, Filipino, spelling mistakes, short messages, and realistic
product names. Keep the intent the same while varying the wording.

## 3. Train the models

Train intent classification first, then NER:

```powershell
python .\model\train_intent.py
python .\model\train_ner.py
```

The scripts save models to `model/intent_model` and `model/model/ner_model`,
which are the paths used by the API.

## 4. Measure before changing settings

```powershell
python .\model\test_intent_regression.py
python .\model\evaluate_ner_manual.py
python .\model\test_api_regression.py
```

Use macro F1 and the regression phrases, not training accuracy alone. When a
phrase fails, add several similar examples to the correct intent, retrain, and
run the same checks again. Do not simply increase epochs: that can memorize
the training data and make new wording worse.

## 5. Start the API

```powershell
python -m uvicorn api.main:app --reload
```

Check readiness before sending chat messages:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Both `intent_model_ready` and `ner_model_ready` should be `True`. The API also
rejects blank messages and messages longer than 500 characters immediately.
It caches the models after the first request, so later responses are faster.

The API should only be considered ready when the regression tests pass and the
NER manual evaluation has acceptable entity-level F1. For a browser or mobile
chat interface, send JSON to `POST /predict` with this shape:

```json
{"text": "I want two Coke"}
```