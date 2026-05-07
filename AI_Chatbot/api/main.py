from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel

app = FastAPI()

classifier = pipeline(
    "text-classification",
    model="./model/results",
)

class RequestBody(BaseModel):
    text: str


@app.post("/predict")
def predict(body: RequestBody):
    result = classifier(body.text)
    return result