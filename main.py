tokenizer = AutoTokenizer.from_pretrained("cajcodes/DistilBERT-PoliticalBias")
model = AutoModelForSequenceClassification.from_pretrained("cajcodes/DistilBERT-PoliticalBias")
model.eval()


from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = FastAPI()

tokenizer = AutoTokenizer.from_pretrained("cajcodes/DistilBERT-PoliticalBias")
model = AutoModelForSequenceClassification.from_pretrained("cajcodes/DistilBERT-PoliticalBias")

class TextInput(BaseModel):
    text: str

@app.post("/predict_bias")
async def predict_bias(input: TextInput):
    inputs = tokenizer(input.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    label_map = {0: "Left", 1: "Center", 2: "Right"}

    return {
        "text": input.text,
        "predicted_class": predicted_class,
        "bias_label": label_map.get(predicted_class, "Unknown")
    }

@app.get("/")
async def root():
    return {"message": "BiasCheck API is running on Vercel with Docker!"}
