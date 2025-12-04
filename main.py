from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str
    score: float

app = FastAPI()

MODEL_NAME = "DeepPavlov/distilrubert-base-cased-conversational"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    load_in_8bit=True,       
    device_map="auto"
)

clf = pipeline(
    "text-classification",
    model=model,
    tokenizer=tokenizer,
    return_all_scores=False
)

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    result = clf(req.text)[0]
    return PredictResponse(label=result["label"], score=float(result["score"]))

