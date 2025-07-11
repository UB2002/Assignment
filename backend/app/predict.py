from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from .model import SentimentModel

router = APIRouter()
sentiment = SentimentModel()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    label: str  
    score: float

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    result = sentiment.predict(req.text)
    return PredictResponse(**result)
