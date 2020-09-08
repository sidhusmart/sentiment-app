from typing import Optional
from fastapi import Depends, FastAPI
from pydantic import BaseModel

app = FastAPI()

class Review(BaseModel):
    reviewText: str
    reviewerID: Optional[str] = None
    asin: Optional[str] = None
    sentiment: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "reviewText": "This was a great purchase, saved me much time test!",
                "reviewerID": "A1VU337W6PKAR3",
                "asin": "B00K0TIC56"
            }
        }

@app.post("/api/v1/sentiment", response_model=Review)
async def predict(review: Review):
    sentiment_label = ["NEGATIVE", "POSITIVE"]
    review.sentiment = sentiment_label[1]
    return review

