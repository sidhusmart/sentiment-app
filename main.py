from typing import Optional
from fastapi import Depends, FastAPI
from pydantic import BaseModel
import pickle
from enum import Enum
import preprocessing

app = FastAPI()

class Sentiment(Enum):
    POSITIVE = 1
    NEGATIVE = 0

class Review(BaseModel):
    text: str
    reviewerID: Optional[str] = None
    asin: Optional[str] = None
    sentiment: Optional[Sentiment] = None

    class Config:
        schema_extra = {
            "example": {
                "text": "This was a great purchase, saved me much time!",
                "reviewerID": "A1VU337W6PKAR3",
                "asin": "B00K0TIC56"
            }
        }

def load_model():
    try:
      global model, vectorizer
      model = pickle.load(open('models/sentiment_classification.pickle','rb'))
      vectorizer = pickle.load(open('models/tfidf_vectorizer.pickle','rb'))
      print ('Models have been loaded')
    except Exception as e:
      raise ValueError('No model here')

@app.post("/api/v1/sentiment", response_model=Review)
async def predict(review: Review, model = Depends(load_model())):
    text_clean = preprocessing.clean(review.text)
    text_tfidf = vectorizer.transform([text_clean])
    sentiment = model.predict(text_tfidf)
    review.sentiment = sentiment.item().name
    return review
