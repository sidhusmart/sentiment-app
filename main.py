import pickle
from typing import Optional
from enum import Enum
from fastapi import Depends, FastAPI
from pydantic import BaseModel

import preprocessing

app = FastAPI()

class Review(BaseModel):
    reviewText: str
    reviewerID: Optional[str] = None
    asin: Optional[str] = None
    sentiment: Optional[str] = None

    class Config:
        schema_extra = {
            "example": {
                "reviewText": "This was a great purchase, saved me much time!",
                "reviewerID": "A1VU337W6PKAR3",
                "asin": "B00K0TIC56"
            }
        }

class ComponentValues(Enum):
    APT = "APT"
    Core = "Core"
    Debug = "Debug"
    Doc = "Doc"
    Text = "Text"
    UI = "UI"

class Bug(BaseModel):
    bugTitle: str
    bugDescription: str
    Issue_id: int
    Component: Optional[ComponentValues] = None

def load_model():
    try:
      global saved_model, tfidf_vectorizer
      saved_model = pickle.load(open('models/sentiment_classification.pickle', 'rb'))
      tfidf_vectorizer = pickle.load(open('models/tfidf_vectorizer.pickle', 'rb'))
      print('sentiment models loaded')
      global bugs_classifier, tfidf_vectorizer_bugs
      bugs_classifier = pickle.load(open('models/bugs_classifier.pickle', 'rb'))
      tfidf_vectorizer_bugs = pickle.load(open('models/tfidf_vectorizer_bugs.pickle', 'rb'))
      print('bugs classification models loaded')
    except Exception as e:
      raise ValueError('No model here')

@app.post("/api/v1/sentiment", response_model=Review)
async def predict(review: Review, model = Depends(load_model())):
    text_clean = preprocessing.clean(review.reviewText)
    text_tfidf = tfidf_vectorizer.transform([text_clean])
    sentiment = saved_model.predict(text_tfidf)
    sentiment_label = ["NEGATIVE", "POSITIVE"]
    review.sentiment = sentiment_label[sentiment.item()]
    return review

@app.post("/api/v1/classifybugs", response_model=Bug)
async def predict(bug: Bug, model = Depends(load_model())):
    text = bug.bugTitle + ' ' + bug.bugDescription
    text_clean = preprocessing.clean(text)
    text_tfidf = tfidf_vectorizer_bugs.transform([text_clean])
    output = bugs_classifier.predict(text_tfidf)
    bug.Component = output.item()
    return bug
