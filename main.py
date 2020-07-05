import pickle
import traceback
import json
import re
import html
from flasgger import Swagger
from flask import Flask, request, redirect, url_for, flash, jsonify
import preprocessing

Sentiment = ["NEGATIVE", "POSITIVE"]

app = Flask(__name__)

swagger = Swagger(app)

@app.route('/api/v1/sentiment', methods=['POST'])
def predict():
  """API Endpoint used for serving the prediction function of the Sentiment Analysis model

  The sentiment analysis model has been trained using Amazon customer reviews and works best in similar scenarios
  When the text of a customer review is passed, it is cleaned and either a POSITIVE or NEGATIVE sentiment prediction is returned

  ---
  parameters:
    - in: body 
      name: body 
      schema:
        id: schema_id
        required:
          - review_text
        properties:
          text:
            type: string
	    description: Input review text that needs to be scored
	    default: "This is a great product. I would highly recommend it!"
  responses:
    200:
      description: SUCCESS
      examples: {"prediction":"POSITIVE"}
  """
  response = dict()
  if request.method == 'POST':
    data = request.get_json()
    text_clean = preprocessing.clean(data['text'])
    text_tfidf = tfidf_vectorizer.transform([text_clean])
    output = saved_model.predict(text_tfidf)
    response["prediction"] = Sentiment[output.item()]
  return jsonify(response)

if __name__ == '__main__':
	try:
		saved_model = pickle.load(open('sentiment_classification.pickle', 'rb'))
		tfidf_vectorizer = pickle.load(open('tfidf_vectorizer.pickle', 'rb'))
		print('model loaded')
	except Exception as e:
		raise ValueError('No model here')
	app.run(host="0.0.0.0", port=5000, debug=True)
