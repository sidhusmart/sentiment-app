#!/bin/bash
source activate sentiment-app 
gunicorn -w 3 -b :5000 -t 5 -k uvicorn.workers.UvicornWorker main:app
