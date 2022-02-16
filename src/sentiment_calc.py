import shap
from transformers import pipeline

HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def calc_sentiment(example_text):
    PATH = '../models/sentiment/sentiment'

    model_path = PATH
    tokenizer_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    sentiment_task = pipeline("sentiment-analysis", framework="pt", model=model_path, tokenizer=tokenizer_path)

    sentiment = sentiment_task(example_text)

    explainer = shap.Explainer(sentiment_task)
    l = []
    l.append(example_text)
    shap_values = explainer(l)

    return shap_values,sentiment
