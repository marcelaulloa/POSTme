# # -*- coding: utf-8 -*-
# Loading Packages
from lime.lime_text import LimeTextExplainer
from transformers import pipeline
import features_functions as ff
import shap
import lime
import joblib
# import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns
# import plotly.graph_objects as go
# import math
# import json
from datetime import datetime

import streamlit as st

# import requests, zipfile, io
# import plotly.graph_objs as go
# from datetime import datetime, timedelta
#
import sklearn
# import sklearn.ensemble
# import sklearn.metrics
# from sklearn import model_selection
# from sklearn import metrics
# from sklearn.pipeline import Pipeline
# from sklearn.compose import ColumnTransformer
# from sklearn.preprocessing import StandardScaler, OneHotEncoder
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (confusion_matrix, classification_report, f1_score, accuracy_score,r2_score)
# from sklearn.metrics import precision_recall_fscore_support, mutual_info_score
# from sklearn.model_selection import cross_val_score
#
import tensorflow as tf
from tensorflow import keras
from transformers import pipeline
import streamlit.components.v1 as components

#
# # !pip install textsearch
# # !pip install contractions
# # !pip install tqdm
# # !pip install shap
# # !pip install lime
# # !pip install transformers
# # !pip install sentencepiece
# # !pip install --upgrade streamlit opencv-python
# # from __future__ import print_function
#
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)
#
# import warnings
# warnings.filterwarnings("ignore")

#####################
# Loading Functions #
#####################
import shap_functions as sf
import features_functions as ff

##################
# Model Loading #
##################

HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

PATH = '../models/sentiment'

model_path = PATH
tokenizer_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer_path)

filename = '../models/pipeline_random.sav'
pipeline_rf = joblib.load(filename)

filename2 = '../models/shap_explainer.sav'
explainer = joblib.load(filename2)

filename3 = '../models/pipeline_nb_lime.sav'
pipeline_nb = joblib.load(filename3)

filename4 = '../models/vectorizer.sav'
vectorizer = joblib.load(filename4)

topics = ["Iron Deficiency", "Chronic Kidney Disease", "ANCA associated vasculitis", "Corporate Communication"]
time = ['morning', 'afternoon', 'evening']
ptypes = ['video', 'photo', 'status']

col1, col2 = st.columns([1,4])
with col1:
    image = '../image/LOGO-01.jpg'
    st.image(image, caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
with col2:
    st.title("POSTMe!")
    st.subheader("Tool for Social Media Messaging Optimization")


##############
# User Input #
##############
m = st.markdown("""
<style>
div.sttext_area > text_area:first-child {
    background-color: rgb(42, 42, 42);
}
</style>""", unsafe_allow_html=True)
example = {'data_text': st.text_area('Post to analyze', "")}


col1, col2 = st.columns(2)
with col1:
    topic = st.selectbox('Select a Topic for the Post', topics)
    date = st.date_input("Day of Posting", datetime.date(datetime.now()))
    day = date.weekday()

with col2:
    post_type = st.selectbox('Select Post Type', ptypes)
    time_post = st.selectbox('Select Time of Post', time)

if st.button('Predict'):
    #######################
    # Feature Engineering #
    #######################

    example['norm_posts'] = ff.text_pre_processor(example['data_text'])
    example = ff.textblob_labls(example)
    example = ff.count_features(example)

    example['post_type'] = post_type
    example['day_time'] = time_post
    example['day_of_week'] = day

    if topic == "Iron Deficiency":
        example['Topic 2'] = 0
    elif topic == "Chronic Kidney Disease":
        example['Topic 2'] = 1
    elif topic == "ANCA associated vasculitis":
        example['Topic 2'] = 2
    else:
        example['Topic 2'] = 3

    sentiment = sentiment_task(example['norm_posts'])
    norm_text = example['norm_posts']

    ###########################
    # Sentiment SHAP Analysis #
    ###########################
    st.header('Sentiment Analysis')
    st.write(
        "The words highlighted below are considered by the model to predict sentiment.\n\n"
        "0 - Negative    1 - Neutral     2 - Positive")
    explainer = shap.Explainer(sentiment_task)
    shap_values = explainer([norm_text])
    shap.initjs()
    components.html(shap.plots.text(shap_values[0, :, sentiment[0]['label']], display=False), height=200)

    #################################
    # Custom Features SHAP Analysis #
    #################################

    st.header('Feature Importance Analysis')
    st.write(
        "Based on the company's dataset the features highlighted below are the ones considered by the model to predict engagement.\n\n"
        " 1 - Above average engagement    0 - Below average engagement.")
    example['Sentiment'] = sentiment[0]['label']
    example = ff.prepare_demo_df(example)
    explainer = joblib.load(filename2)
    explainer.expected_value, shap_values, example = sf.plot_shap(example, pipeline_rf, explainer)
    shap.initjs()
    st.pyplot(shap.plots.force(explainer.expected_value, shap_values, example,matplotlib=True,figsize=(30, 4),link ='logit'))

    #################################
    # Word Embeddings LIME Analysis #
    #################################
    import streamlit.components.v1 as components
    st.header('Text Analysis')
    st.write(
        "The words highlighted below have high impact in predicting the engagement. Orange indicates high engagement and blue indicates low engagement.")
    test_vector = vectorizer.transform([norm_text])
    class_names = [0,1]
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(norm_text, pipeline_nb.predict_proba, num_features=10, labels=[0, 1])
    html = exp.as_html(text=norm_text, labels=(1,))
    components.html(html, height=800)