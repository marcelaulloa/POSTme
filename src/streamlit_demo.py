# # -*- coding: utf-8 -*-
####################
# Loading Packages #
####################
from lime.lime_text import LimeTextExplainer
import shap
import lime
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import streamlit as st
import sklearn
import tensorflow as tf
from tensorflow import keras
from transformers import pipeline
import streamlit.components.v1 as components
import os

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

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

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

model_path = '../models/sentiment'
tokenizer_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"


@st.experimental_singleton
def loading_models():
    return (
        pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer_path),
        joblib.load('../models/pipeline_random.sav'),
        joblib.load('../models/shap_explainer.sav'),
        joblib.load('../models/pipeline_nb_lime.sav'),
        joblib.load('../models/vectorizer.sav'))


sentiment_task, pipeline_rf, explainer, pipeline_nb, vectorizer = loading_models()

col1, col2 = st.columns([1, 4])
with col1:
    image = '../image/LOGO-01.jpg'
    st.image(image, caption=None, width=200, use_column_width=None, clamp=False, channels="RGB", output_format="auto")
with col2:
    st.title("POSTMe!")
    st.subheader("Tool for Social Media Messaging Optimization")

##############
# User Input #
##############
example = {'data_text': st.text_area('Post to Analyze', "")}
topics = ["ANCA associated vasculitis", "Corporate Communication", "Chronic Kidney Disease", "Iron Deficiency"]
time = ['morning', 'afternoon', 'evening']
ptypes = ['video', 'photo', 'status']

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

    example['Sentiment'] = sentiment_task(example['norm_posts'])
    norm_text = example['norm_posts']

    ###########################
    # Sentiment SHAP Analysis #
    ###########################
    st.header('Sentiment Analysis')
    st.write(
        "The words highlighted below are considered by the model to predict sentiment.\n\n"
        "0 - Negative | 1 - Neutral | 2 - Positive")
    explainer_fn = shap.Explainer(sentiment_task)
    shap_values = explainer_fn([norm_text])
    components.html(shap.plots.text(shap_values[0, :, ], display=False), height=200)

    #################################
    # Custom Features SHAP Analysis #
    #################################

    st.header('Feature Importance Analysis')
    st.write(
        "Based on the company's dataset the features highlighted below are the ones considered by the model to predict engagement.\n\n"
        " 1 - Above average engagement | 0 - Below average engagement.")
    example['Sentiment'] = example['Sentiment'][0]['label']
    example = ff.prepare_demo_df(example)
    explainer.expected_value, shap_values, example, y_pred = sf.plot_shap(example, pipeline_rf, explainer)
    shap.initjs()
    st.pyplot(shap.force_plot(explainer.expected_value, shap_values, example, matplotlib=True, figsize=(20, 3)))
    st.write(shap_values)

    #################################
    # Word Embeddings LIME Analysis #
    #################################
    import streamlit.components.v1 as components

    st.header('Text Analysis')
    st.write(
        "The words highlighted below have high impact in predicting the engagement. Orange indicates high engagement and blue indicates low engagement.")
    test_vector = vectorizer.transform([norm_text])
    class_names = [0, 1]
    lime_text_explainer = LimeTextExplainer(class_names=class_names)
    exp = lime_text_explainer.explain_instance(norm_text, pipeline_nb.predict_proba, num_features=10, labels=[0, 1])
    html = exp.as_html(text=norm_text, labels=(1,))
    components.html(html, height=800)
