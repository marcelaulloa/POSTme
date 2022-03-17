#!/usr/bin/env python
# coding: utf-8

# # Loading Dependencies
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import json
import streamlit as st
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, f1_score, accuracy_score,r2_score)
from sklearn.metrics import precision_recall_fscore_support, mutual_info_score
from sklearn.model_selection import cross_val_score


import tensorflow as tf
from tensorflow import keras
from transformers import pipeline
import lime
import sklearn
import numpy as np
import sklearn
import sklearn.ensemble
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")


# # Functions

# In[2]:


# Text Proprocessing and Feature Creation
import nltk

nltk.download('punkt')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
import unicodedata
import contractions
import spacy
import nltk
import pandas as pd

nlp = spacy.load('en_core_web_sm')
ps = nltk.porter.PorterStemmer()


# Links removal
def remove_links(text):
    """Takes a string and removes web links from it"""
    text = re.sub(r'http\S+', '', text)  # remove http links
    text = re.sub(r'bit.ly/\S+', '', text)  # remove bitly links
    text = text.strip('[link]')  # remove [links]
    text = re.sub(r'pic.twitter\S+', '', text)
    return text


# Retweet and @user information removal
def remove_users(text):
    """Takes a string and removes retweet and @user information"""
    text = re.sub('(RT\s@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove re-tweet
    text = re.sub('(@[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove tweeted at
    return text


# Hash tags removal
def remove_hashtags(text):
    """Takes a string and removes any hash tags"""
    text = re.sub('(#[A-Za-z]+[A-Za-z0-9-_]+)', '', text)  # remove hash tags
    return text


# AUDIO/VIDEO tags or labels removal
def remove_av(text):
    """Takes a string and removes AUDIO/VIDEO tags or labels"""
    text = re.sub('VIDEO:', '', text)  # remove 'VIDEO:' from start of tweet
    text = re.sub('AUDIO:', '', text)  # remove 'AUDIO:' from start of tweet
    return text


# HTML removal
def strip_html_tags(text):
    soup = BeautifulSoup(text, "html.parser")
    [s.extract() for s in soup(['iframe', 'script'])]
    stripped_text = soup.get_text()
    stripped_text = re.sub(r'[\r|\n|\r\n]+', '\n', stripped_text)
    return stripped_text


# accent removal
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


# contraction expansion
def expand_contractions(text):
    return contractions.fix(text)


# lemamtization
def spacy_lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text


# stemming
def simple_stemming(text, stemmer=ps):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# special character removal
def remove_special_characters(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    return text


# stopword removal
def remove_stopwords(text, is_lower_case=False, stopwords=None):
    if not stopwords:
        stopwords = nltk.corpus.stopwords.words('english')
    tokens = nltk.word_tokenize(text)
    tokens = [token.strip() for token in tokens]

    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopwords]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopwords]

    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


import tqdm  # progressbar


def text_pre_processor(text, remove_links_=True, remove_users_=True, remove_hashtags_=True, remove_av_=True,
                       html_strip=True, accented_char_removal=True, contraction_expansion=True,
                       text_lower_case=True, text_stemming=False, text_lemmatization=True,
                       special_char_removal=True, remove_digits=True, stopword_removal=True,
                       stopword_list=None):
    # remove links
    if remove_links_:
        text = remove_links(text)

    # remove users and retweets
    if remove_users_:
        text = remove_users(text)

    # remove hash tags
    if remove_hashtags_:
        text = remove_hashtags(text)

        # remove audio video
    if remove_av_:
        text = remove_av(text)

    # strip HTML
    if html_strip:
        text = strip_html_tags(text)

    # remove extra newlines (often might be present in really noisy text)
    text = text.translate(text.maketrans("\n\t\r", "   "))

    # remove accented characters
    if accented_char_removal:
        text = remove_accented_chars(text)

    # expand contractions    
    if contraction_expansion:
        text = expand_contractions(text)

    # lemmatize text
    if text_lemmatization:
        text = spacy_lemmatize_text(text)

        # remove special characters and\or digits
    if special_char_removal:
        # insert spaces between special characters to isolate them    
        special_char_pattern = re.compile(r'([{.(-)!}])')  # 'I will not go!here' => I will not go ! here'
        text = special_char_pattern.sub(" \\1 ", text)
        text = remove_special_characters(text, remove_digits=remove_digits)

        # stem text
    if text_stemming and not text_lemmatization:
        text = simple_stemming(text)

    # lowercase the text    
    if text_lower_case:
        text = text.lower()

    # remove stopwords
    if stopword_removal:
        text = remove_stopwords(text, is_lower_case=text_lower_case,
                                stopwords=stopword_list)

    # remove extra whitespace
    text = re.sub(' +', ' ', text)  # 'I   will  not' => 'I will not'
    text = text.strip()

    return text


def corpus_pre_processor(corpus):
    norm_corpus = []
    for doc in tqdm.tqdm(corpus):
        norm_corpus.append(text_pre_processor(doc))
    return norm_corpus


import textblob


def textblob_labls(example):
    df_snt_obj = textblob.TextBlob(example['data_text']).sentiment
    example['subjectivity'] = df_snt_obj.subjectivity

    return example


def count_features(example):
    example['n_mentions'] = example['data_text'].count('@')
    example['n_hashtags'] = example['data_text'].count('#')
    example['n_links'] = example['data_text'].count('https://')
    return example


def prepare_demo_df(example):
    example = pd.DataFrame.from_dict(example, orient='index')
    example = example.T
    example = example.drop(columns=['data_text', 'norm_posts'])
    example['n_hashtags'] = example['n_hashtags'].astype('int64')
    example['n_mentions'] = example['n_mentions'].astype('int64')
    example['n_links'] = example['n_links'].astype('int64')
    example['subjectivity'] = example['subjectivity'].astype('float64')
    example['post_type'] = example['post_type'].astype('object')
    example['Topic 2'] = example['Topic 2'].astype('category')
    example['day_of_week'] = example['day_of_week'].astype('category')
    example['Sentiment'] = example['Sentiment'].astype('category')
    return example


# In[3]:


# Sentiment Calculation

import shap
from transformers import pipeline

HF_DATASETS_OFFLINE = 1
TRANSFORMERS_OFFLINE = 1

import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def calc_sentiment(example_dict, example_text):
    PATH = '../models/sentiment'

    model_path = PATH
    tokenizer_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

    sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer_path)

    sentiment = sentiment_task(example_text)

    explainer = shap.Explainer(sentiment_task)
    l = []
    l.append(example_text)
    shap_values = explainer(l)

    return shap_values,sentiment


# In[4]:


# SHAP Functions File

def def_feature_cols(pipeline_rf, categorical_features):
    num_features2 = list(pipeline_rf["pre_process"].transformers_[0][2])

    cat_features2 = list(
        pipeline_rf["pre_process"]
            .transformers_[1][1]["onehot"]
            .get_feature_names(categorical_features)
    )
    feature_cols = num_features2 + cat_features2
    return feature_cols


def plot_shap(example, pipeline_rf, explainer):
    categorical_features = example.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_features = example.select_dtypes(include=["int64", "float64"]).columns.tolist()

    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer,
         numeric_features),
        ("cat", categorical_transformer,
         categorical_features)
    ])

    y_pred_rf = pipeline_rf.predict(example)
    y_pred_rf_proba = pipeline_rf.predict_proba(example)

    example = pipeline_rf["pre_process"].transform(example)

    shap_values = explainer.shap_values(example)

    example = pd.DataFrame(example, columns=def_feature_cols(pipeline_rf, categorical_features))

    return explainer.expected_value, shap_values, example


# In[5]:


HF_DATASETS_OFFLINE=1 
TRANSFORMERS_OFFLINE=1


# In[6]:


# import os
# os.environ['CUDA_LAUNCH_BLOCKING']='1'


# In[7]:


# from google.colab import drive
# drive.mount('/content/drive')


# In[8]:


PATH = '../models/sentiment'


# In[9]:


model_path = PATH
tokenizer_path = "cardiffnlp/twitter-xlm-roberta-base-sentiment"

sentiment_task = pipeline("sentiment-analysis", model=model_path, tokenizer=tokenizer_path)


# In[11]:


from lime.lime_text import LimeTextExplainer

def plot_lime(test_clean_text,nb_pipeline,vec):
  test_vector = vec.transform([test_clean_text])

  class_names = [0,1]
  explainer = LimeTextExplainer(class_names=class_names)

  exp = explainer.explain_instance(test_clean_text, nb_pipeline.predict_proba, num_features=20, labels=[0, 1])
  # print('Document id: %d' % idx)
  print('Predicted class =', class_names[nb_pipeline[1].predict(test_vector).reshape(1,-1)[0,0]])
  # print('True class: %s' % y_test.loc[idx])
  print(nb_pipeline.predict_proba([test_clean_text]).round(3))

  exp.show_in_notebook(text=test_clean_text, labels=(1,))


# # Data Loading

# In[12]:


# !gdown --id 1jboYjE-G-lBo_Ad8uwmVc-gr02DaKhou


# In[13]:


# !gdown --id 19nVclYQINAjU04CnGiopJ31WWO53_wp2


# In[14]:


# !gdown --id 1KfTNlVnLWmMl-vbvicUZ7SniiaSlTU0J


# In[15]:


# !gdown --id 1-7Iu-lj5BybGBYn4yn8Emqc85nt0V_dt


# # Model Loading

# In[16]:


# load the model from disk
filename = '../models/pipeline_random.sav'
pipeline_rf = joblib.load(filename)

filename2 = '../models/shap_explainer.sav'
explainer = joblib.load(filename2)

filename3 = '../models/pipeline_nb_lime.sav'
pipeline_nb = joblib.load(filename3)

filename4 = '../models/vectorizer.sav'
vectorizer = joblib.load(filename4)


# # Shap Analysis - Negative

# In[17]:


# Testing Demo
example = {}
example['data_text'] = '''Patient insights play an important role in helping to improve healthcare systems. Thank you @iamnickevans, Julie and Faizan for sharing how chronic conditions impact quality of life!
Learn more: https://t.co/bBAEuMAQqd
#ViforPharmaPatientDay https://t.co/Nsw7ZfNezu'''

# calculating features
example['norm_posts'] = text_pre_processor(example['data_text'])
example = textblob_labls(example)
count_features(example)

# user input: 'day_of_week', 'post_type', 'day_time'
example['day_of_week'] = '0'
# photo,video,status
example['post_type'] = 'status'
# morning, afternoon, evening 
example['day_time'] = 'morning'
# other models input
example['Sentiment'] = sentiment_task(example['data_text'])
example['Topic 2'] = '0'
norm_text = example['norm_posts']


# ### Sentiment SHAP analysis

# In[18]:


import shap
explainer = shap.Explainer(sentiment_task) 
l = []
l.append(example['data_text'])
shap_values = explainer(l)
shap.plots.text(shap_values[0,:, example['Sentiment'][0]['label']])


# In[19]:


example['Sentiment'] = example['Sentiment'][0]['label']


# In[20]:


example = prepare_demo_df(example)
example


# ### Custom Features SHAP Analysis

# In[73]:


explainer = joblib.load(filename2)
explainer.expected_value, shap_values,example = plot_shap(example,pipeline_rf,explainer)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, example)


# ### Word Embeddings LIME Analysis

# In[22]:


plot_lime(norm_text,pipeline_nb,vectorizer)


# # Example of High Engagement Prediction

# ## User Input
# 
# Today on #WorldKidneyDay, at #ViforPharma we support and stand with other organizations, 
# advocacy groups and the patient community to raise global awareness about kidney disease as we continue to help patients lead better, 
# healthier lives. Learn more: https://t.co/rg7AZSRm2X https://t.co/FmnlxJAbM2
# 
# *   Day of the Week: Wednesday
# *   Day Time of the Post: Morning
# *   Type of Post: Status
# *   Topic of the Post: Chronic Kidney Disease

# In[23]:


# Testing Demo
example = {}
example['data_text'] = '''Today on #WorldKidneyDay, at #ViforPharma we support and stand with other organizations, 
advocacy groups and the patient community to raise global awareness about kidney disease as we continue to help patients lead better, 
healthier lives. Learn more: https://t.co/rg7AZSRm2X https://t.co/FmnlxJAbM2'''

# calculating features
example['norm_posts'] = text_pre_processor(example['data_text'])
example = textblob_labls(example)
count_features(example)

# user input: 'day_of_week', 'post_type', 'day_time', 'Topic 2'
# 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, 6 - Sunday
example['day_of_week'] = '0'
# photo,video,status
example['post_type'] = 'status'
# morning, afternoon, evening 
example['day_time'] = 'morning'
# other models input
example['Sentiment'] = sentiment_task(example['data_text'])
# Topics: 0 - Iron Deficiency, 1 - Chronic Kidney Disease, 
# 2 - Anca associated vasculitis, 3 - corporate communication
example['Topic 2'] = '0'
norm_text = example['norm_posts']


# ### Sentiment SHAP analysis

# In[24]:


import shap
explainer = shap.Explainer(sentiment_task) 
l = []
l.append(example['data_text'])
shap_values = explainer(l)
shap.plots.text(shap_values[0,:, example['Sentiment'][0]['label']])


# In[25]:


example['Sentiment'] = example['Sentiment'][0]['label']


# In[26]:


example = prepare_demo_df(example)
example


# ### Custom Features SHAP Analysis

# In[27]:


explainer = joblib.load(filename2)
explainer.expected_value, shap_values,example = plot_shap(example,pipeline_rf,explainer)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, example)


# ### Word Embeddings LIME Analysis

# In[28]:


plot_lime(norm_text,pipeline_nb,vectorizer)


# # Example of Low Eng Detection

# In[49]:


# Testing Demo
example = {}
example['data_text'] = '''Are missed treatment opportunities leaving room for potentially serious 
bone and cardiovascular complications? #kidneydisease #nephrology #kidneyhealth 
#KDIGO https://t.co/AX5aLiX4EF https://t.co/fjFAskdmTW'''

# calculating features
example['norm_posts'] = text_pre_processor(example['data_text'])
example = textblob_labls(example)
count_features(example)

# user input: 'day_of_week', 'post_type', 'day_time', 'Topic 2'
# 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, 6 - Sunday
example['day_of_week'] = '0'
# photo,video,status
example['post_type'] = 'status'
# morning, afternoon, evening 
example['day_time'] = 'morning'
# other models input
example['Sentiment'] = sentiment_task(example['data_text'])
# Topics: 0 - Iron Deficiency, 1 - Chronic Kidney Disease, 
# 2 - Anca associated vasculitis, 3 - corporate communication
example['Topic 2'] = '0'
norm_text = example['norm_posts']


# ### Sentiment SHAP analysis

# In[52]:


import shap
explainer = shap.Explainer(sentiment_task) 
l = []
l.append(example['norm_posts'])
shap_values = explainer(l)
shap.plots.text(shap_values[0,:,])


# In[31]:


example['Sentiment'] = example['Sentiment'][0]['label']


# In[32]:


example = prepare_demo_df(example)
example


# ### Custom Features SHAP Analysis

# In[33]:


explainer = joblib.load(filename2)
explainer.expected_value, shap_values,example = plot_shap(example,pipeline_rf,explainer)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, example)


# ### Word Embeddings LIME Analysis

# In[34]:


plot_lime(norm_text,pipeline_nb,vectorizer)


# # Example Engagement Improvement - Original

# In[56]:


# Testing Demo
example = {}
example['data_text'] = '''Are missed treatment opportunities leaving room for potentially serious 
bone and cardiovascular complications? #kidneydisease #nephrology #kidneyhealth 
#KDIGO https://t.co/AX5aLiX4EF https://t.co/fjFAskdmTW'''

# calculating features
example['norm_posts'] = text_pre_processor(example['data_text'])
example = textblob_labls(example)
count_features(example)

# user input: 'day_of_week', 'post_type', 'day_time', 'Topic 2'
# 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, 6 - Sunday
example['day_of_week'] = '3'

# photo,video,status
example['post_type'] = 'photo'

# morning, afternoon, evening 
example['day_time'] = 'afternoon'

# other models input
example['Sentiment'] = sentiment_task(example['data_text'])

# Topics: 0 - Iron Deficiency, 1 - Chronic Kidney Disease, 
# 2 - Anca associated vasculitis, 3 - corporate communication
example['Topic 2'] = '0'
norm_text = example['norm_posts']


# ### Sentiment SHAP analysis

# In[57]:


import shap
explainer = shap.Explainer(sentiment_task) 
l = []
l.append(example['data_text'])
shap_values = explainer(l)
shap.plots.text(shap_values[0,:, ])


# In[58]:


example['Sentiment'] = example['Sentiment'][0]['label']


# In[59]:


example = prepare_demo_df(example)
example


# ### Custom Features SHAP Analysis

# In[60]:


explainer = joblib.load(filename2)
explainer.expected_value, shap_values,example = plot_shap(example,pipeline_rf,explainer)
shap.initjs()
shap.force_plot(explainer.expected_value, shap_values, example)
print(shap_values)


# ### Word Embeddings LIME Analysis

# In[40]:


plot_lime(norm_text,pipeline_nb,vectorizer)


# # Example Engagement Improvement - Changed

# In[41]:


original_t = '''Are missed treatment opportunities leaving room for potentially serious 
bone and cardiovascular complications? #kidneydisease #nephrology #kidneyhealth 
#KDIGO https://t.co/AX5aLiX4EF https://t.co/fjFAskdmTW '''


# In[76]:


# Testing Demo
example = {}
example['data_text'] = '''frequent checkups help to avoid serious 
bone and cardiovascular complications #nephrology #kidneyhealth 
https://t.co/AX5aLiX4EF'''
st.write(example['data_text'])
# calculating features
example['norm_posts'] = text_pre_processor(example['data_text'])
example = textblob_labls(example)
count_features(example)

# user input: 'day_of_week', 'post_type', 'day_time', 'Topic 2'
# 0 - Monday, 1 - Tuesday, 2 - Wednesday, 3 - Thursday, 4 - Friday, 5 - Saturday, 6 - Sunday
example['day_of_week'] = '4'
# photo,video,status
example['post_type'] = 'photo'
# morning, afternoon, evening 
example['day_time'] = 'morning'
# other models input
example['Sentiment'] = sentiment_task(example['data_text'])
# Topics: 0 - Iron Deficiency, 1 - Chronic Kidney Disease, 
# 2 - Anca associated vasculitis, 3 - corporate communication
example['Topic 2'] = '0'
norm_text = example['norm_posts']


# ### Sentiment SHAP analysis

# In[77]:

import streamlit.components.v1 as components
import shap
explainer = shap.Explainer(sentiment_task) 
l = []
l.append(example['data_text'])
shap_values = explainer(l)
components.html(shap.plots.text(shap_values[0,:, example['Sentiment'][0]['label']], display=False), height=200)


# In[78]:


example['Sentiment'] = example['Sentiment'][0]['label']


# In[79]:


example = prepare_demo_df(example)
st.write(example)


# ### Custom Features SHAP Analysis

# In[80]:


explainer = joblib.load(filename2)
print(pipeline_rf.predict(example))
print(pipeline_rf.predict_proba(example))
explainer.expected_value, shap_values,example = plot_shap(example,pipeline_rf,explainer)
shap.initjs()
st.pyplot(shap.force_plot(explainer.expected_value, shap_values, example, matplotlib=True, figsize=(20, 3)))


# ### Word Embeddings LIME Analysis

# In[47]:


plot_lime(norm_text,pipeline_nb,vectorizer)
