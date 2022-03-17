# # -*- coding: utf-8 -*-
####################
# Loading Packages #
####################
import os
from datetime import datetime
import features_functions as ff
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import shap
import shap_functions as sf
import streamlit as st
from transformers import pipeline

st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

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

    #####################################
    # Calculate Sentiment SHAP Analysis #
    #####################################
    explainer_fn = shap.Explainer(sentiment_task)
    shap_values_text = explainer_fn([norm_text])

    sentiment_output = sentiment_task(norm_text)
    sentiment_label = sentiment_output[0]['label']
    shap_text_df = pd.DataFrame(shap_values_text.values[0], columns=['negative', 'neutral', 'positive'])
    shap_text_df['words'] = shap_values_text.data[0]
    shap_text_df = shap_text_df[shap_text_df['words'] != '']

    norm_text_list = norm_text.split()

    shap_text_df_new = df = pd.DataFrame()
    idx_df = 0
    idx_w = 0

    word_current = ""
    score_neg_current = 0
    score_neu_current = 0
    score_pos_current = 0

    while idx_w < len(norm_text_list) and idx_df < len(shap_text_df['words']):
        score_0, score_1, score_2, word = shap_text_df.iloc[idx_df]
        score_neg_current += score_0
        score_neu_current += score_1
        score_pos_current += score_2
        word = word.strip()
        word_current += word
        if word_current == norm_text_list[idx_w]:
            new_row = {'Word': word_current, 'Negative': score_neg_current, 'Neutral': score_neu_current,
                       'Positive': score_pos_current}
            shap_text_df_new = shap_text_df_new.append(new_row, ignore_index=True)
            word_current = ""
            score_neg_current = 0
            score_neu_current = 0
            score_pos_current = 0
            idx_w = idx_w + 1
        idx_df += 1

    shap_text_df_new = shap_text_df_new[['Word', sentiment_label]]
    shap_text_df_new[sentiment_label] = shap_text_df_new[sentiment_label].apply(lambda x: np.round(-x, 2))
    shap_text_df_new['Contribution'] = shap_text_df_new[sentiment_label].apply(lambda x: ff.class_shap(x))

    ###########################################
    # Calculate Custom Features SHAP Analysis #
    ###########################################

    example['Sentiment'] = example['Sentiment'][0]['label']
    example = ff.prepare_demo_df(example)
    example = example.loc[:,
              ["day_of_week", "n_hashtags", "n_links", "n_mentions", 'post_type', 'subjectivity', 'Topic 2',
               'Sentiment', 'day_time']]
    explainer.expected_value, shap_values, example_shap = sf.plot_shap(example, pipeline_rf, explainer)
    prediction = pipeline_rf.predict(example)

    # 'day_of_week', 'post_type', 'Topic 2', 'Sentiment', 'day_time'
    example['day_of_week'] = example['day_of_week'].astype('object')
    example['post_type'] = example['post_type'].astype('object')
    example['Topic 2'] = example['Topic 2'].astype('object')
    example['Sentiment'] = example['Sentiment'].astype('object')
    example['day_time'] = example['day_time'].astype('object')

    categorical_features = example.select_dtypes(include=["object", "category"]).columns.tolist()
    # numeric_features = example.select_dtypes(include=["int64", "float64"]).columns.tolist()

    numeric_features = list(pipeline_rf['pre_process'].transformers_[0][2])
    # categorical_features = list(
    #     pipeline_rf['pre_process'].transformers_[1][1]['onehot'].get_feature_names_out(categorical_features))

    numeric_features2 = ['Number of hashtags', 'Number of links', 'Number of mentions', 'Subjectivity']

    categorical_features = ['day_of_week_0', 'day_of_week_1', 'day_of_week_2',
                            'day_of_week_3', 'day_of_week_4', 'day_of_week_5', 'day_of_week_6',
                            'post_type_photo', 'post_type_status', 'post_type_video',
                            'Topic 2_0', 'Topic 2_1', 'Topic 2_2', 'Topic 2_3', 'Sentiment_Negative',
                            'Sentiment_Neutral', 'Sentiment_Positive', 'day_time_afternoon',
                            'day_time_evening', 'day_time_morning']
    categorical_features2 = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                             'Friday', 'Saturday', 'Sunday',
                             'Post type: photo', 'Post type: status', 'Post type: video',
                             'Topic: Iron Deficiency', 'Topic: Chronic Kidney Disease',
                             'Topic: Anca Associated Vasculitis', 'Topic: Corporate Communication',
                             'Sentiment: Negative', 'Sentiment: Neutral', 'Sentiment: Positive',
                             'Day time: afternoon', 'Day time: evening', 'Day time: morning']

    features = numeric_features + categorical_features
    features2 = numeric_features2 + categorical_features2

    example_num = example[['n_hashtags', 'n_links', 'n_mentions', 'subjectivity']].T
    example_num['values'] = example_num[0]
    example_num = example_num.reset_index()
    example_num['Features_orig'] = example_num['index']
    example_num = example_num[['Features_orig', 'values']]

    example_cat = example[['day_of_week', 'post_type', 'Topic 2', 'Sentiment', 'day_time']]
    example_cat = example_cat.transpose().astype(str)
    example_cat = example_cat.reset_index()
    example_cat['values'] = example_cat[0]

    example_cat['Features_orig'] = example_cat['index'] + '_' + example_cat['values']
    example_cat = example_cat[['Features_orig', 'values']]
    example_values = pd.concat((example_num, example_cat), axis=0)

    shap_df = pd.DataFrame(shap_values[0])
    shap_df['Features_orig'] = features
    shap_df['Features'] = features2
    shap_df = shap_df.rename(columns={0: 'Shap'})

    shap_df = pd.merge(shap_df, example_values, on='Features_orig', how='left')
    shap_df.fillna('', inplace=True)
    shap_df['Shap'] = shap_df['Shap'].apply(lambda x: round(x, 2))

    shap_df['Category'] = shap_df['values'].apply(lambda x: ff.feat_cat(x))
    shap_df = shap_df[shap_df['Shap'] != 0]
    shap_df['Impact'] = shap_df['Shap'].apply(lambda x: ff.class_shap(x))
    shap_df['Contribution'] = shap_df['Category'] + ' - ' + shap_df['Impact']

    #####################
    # Prediction Result #
    #####################
    if prediction == 0:
        result = 'Low'
        shap_df = shap_df.sort_values(by=['Shap'], ascending=False)
    else:
        result = 'High'
        shap_df = shap_df.sort_values(by=['Shap'], ascending=True)

    st.markdown(f'<h1 style="color:#fb5c3e;font-size:30px;">{"Prediction Result: " + result}</h1>',
                unsafe_allow_html=True)

    #######################################
    # Print Custom Features SHAP Analysis #
    #######################################
    st.subheader('Feature Importance Analysis')
    st.write('The following features impact positively and negatively in model the prediction:')

    fig = px.bar(shap_df, x='Shap', y='Features',
                 color='Contribution', orientation='h',
                 text_auto=True,
                 width=1000, height=600,
                 color_discrete_map={  # replaces default color mapping by value
                     "Feature in Post - Positive Impact": "#fb5c3e",
                     "Feature in Post - Negative Impact": "#fb5c3e",
                     "Feature not in Post - Positive Impact": "gray",
                     "Feature not in Post - Negative Impact": "gray"}
                 )
    fig.update_layout(showlegend=True, xaxis_title="SHAP Values")
    fig.update_layout(legend_traceorder="reversed")
    fig.update_traces(textposition='outside')
    fig.update_layout(uniformtext_minsize=5)
    # fig.update_xaxes(range=list([min(shap_df['Shap'] - 0.05), max(shap_df['Shap']) + 0.05]))
    st.plotly_chart(fig, use_container_width=True)
    st.write('(Features not in the graph represent zero impact in the model for this particular prediction)')

    #################################
    # Print Sentiment SHAP Analysis #
    #################################
    top5 = shap_df['Features_orig'][-6:]
    sentiment_feature = ['Sentiment_Negative', 'Sentiment_Neutral', 'Sentiment_Positive']
    imprimir = False
    for sent in sentiment_feature:
        for top in top5:
            if sent == top:
                print("igual")
                imprimir = True

    if imprimir:
        st.subheader('Sentiment Analysis')
        st.markdown(f'<h4 style="color:#fb5c3e;font-size:20px;">{"Sentiment of Post: " + sentiment_label}</h4>',
                    unsafe_allow_html=True)
        st.write(
            "The graph below shows the impact each word from the pre-processed "
            "text has on the model for Sentiment prediction.")

        fig = px.bar(shap_text_df_new, x=sentiment_label, y='Word',
                     color='Contribution', orientation='h',
                     text_auto=True,
                     width=200, height=500,
                     category_orders={  # replaces default order by column name
                         'Word': norm_text_list
                     },
                     color_discrete_map={  # replaces default color mapping by value
                         "Positive Impact": "#fb5c3e", "Negative Impact": "gray"
                     })
        fig.update_traces(textposition='outside')
        fig.update_layout(uniformtext_minsize=5, xaxis_title="Sentiment: " + sentiment_label + ' SHAP Values')
        fig.update_xaxes(
            range=list([min(shap_text_df_new[sentiment_label] - 0.05), max(shap_text_df_new[sentiment_label]) + 0.05]))
        st.plotly_chart(fig, use_container_width=True)
