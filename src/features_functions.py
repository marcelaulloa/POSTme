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
import numpy as np

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

def feat_cat(x):
    if x == '':
        return 'Feature not in Post'
    else:
        return 'Feature in Post'

def class_shap(x):
      if x < 0:
        return 'Negative Impact'
      else:
        return 'Positive Impact'