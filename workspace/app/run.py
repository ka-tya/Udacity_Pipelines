import json
import plotly

import sys
import os
import pandas as pd
import numpy as np
import string
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords as sw
from nltk import WordNetLemmatizer
from nltk import sent_tokenize


from nltk import pos_tag
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger', 'stopwords'])
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle
import re
from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text, lemmatizer = WordNetLemmatizer()):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
        # find URLs
        url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        found_urls = re.findall(url_regex, text)
        for url in found_urls:
            text = text.replace(url, 'urlplaceholder')
    
        tokens = []
        # Break the document into sentences
        for sent in sent_tokenize(text):
            # Break the sentence into part of speech tagged tokens
            for token, tag in pos_tag(word_tokenize(sent)):
                # Apply preprocessing to the token
                token = token.lower() 
                token = token.strip()
                token = token.strip('_')
                token = token.strip('*')

                # If punctuation or stopword, ignore token and continue
                if token in set(sw.words('english')) or all(char in string.punctuation for char in token):
                    continue
                
                tokens.append(token)
        # Lemmatize the token and yield
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
        return tokens


# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')  
df = pd.read_sql_table('PreppedDataTable', engine)


# load model
model = joblib.load("../workspace/models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    # extract data needed for visuals
    #plot 1
    classes = df.columns[4:] 
    class_counts = df[classes].sum().sort_values(ascending = False)
    labels  = class_counts.index
    #plot 2
    #most common words for the social genre
    word_list = []
    for message in df[df.genre == 'social'].message:
        word_list.append(tokenize(message))
    word_counts = pd.Series(word_list).apply(pd.Series).stack().value_counts().head(10)
    word_labels = word_counts.index
    #create visuals
    # TODO: Below is an example - modify to crea your own visls
    graphs = [
         { 'data': [
                Bar(
                  x=labels,
                  y =class_counts
              )
          ],

            'layout': {
                'title': 'Distribution of Message Classes',
                'yaxis': {
                  'title': "Count"
              },
                 'xaxis': {
                 'title': "Class"
              }
            }
      },
        {
            'data': [
                Bar(
                    x=word_labels,
                    y=word_counts
                )
            ],
            'layout': {
                'title': 'Top 10 Most common words in Social Genre',
                'yaxis': {
                    'title': "Counts"
                },
                'xaxis': {
                    'title': "Word"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
   
