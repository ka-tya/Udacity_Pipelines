import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = []
    lemmatizer = WordNetLemmatizer()

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

# load data from database
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('CleanedTableName', engine)

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
