import pandas as pd
import numpy as np
import string
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords as sw
from nltk.corpus import wordnet as wn
from nltk import WordNetLemmatizer
from nltk import sent_tokenize
from nltk import pos_tag
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')
from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath, table_name='PreppedDataTable'):
    """Load cleaned data from database into dataframe.
    Args:
        database_filepath: String, file path for the table
        table_name: String. no space,  ends with .db
    Returns:
       X: numpy.ndarray. Disaster messages.
       Y: numpy.ndarray. Disaster categories for each messages.
       category_name: list. Disaster category names.
    """
    engine = create_engine('sqlite:///'+ database_filepath)
    df = pd.read_sql_table(table_name, engine)
    category_names = df.columns[4:]
    X = df[['message']].values[:, 0]
    y = df[category_names].values

    return X, y, category_names

ef tokenize(text, lemmatizer = WordNetLemmatizer()):
        """
        Returns a normalized, lemmatized list of tokens from a document by
        applying segmentation (breaking into sentences), then word/punctuation
        tokenization, and finally part of speech tagging. It uses the part of
        speech tags to look up the lemma in WordNet, and returns the lowercase
        version of all the words, removing stopwords and punctuation.
        """
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

def build_model():
   ''''
   This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

   ''''
   pipeline= Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))),
        ])
   parameters = {
    'classifier__estimator__n_neighbors':list(range(3,10)),
     'classifier__estimator__weights':['uniform', 'distance']}

   cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)
   return cv


def evaluate_model(model, X_test, Y_test, category_names):
    """Evaluate model
    Args:
        model: sklearn.model_selection.GridSearchCV.  It contains a sklearn estimator.
        X_test: numpy.ndarray. Disaster messages.
        Y_test: numpy.ndarray. Disaster categories for each messages
        category_names: Disaster category names.
    """
    y_pred = model.predict(X_test)

    # Print accuracy, precision, recall and f1_score for each categories
    i = 0
    for item in Y_test.columns:
        print(classification_report(Y_test[item],y_pred[:,i]))
        i+=1

def save_model(model, model_filepath):
    """Save model
    Args:
        model: sklearn.model_selection.GridSearchCV. It contains a sklearn estimator.
        model_filepath: String. Trained model is saved as pickel into this file.
    """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)
        
        
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
