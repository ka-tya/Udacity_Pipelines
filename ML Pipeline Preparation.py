#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries
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


# In[2]:


# load data from database
engine = create_engine('sqlite:///InsertDatabaseName.db')
df = pd.read_sql_table('InsertTableName', engine)
X = df['message']
Y = df.drop(columns = ['id','original', 'message', 'genre'])


# In[3]:


#check for nulls in X and Y
print(X.isnull().sum())
print(Y.isnull().sum()/Y.shape[0])
print('X shape is {}, Y shape is {}'.format(X.shape,Y.shape))


# In[4]:


# drop NAs in the Y
Y = Y.dropna(how = 'all', axis = 0)
#drop same rows in X
X = X[0:Y.shape[0]]


# ### 2. Write a tokenization function to process your text data

# In[5]:


def tokenize(text, lemmatizer = WordNetLemmatizer()):
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


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[6]:


pipeline= Pipeline([
            ('vectorizer', CountVectorizer(tokenizer=tokenize)),
            ('tfidf', TfidfTransformer()),
            ('classifier', MultiOutputClassifier(KNeighborsClassifier(n_neighbors=3))),
        ])


# In[7]:


pipeline.get_params().keys()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[8]:


X_train,X_test,y_train,y_test = train_test_split(X, Y, test_size=0.5, random_state = 33)


# In[9]:


pipeline.fit(X_train, y_train)


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[10]:


y_pred  = pipeline.predict(X_test)


# In[11]:


def report(y_pred,y_original):
    i = 0
    target_names = ['class 0', 'class 1', 'class 2']
    for item in y_test.columns:
        print(classification_report(y_original[item],y_pred[:,i], target_names=target_names))
        i +=1


# In[12]:


report(y_pred,y_test)


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


parameters = {
'classifier__estimator__n_neighbors':list(range(3,10)),
'classifier__estimator__weights':['uniform', 'distance']
}

cv = GridSearchCV(pipeline, param_grid=parameters, cv=2, n_jobs=-1, verbose=3)

cv


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:


cv.fit(X_train,y_train)
print(cv.best_params_)


# In[15]:


cv_pred  = cv.predict(X_test)


# In[16]:


cv_report = report(cv_pred,y_test)


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[17]:


# try a different classifier
pipeline2 = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(AdaBoostClassifier()))
])
pipeline2.fit(X_train, y_train)
y_pred2  = pipeline2.predict(X_test)


# In[18]:


report(y_pred2,y_test)


# In[19]:


pipeline2.get_params().keys()


# In[20]:


parameters2 = {'clf__estimator__learning_rate': [0.1, 0.3],
            'clf__estimator__n_estimators': [10, 50, 100]
             }
cv2 = GridSearchCV(pipeline2, param_grid=parameters2, cv=2, n_jobs=-1, verbose=3)

cv2


# In[21]:


cv2.fit(X_train,y_train)


# In[25]:


print(cv2.best_params_)


# In[22]:


cv2_pred = cv2.predict(X_test)


# In[23]:


report(cv2_pred,y_test)


# ### 9. Export your model as a pickle file

# In[24]:


with open('model.pkl', 'wb') as f:
    pickle.dump(cv2, f)


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# ### Note to self:
#     To further improve the quality of the output, the set needs to be more balanced out. With some classes more prevalent than the others, the prediction can be skewed
