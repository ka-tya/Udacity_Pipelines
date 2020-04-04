# Udacity Disaster Response Pipelines
_______________________________________________________________________________
This is the second project for the Data Science NanoDegree on Udacity platform.
The idea behind this project is to be able to quickly identify a topic of a real human written text, using an multi label classifier.
### The goal for this projects were:
* Learn Natural Language Processing
* Learn how no create clean reusable solution for ML classifier using Pipelines
* Lean how to automate the project as a web app

### Natural Language Processing
The raw text supplied to the tool is cleaned, normalized and tokenzed by breaking sentences into single words.

### Pipelines 
To create a clean and reusable solution a pipeline is created. Pipelines are powerful helper of python science package sk-learn.
It helps chain multiple steps together and have a clean reusable peace of code to simplify a search for the best machine learning algorythm.

### Web App
Finally to simplify the user experience even further, a web app is created to automate the run of the solution.

## Instructions
_____________________________________________________________________________________________
### Required libraries
* pandas
* numpy
* string
* sqlalchemy
* nltk
* sklearn
* pickle

## Files
- ETL Pipeline Preparation.ipynb: Description for workspace/data/process_data.py
- ML Pipeline Preparation.ipynb: Description for workspace/model/train_classifier.py

##### workspace/data/process_data.py: A script that preps and stores the data through the following steps:
* Load the messages and categories datasets
* Clean the data
* Store output in a database
##### workspace/model/train_classifier.py: A script that processes and classifies text into categories through the following steps:
* Load data from the SQLite database
* Split the dataset into training and test sets
* Build a text cleansing and machine learning pipeline
* Tune a model using GridSearchCV
* Apply results on the test set
* Exports the final model as a pickle file
