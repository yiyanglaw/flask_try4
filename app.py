from flask import Flask, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import re

app = Flask(__name__)

# Load the data for training
data = pd.read_csv('m_data2.csv')
data.columns = ['url', 'label']
data['label'] = data['label'].apply(lambda x: 1 if x == 'bad' else 0)

# Preprocess the data
X = data['url'].values
y = data['label'].values

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a custom transformer to convert URLs to character-level n-gram representations
class URLToNgram(BaseEstimator, TransformerMixin):
    def __init__(self, ngram_range=(2, 3)):
        self.ngram_range = ngram_range
        self.vectorizer = CountVectorizer(analyzer='char', ngram_range=ngram_range)

    def fit(self, X, y=None):
        self.vectorizer.fit(X)
        return self

    def transform(self, X, y=None):
        return self.vectorizer.transform(X)

# Define a pipeline for training the model
pipeline = Pipeline([
    ('url_to_ngram', URLToNgram()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='log', alpha=1e-5, penalty='elasticnet', max_iter=100))
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Function to preprocess URLs
def preprocess_url(url):
    # Remove common prefixes like "http://", "https://", and "www."
    url = re.sub(r'^https?://|www.', '', url)
    return url

@app.route('/predict_url', methods=['POST'])
def predict_url():
    url = request.form['url']
    url = preprocess_url(url)
    prediction = pipeline.predict([url])[0]
    if prediction == 1:
        result = 'Bad URL'
    else:
        result = 'Good URL'
    return result

app.run(host='0.0.0.0', port=10000, debug=False)

