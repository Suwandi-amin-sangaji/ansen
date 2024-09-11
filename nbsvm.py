import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report

default_C_param = {'linear': 1.0, 'poly': 1.0, 'sigmoid': 1.0, 'rbf': 1.0}

class NBSVM:
    def __init__(self):
        self.nb_model = None
        self.svm_model = None
        self.vectorizer = TfidfVectorizer()

    def fit(self, X_train, y_train, C=1.0, kernel='linear'):
        # Step 1: Vectorize text data
        X_train_tfidf = self.vectorizer.fit_transform(X_train)

        # Step 2: Train Naive Bayes model
        self.nb_model = MultinomialNB(alpha=1.0)
        nb_probs = self.nb_model.fit(X_train_tfidf, y_train).predict_proba(X_train_tfidf)

        # Step 3: Use Naive Bayes predictions as features for SVM
        X_train_nbsvm = nb_probs

        # Step 4: Train SVM using Naive Bayes-transformed features
        self.svm_model = SVC(C=C, kernel=kernel, probability=True)
        self.svm_model.fit(X_train_nbsvm, y_train)

    def predict(self, X_test):
        # Step 1: Vectorize test data
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Step 2: Get Naive Bayes probabilities
        nb_probs = self.nb_model.predict_proba(X_test_tfidf)

        # Step 3: Use SVM to make predictions based on Naive Bayes probabilities
        return self.svm_model.predict(nb_probs)
