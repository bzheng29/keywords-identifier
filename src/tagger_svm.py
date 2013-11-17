from classifier import Classifier

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def multiLabelClassifier(X, Y, ngram_range=None):
    if ngram_range:
        vectorizer = CountVectorizer(ngram_range=ngram_range,
                                     token_pattern=r'\b\w+\b',
                                     min_df=1)
    else:
        vectorizer = CountVectorizer(min_df=1)

    classifier = Pipeline([
        ('vectorizer', CountVectorizer()),
        ('tfidf', TfidfTransformer()), # reduces the importance of words that are very frequent, like "the"
        ('clf', OneVsRestClassifier(LinearSVC())),
    ])
    return classifier

class SVMMultiLabelClassifier(Classifier):
    def __init__(self, X, Y, ngram_range=None):
        self.classifier = multiLabelClassifier(X, Y, ngram_range)
        self.X = X
        self.Y = Y

    def train(self):
        self.classifier.fit(self.X, self.Y)

    def test(self, X):
        return classifier.predict(X)
