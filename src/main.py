import numpy as np
import pdb
import re
from collections import defaultdict

import tagger_model
import tagger_svm
import naive_bayes
import perceptron
import data_parser
import data_visualization
import feature_extractors

def predictTagCount():
    """
    Predicts the number of tags this question has
    Naive Bayes implementation. A question can have
    a maximum of 5 tags

    P(tagCount = k | question) = P (question | tagCount

    @return tagCount int - predicted tag count
    """

    return tagCount

# TODO I ran the classifier on the whole training set for ten minutes
# until I ran into this error: UnicodeDecodeError: 'ascii' codec can't
# decode byte 0xe2 in position 45: ordinal not in range(128).
if __name__ == '__main__':
    training_set = data_parser.loadDataSet('out0')
    X_train, codeSegments, Y_train = data_parser.sanitizeTrainingSet(training_set)
    pdb.set_trace()
    tags = generateTagsForNaiveBayes(Y_train)
    pdb.set_trace()
    model = tagger_model.TaggerModel(
        features = X_train_nb,
        featureExtractor = feature_extractors.unigrams,
        labels = tags,
    )

    #prob_feature_given_tag_list, prob_tag_list = trainNaiveBayes(model)


    # might want to reduce the size of vocabulary
    print "tags count: ", len(tags)
    print "words count: ", len(vocab)
    print "questions count: ", len(X_train_nb)
    print "time complexity: ", len(tags) * len(vocab) * len(X_train_nb)
    print "Parsed the training data"

    testingSet = data_parser.loadDataSet('out1')
    X_test, Y_test = sanitizeTestingSet(testingSet)
    X_test_nb = convertDataForNaiveBayes(X_test, vocab)
    testNaiveBayes(prob_feature_given_tag_list, prob_tag_list, X_test_nb, len(vocab), tags)

    #testingSet = data_parser.loadTrainingSet('xwi')
    #predicted = testNaiveBayes(X_train, Y_train, testingSet)
    #plotPrediction(predicted, Y_test)

