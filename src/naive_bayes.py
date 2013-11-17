from tagger_model import TaggerModel
from classifier import Classifier

class NaiveBayesMultiLabelClassifier(Classifier):
    def __init__(self, labels, X, Y, featureExtractor=None):
        self.labels = labels
        self.X = map(featureExtractor, X) if featureExtractor is not None else X
        self.Y = Y
        self.prob_feature_given_label = {}
        self.prob_label = {}

    def train(self):
        """
        The output is a list of p(tag|feature) for each tag and feature
        p(tag | feature) = p(feature | label) * p (label) / p(feature)
        and with laplace smoothing: # TODO
        p(tag | feature) = (p(feature | label) * p (label) + 1) / p(feature) + len(features)

        The denominator is the same when comparing between labels, so I leave it out.

        """
        for label in self.labels:
            training_samples_with_label = []
            for i in range(len(self.X)):
                if label in self.Y[i]:
                    training_samples_with_label.append(X[i])

            # p(feature | label) = in all training samples that have that label, how many have the feature
            #                               /
            #                      the sum of all features in training samples that have that label
            # denom
            # for example, the number of all words in questions with the label
            # numerator
            # for example, an element in this list is the number of times this word appeared
            # in questions with the label.
            prob_feature_given_label[label] = sum(training_samples_with_label) \
                                            / sum(sum(training_samples_with_label))

            # p(label)
            prob_label[label] = float(len(training_samples_with_label)) / float(len(X))

    def test(self, X, K=None):
        """
        @param X - test set examples
        @param K list int - number of labels per training example
        @return labels list list string - the labels predicted, per training sample
        """
        if self.featureExtractor is not None:
            X = map(self.featureExtractor, X)
        predictions = []
        for training_sample in range(X):
            label_probabilities = []
            for label in self.labels:
                label_probabilities.append(log(label_prob[label] + \
                     log(tagger_math.sparseVectorDotProduct(
                        self.prob_feature_given_tag, self.training_sample
                         )  ))) # TODO np
            pdb.set_trace()
            # sort label_probabilities
            if K is not None:
                predictions.append(k) # the top K predictions
            else:
                raise "Not implemented yet"
        return predictions

def convertDataForNaiveBayes(X, vocab):
    """
    @param X list list string - a list of training questions
    @param vocab dictionary string : int - keys are all the words in the vocabulary
                                           each values is a word index in the dict
    @return list list int - a design matrix containing word frequencies
                            Rows correspond to questions.
                            Columns correspond to words in the vocabulary.

    converts X_train to a list of lists. Each row represents a question,
    and is composed of word frequencies

    Can be used for X_train or X_test
    """
    # each row contains a dictionary of word frequencies.
    rows = []
    for x in X:
        row = [0] * len(vocab)
        for word in x.split():
            word = data_parser.purify(word)
            if word == "":
                continue
            if not vocab.has_key(word):
                continue
            row[vocab[word]] += 1
        rows.append(row)

    return rows

def createVocabularyForNaiveBayes(X_train):
    """
    Create a vocabulary (dictionary) for all the words appeared.
    Used for multinomial Naive Bayes.

    Filters out stopwords.

    NOTE: Vocabulary doens't have to be generated this way.
          Can come up with better vocabulary. (say filter out some words)
    """
    vocab = defaultdict(int)
    for x in X_train:
        # trivial word split and purification
        for word in x.split():
            word = data_parser.purify(word)
            vocab[word] +=1
    del vocab[""]
    # create a sorted list of ordered tuples, ordered by word frequencies
    sorted_vocab_list = sorted(vocab.items(), key=lambda x:x[1], reverse=True)
    # take out, say, the top 2% most frequent words.
    removePercentMostFrequentWords(2, sorted_vocab_list, vocab)
    return dict(zip(vocab, range(len(vocab))))


def createLabelList(labels):
    """
    Turn the labels into dictionaries with label:index for building nb training matrix
    """
    return dict(zip(labels, range(len(labels))))

    #TODO use classfication/regression to predict k (#tags) for each question

