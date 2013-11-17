
# WORK IN PROGRESS

def sparseVectorDotProduct(v1, v2):
    # smaller vector,
    smallerVector = []
    largerVector = []

    if len(v1) <= len(v2):
        smallerVector = v1
        largerVector = v2
    else:
        largerVector = v1
        smallerVector = v2

    # iterate over the keys
    dot_product = 0
    for key in smallerVector.keys():
        dot_product += smallerVector[key] * largerVector[key]
    return dot_product

class LinearPredictor():
    def __init__(self, labels, featureFunction, params):
        """
        @param (string, string): Pair of positive, negative labels
        @param func featureFunction: function to featurize text, e.g. extractUnigramFeatures
        @param dict params: the parameter weights used to predict
        """
        self.featureFunction = featureFunction
        self.params = defaultdict(int, params)

    def predict(self, x):
        """
        @param string x: the text message
        @return double y: classification score (e.g. tagCount)
        """
        # BEGIN_YOUR_CODE (around 2 lines of code expected)
        v1 = self.featureFunction(x)
        return sparseVectorDotProduct(v1, self.params)
        # END_YOUR_CODE

def learnWeightsFromPerceptron(trainExamples, featureExtractor, labels, iters = 20):
    """
    @param list trainExamples: list of (x,y) pairs, where
      - x is a string representing the question body
      - y is a string representing the label ('ham' or 'spam')
    @params func featureExtractor: Function to extract features, e.g. extractUnigramFeatures
    @params labels: tuple of labels ('positive', 'negative'), e.g. ('spam', 'ham').
    @params iters: Number of training iterations to run.
    @return dict: parameters represented by a mapping from feature (string) to value.
    """
    positiveLabel = labels[0]
    weights = defaultdict(int)
    featuresList = []
    for index in range(len(trainExamples)):
        features = featureExtractor(trainExamples[index][0]) # this gives the word count vector. just have this as a parameter from the getgo
        featuresList.append(features)
    for iteration in range(iters):
        for index in range(len(trainExamples)):
            message = trainExamples[index][0]
            guessValue = sparseVectorDotProduct(featuresList[index], weights)
            if guessValue >= 0 and trainExamples[index][1] != positiveLabel:
                keys = featuresList[index].keys()
                for key in keys:
                    weights[key] -= featuresList[index][key]
            elif guessValue < 0 and trainExamples[index][1] == positiveLabel:
                keys = set(message.split())
                for key in keys:
                    weights[key] += featuresList[index][key]
    return weights

# normalize based on length of question
def predictTagCount(linearPredictor, x):
    prediction = linearPredictor.predict(x)
    return prediction # TODO normalized by weight


