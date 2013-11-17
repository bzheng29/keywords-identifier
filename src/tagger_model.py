import collections
import math

class TaggerModel:
    def __init__(self,
                 features,
                 featureExtractor,
                 labels):
        self.features = features
        self.featureExtractor = featureExtractor,
        self.labels = labels
