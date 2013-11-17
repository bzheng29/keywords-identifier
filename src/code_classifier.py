def trainCodeClassifier(X_code_segments_train, Y_train):
    """
    NB implementation. The code classifier differs from the classifier
    that operates on the entire text body, in that it uses more domain
    knowledge about code. That is, perhaps it can identify a python dictionary
    and add the tag "dictionary".
    @param X_code_segments_train list list string - code segments per question
    @param Y_train list list string - tags
    @return languageDetector lambda string: [string] - a classifier that returns
             a list of tags
    """
    codeClassifier = lambda x: ["javascript", "c++", "dictionary"]
    return codeClassifier
