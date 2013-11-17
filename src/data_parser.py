from HTMLParser import HTMLParser
import csv
import re
import numpy as np

# The data files were generated via split -l 355100 Train.csv
# That leaves partial CSV entries on the top and bottom of
# each file, so delete these partial entries by hand before loading
# the CSV files
# returns (trainingSet, tags)
# where tags are all the tags in the training set

# works for both training dataset and testing dataset
def loadDataSet(filename):
    tags = set()
    with open('../data/train_data/' + filename, 'rb') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for row in reader:
            yield row
    # in xzz there are 3980 tags, 1694 of which are unique

class MyHTMLParser(HTMLParser):
    def __init__(self):
        self.reset()
        self.HTMLDATA = []
        self.CODEDATA = []
        self.code_tag_opened = False
    def handle_data(self, d):
        if self.code_tag_opened:
            self.CODEDATA.append(d)
        self.HTMLDATA.append(d)
    def handle_starttag(self, tag, attrs):
        if tag == 'code':
            self.code_tag_opened = True
    def handle_endtag(self, tag):
        if tag == 'code':
            self.code_tag_opened = False
    def get_data(self):
        return ''.join(self.HTMLDATA)

def stripHTMLTags(html):
    s = MyHTMLParser()
    s.feed(html)
    return s.get_data()

def stripNewlines(str):
    return re.sub(r'\n', '', str)

# TODO length of coding segments would probably inform the
# tag count classifier, because, for example, really long
# code segments could be indicative of a lower number of
# tags, because newbies post long code segments and
# don't know how to use tags well.

def getDataAndCodeSegments(body):
    """
    @param body string - the question body
    @return (body list string, codeSegments list string)
    """
    # find the <code> starting tag.
    # put all the text until the </code>
    # ending tag in the cadingSegments list
    parser = MyHTMLParser()
    parser.feed(body)
    return (parser.get_data(), parser.CODEDATA)

def sanitizeTrainingSet(trainingSet):
    """
    Returns X, Y, and a list of coding segments removed
    from the question body.
    """
    X = []
    Y = []
    codeSegments = []
    for example in trainingSet:
        qid, title, body, tags = example
        title = stripHTMLTags(title)
        body = stripNewlines(body)
        body, questionCodeSegments = getDataAndCodeSegments(body)
        X.append(title + ' ' + body)
        codeSegments.append(questionCodeSegments)
        Y.append(tags.split())
    X = np.array(X)
    Y = np.array(Y)
    return(X, codeSegments, Y)

def sanitizeTestingSet(testingSet):
    # each entry in the trainingSet looks like this
    # ['996004',
    #  '.NET Dictionary: is only enumerating thread safe?',
    #  '<p>Is simply enumerating a .NET Dictionary from multiple threads safe? </p>\n\n<p>No modification of the Dictionary takes place at all.</p>\n',
    #  '.net multithreading dictionary thread-safety'
    # ]
    X = []
    Y = []
    for example in testingSet:
        qid, title, body, tags = example
        title = stripHTMLTags(title)
        body = stripNewlines(body)
        body = stripHTMLTags(body)
        X.append(title + ' ' + body)
        Y.append(tags.split())
    X = np.array(X)
    Y = np.array(Y)
    return (X, Y)

def removePercentMostFrequentWords(percent, sorted_vocab_list, vocab):
    partition = int((float(percent)/100.0) * len(vocab))
    for word in sorted_vocab_list[0:partition]:
        del vocab[word[0]]

def purify(word):
    """
    remove punctuations/digits, lower case word
    """
    punc = set(string.punctuation)
    digit = set(string.digits)
    filt = punc.union(digit)
    word = "".join(ch for ch in word if ch not in filt)
    word = word.lower()
    return word

def getAllTags(y):
    """
    Generates the set of all tags in y
    @param y list list string - observed tags
    @return set string - all tags
    """
    return list(set([tag for l in Y_train for tag in l]))
