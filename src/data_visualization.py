import matplotlib.pyplot as plt

def printPrediction(X, predicted):
    labelsArray = []
    count = 0
    for item, labels in zip(X, predicted):
        if len(labels) > 0:
            count += 1
            print '%s\n=> %s\n\n' % (item, ', '.join(labels))
            labelsArray.append(labels)
    print "Out of %d predictions, %d of them have labels." % (len(predicted), count)
    print "====================================================="
    for labels in labelsArray:
        print repr(labels)

def printResultsTable(frequencies, numPredictions):
    print ""
    print "------------------"
    print "0:       %.2f" % ( 100 * (float(frequencies[0]) / float(numPredictions)))
    print "1-25:    %.3f" % ( 100 * (float(sum(frequencies[1:25])) / float(numPredictions)))
    print "25-50:   %.3f" % ( 100 * (float(sum(frequencies[25:50])) / float(numPredictions)))
    print "50-75:   %.3f" % ( 100 * (float(sum(frequencies[50:75])) / float(numPredictions)))
    print "75-100:  %.3f" % ( 100 * (float(sum(frequencies[75:100])) / float(numPredictions)))
    print "------------------"
    print ""

def plotPrediction(predicted, Y_train):
    print "There are %d predictions" % len(predicted)
    X_labels = 100*[0]
    for i in range(100):
        X_labels[i] = str(i)
    frequencies = [0]*100
    totalNumTags = 0
    totalTagsCorrect = 0
    for index, prediction in enumerate(predicted):
        numTags = max(len(Y_train[index]), len(predicted[index]))
        totalNumTags += numTags
        correctCount = 0
        for i in range(len(predicted[index])):
            if predicted[index][i] in Y_train[index]:
                correctCount += 1
        totalTagsCorrect += correctCount
        percentCorrect = float(correctCount) / float(numTags)
        bucketIndex = int(100*percentCorrect)
        if bucketIndex == 100:
            bucketIndex = 99 # put 100% in the last bucket
        frequencies[bucketIndex] += 1
    printResultsTable(frequencies, len(predicted))
    print "%.3f percent of tags are correct" % (100.0 * (float(totalTagsCorrect) / float(totalNumTags)))
    pos = np.arange(len(X_labels))
    width = 1.0 # gives histogram aspect to the bar diagram
    ax = plt.axes()
    ax.set_xticks(pos + (width / 2)) # center the ticks
    ax.set_xticklabels(X_labels)
    plt.bar(pos, frequencies, width, color='r')
    plt.show()
    # TODO the histogram of predictions that predicted too many tags

def printPercentMostFrequentWords(percent, vocab):
    partition = int((float(percent)/100.0) * len(vocab))
    for word in vocab[0:partition]:
        print word

def computeErrorRate(examples, classifier):
    """
    @param list examples
    @param dict params: parameters
    @param function predict: (params, x) => y
    @return float errorRate: fraction of examples we make a mistake on.
    """
    numErrors = 0
    for x, y in examples:
        if classifier.classifyWithLabel(x) != y:
            numErrors += 1
    return 1.0 * numErrors / len(examples)

def countTags(dataset, suffix):
    tags_count = collections.defaultdict(lambda: 0)
    for example in dataset:
        tags = example[-1]
        for tag in tags.split():
            tags_count[tag] += 1;
    total_count = sum(tags_count.values());
    with open("tags_count_" + suffix, 'w') as f:
        print >> f, "total_count: ", total_count
        sorted_tags = sorted(tags_count.items(), key=lambda x:x[1], reverse=1)
        for tag, count in sorted_tags:
            print >> f, '{0:40} : {1:10} : {2:.2f}%'\
                    .format(tag, count, count / float(total_count) * 100)
