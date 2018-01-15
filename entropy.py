import numpy as np
#import random
from sklearn import linear_model
from sklearn import metrics
#import matplotlib
#matplotlib.use('agg')
#import matplotlib.pyplot as plt

# input is numpy array of size batchSize * (2k_0 + 1) * 4, where k_0<k is a smaller window around the gene
# output is a numpy array of size batchSize * 1, where each location contains the entropy of that sequence
def entropySequence(sequenceStrings):
    lenSequenceString = sequenceStrings.shape
    entropyString = np.zeros(lenSequenceString[0])
    for j in range(4):
        probLetter = np.sum(sequenceStrings[:, :, j], axis = 1, dtype = float)/lenSequenceString[1]
        logProbLetter = np.log(probLetter)
        logProbLetter[logProbLetter == -np.Inf] = 0
        entropyString -= np.multiply(probLetter, logProbLetter)
    return entropyString


# input is numpy array of size batchSize * (2k + 1) * 4, ie. the entire batched dataset
# output is a numpy array of size batchSize * k
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
def entropyVector(sequenceStrings):
    lenSequenceString = sequenceStrings.shape
    centr = int((lenSequenceString[1]-1)/2)
    entropyVal = np.zeros((lenSequenceString[0], centr))
    for j in range(centr + 1, lenSequenceString[1]):
        entropyVal[:, j - centr - 1] = entropySequence(sequenceStrings[:, range((2*centr-j), (j+1)), :])
    return entropyVal



# print metrics on test data and predictions
def print_metrics_for_binned(testLabels, predictions, validIndices):
    print "Accuracy Score: %f" % metrics.accuracy_score(testLabels[validIndices], predictions[validIndices])
    print "F1 Score: %f" % metrics.f1_score(testLabels[validIndices], predictions[validIndices])
    print "ROCAUC Score: %f" % metrics.roc_auc_score(testLabels[validIndices], predictions[validIndices])
    fpr, tpr, thresholds = metrics.roc_curve(testLabels[validIndices], predictions[validIndices])
    print "PRAUC Score: %f" % metrics.average_precision_score(testLabels[validIndices], predictions[validIndices])
    print "Confusion Matrix:"
    print metrics.confusion_matrix(testLabels[validIndices], predictions[validIndices])



# input is numpy array of size batchSize * k (entropyStrings), numpy array of size batchSize * 1 (labels), trainIndex, testIndex
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
# prints metrics for logistic regression
def logisticRegression(entropyStrings, labels, trainIndex, testIndex, testAC=None):
    logReg = linear_model.LogisticRegression(C=1e5)
    logReg.fit(entropyStrings[trainIndex, :], labels[trainIndex])
    logRegPred = logReg.predict(entropyStrings[testIndex, :])
    testLabels = labels[testIndex]
    testEntropy = entropyStrings[:, -1]
    testEntropy = entropyStrings[testIndex, -1]
    print_metrics_for_binned(testLabels, logRegPred, range(len(logRegPred)))
    # Procedure to print model accuracy in different complexity bins
    # Not needed at the moment
    '''
    minEntropy = [0.9, 1.0, 1.1, 1.2, 1.3]
    maxEntropy = [1.0, 1.1, 1.2, 1.3, 1.4]
    print "Model Prediction in different entropy bins\n"
    for minval, maxval in zip(minEntropy, maxEntropy):
        print "%f <= Entropy < %f" % (minval, maxval)
        validIndices = np.logical_and(testEntropy >= minval, testEntropy < maxval)
        print_metrics_for_binned(testLabels, logRegPred, validIndices)
    print "End of Model Prediction in different entropy bins\n"
    print "Model Prediction in AC = 1 set\n"
    validIndices = np.logical_or(testAC == 1, testLabels == 0)
    print_metrics_for_binned(testLabels, logRegPred, validIndices)
    print "Model Prediction in AC > 1 set\n"
    validIndices = np.logical_or(testAC > 1, testLabels == 0)
    print_metrics_for_binned(testLabels, logRegPred, validIndices)
    '''