import numpy as np
import random
from sklearn import linear_model
from sklearn import metrics
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

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
    #print "True Positive Rate: %f" % tpr
    #print "False Positive Rate: %f" % fpr
    print "PRAUC Score: %f" % metrics.average_precision_score(testLabels[validIndices], predictions[validIndices])
    print "Confusion Matrix:"
    print metrics.confusion_matrix(testLabels[validIndices], predictions[validIndices])



# input is numpy array of size batchSize * k (entropyStrings), numpy array of size batchSize * 1 (labels), trainIndex, testIndex
# element i, j contains the entropy of sequence i for a window of size 2*j + 1 around its center
# prints metrics for logistic regression
def logisticRegression(entropyStrings, labels, trainIndex, testIndex, plot_complexity=False, testAC=None):
    logReg = linear_model.LogisticRegression(C=1e5)
    logReg.fit(entropyStrings[trainIndex, :], labels[trainIndex])
    logRegPred = logReg.predict(entropyStrings[testIndex, :])
    testLabels = labels[testIndex]
    testEntropy = entropyStrings[:, -1]
    plt.hist(testEntropy[testAC == 1])
    plt.title('Histogram of sequence complexity for AC = 1')
    plt.savefig('ComplexityAC1.png')
    plt.clf()
    plt.hist(testEntropy[testAC > 1])
    plt.title('Histogram of sequence complexity for AC > 1')
    plt.savefig('ComplexityACG1.png')
    plt.clf()
    plt.hist(np.log(testEntropy[testAC == 1]))
    plt.title('Histogram of log of sequence complexity for AC = 1')
    plt.savefig('ComplexityAC1log.png')
    plt.clf()
    plt.hist(np.log(testEntropy[testAC > 1]))
    plt.title('Histogram of log of sequence complexity for AC > 1')
    plt.savefig('ComplexityACG1log.png')
    plt.clf()
    testAC = testAC[testIndex]
    testEntropy = entropyStrings[testIndex, -1]
    print_metrics_for_binned(testLabels, logRegPred, range(len(logRegPred)))
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

    #k = entropyStrings.shape[1]
    #plt.plot(range(1, k+1), logReg.coef_[0, :])
    #plt.xlabel('Window Size')
    #plt.ylabel('Regression Coefficient')
    #plt.title('Variation of Regression Coefficient with Window Size')
    #plt.savefig('LogRegCoeff.png')
    #plt.clf()
    #plt.plot(range(1, k+1), np.fabs(logReg.coef_[0, :]))
    #plt.xlabel('Window Size')
    #plt.ylabel('Absolute Regression Coefficient')
    #plt.title('Variation of Absolute Regression Coefficient with Window Size')
    #plt.savefig('LogRegCoeffAbs.png')
    #plt.clf()
    #np.save("/datadrive/project_data/LogRegCoeff.npy", logReg.coef_[0, :])
    if(plot_complexity):
        testLabels = labels[testIndex]
        testEntropy = entropyStrings[testIndex, -1]
        plt.hist(testEntropy[np.logical_and(testLabels == 1, logRegPred == 1)])
        plt.title('Histogram of sequence entropy for true positives')
        plt.savefig('EntropyComplexityTruePositives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 0, logRegPred == 1)])
        plt.title('Histogram of sequence entropy for false positives')
        plt.savefig('EntropyComplexityFalsePositives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 0, logRegPred == 0)])
        plt.title('Histogram of sequence entropy for true negatives')
        plt.savefig('EntropyComplexityTrueNegatives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 1, logRegPred == 0)])
        plt.title('Histogram of sequence entropy for false negatives')
        plt.savefig('EntropyComplexityFalseNegatives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 1, logRegPred == 1), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for true positives')
        plt.savefig('EntropyComplexityTruePositivesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 0, logRegPred == 1), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for false positives')
        plt.savefig('EntropyComplexityFalsePositivesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 0, logRegPred == 0), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for true negatives')
        plt.savefig('EntropyComplexityTrueNegativesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 1, logRegPred == 0), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for false negatives')
        plt.savefig('EntropyComplexityFalseNegativesGEQ1.png')
        plt.clf()
