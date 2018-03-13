import numpy as np
import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
from matplotlib.patches import PathPatch
from matplotlib.font_manager import FontProperties
from sklearn import linear_model
from sklearn import metrics

fp = FontProperties(family="Arial", weight="bold") 
globscale = 1.35
LETTERS = { "T" : TextPath((-0.305, 0), "T", size=1, prop=fp),
            "G" : TextPath((-0.384, 0), "G", size=1, prop=fp),
            "A" : TextPath((-0.35, 0), "A", size=1, prop=fp),
            "C" : TextPath((-0.366, 0), "C", size=1, prop=fp) }
COLOR_SCHEME = {'G': 'orange', 
                'A': 'red', 
                'C': 'blue', 
                'T': 'darkgreen'}

def letterAt(letter, x, y, yscale = 1, ax = None):
  text = LETTERS[letter]
  t = mpl.transforms.Affine2D().scale(1*globscale, yscale*globscale) + \
      mpl.transforms.Affine2D().translate(x,y) + ax.transData
  p = PathPatch(text, lw=0, fc=COLOR_SCHEME[letter],  transform=t)
  if ax != None:
    ax.add_artist(p)
  return p

def plot_seq_logo(logo_heights, name):
  logo_heights = np.reshape(logo_heights, (-1, 4))
  fig, ax = plt.subplots(figsize = (10, 3))
  x = 1
  maxi = 0
  mini = 0
  for i in range(logo_heights.shape[0]):
    list_of_heights = [('A', logo_heights[i, 0]), ('C', logo_heights[i, 1]), ('G', logo_heights[i, 2]), ('T', logo_heights[i, 3])]
    list_of_heights = sorted(list_of_heights, key = lambda x: x[1])
    y = np.sum(logo_heights[i][logo_heights[i] < 0])
    mini = min(y, mini)
    for base, score in list_of_heights:
      letterAt(base, x, y, abs(score), ax)
      y += abs(score)
    x += 1
    maxi = max(maxi, y)

  plt.xticks(range(1, x))
  plt.xlim((0, x)) 
  plt.ylim((mini, maxi)) 
  plt.tight_layout()      
  plt.savefig(name + 'SequenceLogos.png')
  plt.clf()

def sequenceLogos(dataset, plotName):
    datasetShape = dataset.shape
    positionWiseProbability = np.zeros((datasetShape[1], 4))
    for j in range(4):
        positionWiseProbability[:, j] = np.sum(dataset[:, :, j], axis = 0, dtype = float)/datasetShape[0]
    maxEntropy = -np.log(0.25)
    for j in range(datasetShape[1]):
        for k in range(4):
            positionWiseHeight = maxEntropy
            if(positionWiseProbability[j, k] != 0):
                positionWiseHeight -= positionWiseProbability[j, k]*np.log(positionWiseProbability[j, k])
        for k in range(4):
            positionWiseProbability[j, k] *= positionWiseHeight

    fig, ax = plt.subplots(figsize = (10, 3))
    x = 1
    maxi = 0
    mini = 0
    for i in range(positionWiseProbability.shape[0]):
      list_of_heights = [('A', positionWiseProbability[i, 0]), ('C', positionWiseProbability[i, 1]), ('G', positionWiseProbability[i, 2]), ('T', positionWiseProbability[i, 3])]
      list_of_heights = sorted(list_of_heights, key = lambda x: x[1])
      y = np.sum(positionWiseProbability[i][positionWiseProbability[i] < 0])
      mini = min(y, mini)
      for base, score in list_of_heights:
        letterAt(base, x, y, abs(score), ax)
        y += abs(score)
      x += 1
      maxi = max(maxi, y)

    plt.xticks(range(1, x))
    plt.xlim((0, x)) 
    plt.ylim((mini, maxi)) 
    plt.tight_layout()      
    plt.savefig(plotName + 'SequenceLogos.png')
    plt.clf()

def sequence_2_mer_generate(dataset):
    dataset_string_like = dataset[:, :, 0] + 2*dataset[:, :, 1] + 3*dataset[:, :, 2] + 4*dataset[:, :, 3]
    total_indices = np.arange(dataset.shape[1])
    even_indices = total_indices%2 == 0
    odd_indices = total_indices%2 == 1
    if(dataset.shape[1] %2 == 0):
        dataset_2_mer_1 = dataset_string_like[:, odd_indices] + 5*dataset_string_like[:, even_indices]
        dataset_2_mer_2 = 5*dataset_string_like[:, np.append(odd_indices[:-1], np.array([False]))] + dataset_string_like[:, np.append(np.array([False]), even_indices[1:])]
    else:
        dataset_2_mer_1 = dataset_string_like[:, odd_indices] + 5*dataset_string_like[:, np.append(even_indices[:-1], np.array([False]))]
        dataset_2_mer_2 = 5*dataset_string_like[:, odd_indices] + dataset_string_like[:, np.append(np.array([False]), even_indices[1:])]
    list_of_valid_words = [6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    freq_2_mer = np.zeros((dataset.shape[0], len(list_of_valid_words)), dtype = float)
    for i in range(len(list_of_valid_words)):
        dataset_2_mer_2_mod = dataset_2_mer_2 == list_of_valid_words[i]
        dataset_2_mer_1_mod = dataset_2_mer_1 == list_of_valid_words[i]
        for j in range(dataset.shape[0]):
            count_of_2_mer = 0
            for j0 in range(dataset_2_mer_1_mod.shape[1]):
                if(dataset_2_mer_1_mod[j, j0]):
                    count_of_2_mer += 1
                else:
                    count_of_2_mer = 0
                freq_2_mer[j, i] = max(freq_2_mer[j, i], count_of_2_mer)
            count_of_2_mer = 0
            for j0 in range(dataset_2_mer_2_mod.shape[1]):
                if(dataset_2_mer_2_mod[j, j0]):
                    count_of_2_mer += 1
                else:
                    count_of_2_mer = 0
                freq_2_mer[j, i] = max(freq_2_mer[j, i], count_of_2_mer)
    return freq_2_mer, freq_2_mer/dataset.shape[1]

def plot_seq_2_mer_freq(freq_val, plotName):
    freq_val = np.mean(freq_val, axis = 0)
    #list_of_valid_words = ['AA', 'AC', 'AG', 'AT', 'CA', 'CC', 'CG', 'CT', 'GA', 'GC', 'GG', 'GT', 'TA', 'TC', 'TG', 'TT']
    list_of_valid_words = [6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24]
    print freq_val
    plt.bar(list_of_valid_words, freq_val)
    plt.xlabel('2-mer')
    plt.ylabel('Average length of longest repeating sequence')
    plt.title('Longest repeating sequence for ' + plotName)
    plt.tight_layout()
    plt.savefig(plotName + '2merLongest.png')
    plt.clf()

def print_metrics_for_binned(testLabels, predictions, validIndices):
    print "Accuracy Score: %f" % metrics.accuracy_score(testLabels[validIndices], predictions[validIndices])
    print "F1 Score: %f" % metrics.f1_score(testLabels[validIndices], predictions[validIndices])
    print "ROCAUC Score: %f" % metrics.roc_auc_score(testLabels[validIndices], predictions[validIndices])
    fpr, tpr, thresholds = metrics.roc_curve(testLabels[validIndices], predictions[validIndices])
    print "PRAUC Score: %f" % metrics.average_precision_score(testLabels[validIndices], predictions[validIndices])
    print "Confusion Matrix:"
    print metrics.confusion_matrix(testLabels[validIndices], predictions[validIndices])

def logistic_regression_2_mer(freq_val, labels, trainIndex, testIndex):
    logReg = linear_model.LogisticRegression(C=1e5)
    logReg.fit(freq_val[trainIndex, :], labels[trainIndex])
    logRegPred = logReg.predict(freq_val[testIndex, :])
    testLabels = labels[testIndex]
    print_metrics_for_binned(testLabels, logRegPred, range(len(logRegPred)))
    return logReg
