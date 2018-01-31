# Process the mutation rate / mutation number predictions made by some model (e.g. simple_conv_predict)
# and compare the predicted values to the true values.

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from sklearn import metrics
from scipy import stats
from sklearn import linear_model

def process_predictions(arr, small_window, small_window_size, bsize):
  arr3 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part, with a size divisible by bsize
  #arr2.append(arr[(len(arr)//bsize)*bsize:]) # tack on the last piece

  thresh = 0.5 # Threshold above which an example is declared to be an indel

  results = [[], []]

  for window in arr3:
    num_indels_true = np.sum(window[:, 1]) # Actual number of indels in the bucket
    num_indels_pred = np.sum(window[window[:, 0]%small_window == small_window_size, 2] > thresh) # Predicted number of indels in the bucket, using thresholding
    results[0].append(num_indels_true)
    results[1].append(num_indels_pred)
    avg_acc += metrics.accuracy_score(window[:, 1], window[:, 2].round())
  avg_acc /= len(arr3)
  print "Average accuracy for bsize %d is %f" % (bsize, avg_acc)

  '''
  adjacent_corr = []
  for i in range(len(arr3) - 1):
    window1 = arr3[i]
    window2 = arr3[i+1]
    r, p = stats.pearsonr(window1[:, 2], window2[:, 2])
    adjacent_corr.append(r)

  plt.hist(adjacent_corr)
  plt.title('Histogram of correlation values for adjacent bins of size {}'.format(bsize))
  plt.savefig('hist_adj_corr_{}.png'.format(bsize))
  plt.clf()
  '''

  return results

small_window = 2*20 + 1
small_window_size = 20

arr1 = np.load("/datadrive/project_data/genomeIndelPredictionsValChrom.npy")
arr2 = np.load("/datadrive/project_data/genomeIndelPredictionsTestChrom.npy")
#arr1 = np.load("/datadrive/project_data/genomeIndelPredictionsValChromEntrpy.npy")
#arr2 = np.load("/datadrive/project_data/genomeIndelPredictionsTestChromEntrpy.npy")


bsize_val = [10000, 20000, 50000]
for bsize in bsize_val:
  print "Bsize = {}".format(bsize)
  print "Adjacent non-overlapping window"
  results = process_predictions(arr1, small_window, small_window_size, bsize)

  print "On the validation set"
  print stats.pearsonr(results[0], results[1]) # Compute correlation between the model predictions from the above process, and the true values

  # Linearly rescale predicted values to match true test labels, since we didn't sum over all locations
  regr = linear_model.LinearRegression()
  regr.fit(np.expand_dims(results[0], axis=1), results[1])


  results = process_predictions(arr2, small_window, small_window_size, bsize)
  reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

  print "On the test set"
  print stats.pearsonr(results[1], reg_pred) # Compute correlation between the model predictions from the above process, and the true values
  print stats.pearsonr(results[1], results[0])
  print stats.pearsonr(results[1][2:150], reg_pred[2:150])
  print stats.pearsonr(results[1][2:150], results[0][2:150])
  print metrics.mean_squared_error(results[1], results[0])
  print metrics.mean_squared_error(results[1][2:150], results[0][2:150])
  plt.hist(results[1])
  plt.title('Histogram of predicted indels in bin size {}'.format(bsize))
  plt.savefig('hist_pred_indel_spaced_{}.png'.format(bsize))
  plt.clf()

  plt.hist(results[0])
  plt.title('Histogram of true indels in bin size {}'.format(bsize))
  plt.savefig('hist_true_indel_spaced_{}.png'.format(bsize))
  plt.clf()

  print "All locations"
  results = process_predictions(arr1, 1, 0, bsize)

  print "On the validation set"
  print stats.pearsonr(results[0], results[1]) # Compute correlation between the model predictions from the above process, and the true values

  # Linearly rescale predicted values to match true test labels, since we didn't sum over all locations
  regr = linear_model.LinearRegression()
  regr.fit(np.expand_dims(results[0], axis=1), results[1])

  results = process_predictions(arr2, 1, 0, bsize)
  reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

  print "On the test set"
  print stats.pearsonr(results[1], reg_pred) # Compute correlation between the model predictions from the above process, and the true values
  print stats.pearsonr(results[1], results[0])
  print stats.pearsonr(results[1][2:150], reg_pred[2:150])
  print stats.pearsonr(results[1][2:150], results[0][2:150])
  print metrics.mean_squared_error(results[1], results[0])
  print metrics.mean_squared_error(results[1][2:150], results[0][2:150])
  plt.hist(results[1])
  plt.title('Histogram of predicted indels in bin size {}'.format(bsize))
  plt.savefig('hist_pred_indel_{}.png'.format(bsize))
  plt.clf()

  plt.hist(results[0])
  plt.title('Histogram of true indels in bin size {}'.format(bsize))
  plt.savefig('hist_true_indel_{}.png'.format(bsize))
  plt.clf()

'''
plt.plot(range(2000), arr1[2000:4000, 2])
plt.plot(range(2000), arr1[2000:4000, 1], color='r')
plt.title('Variation of predicted probabilities and true labels on validation chromosome')
plt.savefig('var_pred_prob.png')
#plt.savefig('var_pred_prob_entrpy.png')
plt.clf()

bsize = 10000
results = process_predictions(arr1, small_window, small_window_size, bsize)
regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(results[0], axis=1), results[1])
results = process_predictions(arr2, small_window, small_window_size, bsize)
reg_pred = regr.predict(np.expand_dims(results[0], axis=1))


plt.scatter(results[0], results[1])
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels without correction')
plt.savefig('true_vs_pred.png')
#plt.savefig('true_vs_pred_entrpy.png')
plt.clf()

#plt.scatter(results[1], reg_pred)
#plt.xlabel('True number of indels')
#plt.ylabel('Predicted number of indels with correction')
#plt.savefig('true_vs_pred_corr.png')
#plt.clf()
'''