# Process the mutation rate / mutation number predictions made by some model (e.g. simple_conv_predict)
# and compare the predicted values to the true values.

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from sklearn import metrics
from scipy import stats
from sklearn import linear_model

def process_predictions(arr, small_window, small_window_size, bsize):
  arr2 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part, with a size divisible by bsize
  arr2.append(arr[(len(arr)//bsize)*bsize:]) # tack on the last piece

  thresh = 0.8 # Threshold above which an example is declared to be an indel

  results = [[], []]

  for window in arr2:
    num_indels_true = np.sum(window[:, 1]) # Actual number of indels in the bucket
    num_indels_pred = np.sum(window[window[:, 0]%small_window == small_window_size, 2] > thresh) # Predicted number of indels in the bucket, using thresholding
    results[0].append(num_indels_true)
    results[1].append(num_indels_pred)
  return results

small_window = 2*50 + 1
small_window_size = 50

arr1 = np.load("/datadrive/project_data/genomeIndelPredictionsValChrom.npy")
arr2 = np.load("/datadrive/project_data/genomeIndelPredictionsTestChrom.npy")
#arr1 = np.load("/datadrive/project_data/genomeIndelPredictionsValChromEntrpy.npy")
#arr2 = np.load("/datadrive/project_data/genomeIndelPredictionsTestChromEntrpy.npy")


bsize_val = [10000, 20000, 50000]
for bsize in bsize_val:
  print "Bsize = {}".format(bsize)
  print "Adjacent non-overlapping window"
  results = process_predictions(arr1, small_window, small_window_size, bsize)

  r, p = stats.pearsonr(results[0], results[1]) # Compute correlation between the model predictions from the above process, and the true values
  print "On the validation set"
  print(r)
  print(p)

  # Linearly rescale predicted values to match true test labels, since we didn't sum over all locations
  regr = linear_model.LinearRegression()
  regr.fit(np.expand_dims(results[0], axis=1), results[1])


  results = process_predictions(arr2, small_window, small_window_size, bsize)
  reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

  r, p = stats.pearsonr(results[1], reg_pred) # Compute correlation between the model predictions from the above process, and the true values
  print "On the test set"
  print(r)
  print(p)

  print "All locations"
  results = process_predictions(arr1, 1, 0, bsize)

  r, p = stats.pearsonr(results[0], results[1]) # Compute correlation between the model predictions from the above process, and the true values
  print "On the validation set"
  print(r)
  print(p)

  # Linearly rescale predicted values to match true test labels, since we didn't sum over all locations
  regr = linear_model.LinearRegression()
  regr.fit(np.expand_dims(results[0], axis=1), results[1])

  results = process_predictions(arr2, 1, 0, bsize)
  reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

  r, p = stats.pearsonr(results[1], reg_pred) # Compute correlation between the model predictions from the above process, and the true values
  print "On the test set"
  print(r)
  print(p)

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
