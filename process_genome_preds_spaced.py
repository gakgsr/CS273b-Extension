# Process the mutation rate / mutation number predictions made by some model (e.g. simple_conv_predict)
# and compare the predicted values to the true values.

import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from sklearn import metrics
from scipy import stats
from sklearn import linear_model

def process_predictions(arr):
  bsize = 10000 # Size of the buckets that we aggregate predictions over
  sumprobs = True # Whether to make predictions based on summing the probabilites of each position being an indel over each bucket, or counting the number of positions whose predicted probability of being an indel is above some amount.
  arr2 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part, with a size divisible by bsize
  arr2.append(arr[(len(arr)//bsize)*bsize:]) # tack on the last piece

  thresh = 0.5 # Threshold above which an example is declared to be an indel, if we are thresholding rather than summing probabilities. Set higher since the classes are heavily imbalanced.
  avg_sum = 0
  avg_pred = 0
  sumcount = 0
  results = [[], []]
  probs = []
  small_window = 2*50 + 1
  small_window_size = 50
  for window in arr2:
    num_indels_true = np.sum(window[:, 1]) # Actual number of indels in the bucket
    avg_sum += num_indels_true
    sumcount += 1
    num_indels_pred = np.sum(window[window[:, 0]%small_window == small_window_size, 2] > thresh) # Predicted number of indels in the bucket, using thresholding
    sumprobs = np.sum(window[window[:, 0]%small_window == small_window_size, 2]) # Predicted number of indels in the bucket, using summing of probabilities
    avg_pred += num_indels_pred
    results[0].append(num_indels_true)
    results[1].append(num_indels_pred)
    probs.append(sumprobs)
  return results

arr1 = np.load("/datadrive/project_data/genomeIndelPredictionsValChrom.npy")
print arr1[:, 1], arr1[:, 2]
print np.sum(arr1[:, 1]), np.sum(arr1[:, 2])
pred_pos = arr1[arr1[:, 1] == 0.0, 2]
pred_pos_len = len(pred_pos)
print float(np.sum(pred_pos < 0.5))/pred_pos_len
arr2 = np.load("/datadrive/project_data/genomeIndelPredictionsValChrom.npy")

results = process_predictions(arr1)
#print results[0, :50], results[1, :50]

pred = results[1]
r, p = stats.pearsonr(results[0], pred) # Compute correlation between the model predictions from the above process, and the true values
print "On the validation set"
print(r)
print(p)
#print results[0, :50], results[1, :50]

# Linearly rescale predicted values to match true test labels (otherwise they will be way too high as the model was calibrated assuming a 50-50 indel vs nonindel split)
regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(results[0], axis=1), pred)


results = process_predictions(arr2)
reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

pred = results[1]
r, p = stats.pearsonr(results[0], reg_pred) # Compute correlation between the model predictions from the above process, and the true values
print "On the test set"
print(r)
print(p)

plt.scatter(results[0], pred)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-12}$)')
plt.plot(results[0], reg_pred, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred_Spaced.png')
