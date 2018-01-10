# Process the mutation rate / mutation number predictions made by some model (e.g. simple_conv_predict)
# and compare the predicted values to the true values.

import numpy as np

chrom = 21
#arr = np.load("/datadrive/project_data/genomeIndelPredictions{}.npy".format(chrom))
arr = np.load("/datadrive/project_data/genomeIndelPredictionsUnSpaced.npy")

bsize = 10000 # Size of the buckets that we aggregate predictions over
sumprobs = True # Whether to make predictions based on summing the probabilites of each position being an indel over each bucket, or counting the number of positions whose predicted probability of being an indel is above some amount.
arr2 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part, with a size divisible by bsize
arr2.append(arr[(len(arr)//bsize)*bsize:]) # tack on the last piece

thresh = 0.8 # Threshold above which an example is declared to be an indel, if we are thresholding rather than summing probabilities. Set higher since the classes are heavily imbalanced.
avg_sum = 0
avg_pred = 0
sumcount = 0
results = [[], []]
probs = []
for window in arr2:
  num_indels_true = np.sum(window[:, 1]) # Actual number of indels in the bucket
  avg_sum += num_indels_true
  sumcount += 1
  num_indels_pred = np.sum(window[:, 2] > thresh) # Predicted number of indels in the bucket, using thresholding
  sumprobs = np.sum(window[:, 2]) # Predicted number of indels in the bucket, using summing of probabilities
  avg_pred += num_indels_pred
  results[0].append(num_indels_true)
  results[1].append(num_indels_pred)
  probs.append(sumprobs)

results[1] = np.array(results[1]) * avg_sum / avg_pred # Rescale to bring predictions around the same order of magnitude as true number of indels

print(results[0][:50])
print(results[1][:50])

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
#from sklearn import metrics
from scipy import stats
from sklearn import linear_model

#r2 = metrics.r2_score(results[0], results[1]) #this gives bizarre values?? TODO

# we should really do this trained on only some of the datapoints? FIXED. Also, can easily change to use sumprobs rather than thresholding.
pred = results[1]
print(pred[:50])
r, p = stats.pearsonr(results[0], pred) # Compute correlation between the model predictions from the above process, and the true values
print(r)
print(p)

# Linearly rescale predicted values to match true test labels (otherwise they will be way too high as the model was calibrated assuming a 50-50 indel vs nonindel split)
regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(results[0], axis=1), pred)
reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

plt.scatter(results[0], pred)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-12}$)')
plt.plot(results[0], reg_pred, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred_unspaced.png')
