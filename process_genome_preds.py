# Process the mutation rate / mutation number predictions made by some model (e.g. simple_conv_predict)
# and compare the predicted values to the true values.

import numpy as np

chrom = 21
arr = np.load("/datadrive/project_data/genomeIndelPredictionsUnSpaced{}.npy".format(chrom))

sumprobs = True # Whether to make predictions based on summing the probabilites of each position being an indel over each bucket, or counting the number of positions whose predicted probability of being an indel is above some amount.

print('Loaded true and predicted indel locations') # arr[:,1] and arr[:,2], respectively (arr[:,0] contains the indices)

bsize = 10000 # Size of the buckets that we aggregate predictions over
arr2 = np.array_split(arr[:(len(arr)//bsize)*bsize], len(arr)//bsize) # split the first part, with a size divisible by bsize

thresh = 0.8 # Threshold above which an example is declared to be an indel (if we are thresholding rather than summing probabilities). Set higher since the classes are heavily imbalanced.
avg_sum = 0
avg_pred = 0
sumcount = 0
results = [[], []]
probs = []
for window in arr2:
  num_indels_true = np.sum(window[:, 1]) # Actual number of indels in the bucket
  avg_sum += num_indels_true
  sumcount += 1
  if sumprobs:
    num_indels_pred = np.sum(window[:, 2]) # Predicted number of indels in the bucket, using summing of probabilities
  else: # Threshold and count instead
    num_indels_pred = np.sum(window[:, 2] > thresh) # Predicted number of indels in the bucket, using thresholding
  avg_pred += num_indels_pred
  results[0].append(num_indels_true)
  results[1].append(num_indels_pred)
  probs.append(sumprobs)

results[1] = np.array(results[1]) * avg_sum / avg_pred # Rescale to bring predictions around the same order of magnitude as true number of indels

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import linear_model

#r2 = metrics.r2_score(results[0], results[1]) # This gives bizarre values?? TODO: investigate

pred = results[1]
r, p = stats.pearsonr(results[0], pred) # Compute correlation between the model predictions from the above process, and the true values
print('r value: {}'.format(r))
print('p value: {}'.format(p))

# Linearly rescale predicted values to match true test labels (otherwise they will be way too high as the model was calibrated assuming a 50-50 indel vs nonindel split)
regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(results[0], axis=1), pred)
reg_pred = regr.predict(np.expand_dims(results[0], axis=1))

plt.scatter(results[0], pred)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation counts ($r = {:.2f}'.format(r) + (', p =$ {:.2g})'.format(p) if p else ', p < 10^{-15})$'))
plt.plot(results[0], reg_pred, color='m', linewidth=2.5)
descr = 'summed' if sumprobs else 'thresh_{}'.format(thresh)
plt.savefig('indel_rate_pred_{}_unspaced_{}.png'.format(chrom, descr))
