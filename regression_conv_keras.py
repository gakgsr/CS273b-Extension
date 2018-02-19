'''Splits the selected chromosomes into contiguous "buckets" of a specified size and directly attempts to predict the number of
   indel mutations within each bucket. Uses a very simple convolutional layer followed by two fully-connected layers'''

# TODO: Compare correlation with that of random selected examples

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import tensorflow as tf
import numpy as np
import random
from sys import argv
import utils
import cs273b

np.random.seed(1)

data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

# TODO: Can augment training set with overlapping windows (i.e. starting at random positions)
forbidden_chroms = [1, 2]
validation_chrom = np.random.choice(19) + 3
print "Validation chromosome is %d" % validation_chrom
k = 200
window_size = 2*k+1
windows_per_bin = 50
margin = 15
expanded_window_size = window_size + 2*margin
batch_size = 50
num_train_ex = 500000
epochs = 12
complexity_thresh = 1.1

num_indels = []
seq = []
for i in range(1, 23):
  if i in forbidden_chroms:
    continue
  if i == 23:
    ch = 'X'
  else:
    ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr) # Total chromosome length
  num_windows = (c_len-2*margin)//window_size
  num_indels_ch = [0]*num_windows # True number of indels in each window

  #insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
  #deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
  #indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1
  indel_data_load = np.load(data_dir + "indelLocationsFiltered" + ch + ".npy")
  indelLocations = indel_data_load[np.logical_and(indel_data_load[:, 2] == 1, indel_data_load[:, 4] > complexity_thresh), 0] - 1
  indelLocations = np.array(indelLocations, dtype = int)
  del indel_data_load

  for il in indelLocations:
    if il < margin: continue
    if il >= num_windows*window_size: break
    num_indels_ch[il // window_size] += 1

  if i == validation_chrom:
    num_indels_val = num_indels_ch
  else:
    num_indels.extend(num_indels_ch)
  #del num_indels_ch, insertionLocations, deletionLocations, indelLocations # Preserve memory
  del num_indels_ch, indelLocations # Preserve memory
  seq_ch = []
  for w in range(num_windows):
    # First window predictions start at index margin, but we include sequence context of length 'margin' around it, so its input array starts at index 0
    window_lb, window_ub = w*window_size, (w+1)*window_size + 2*margin # Include additional sequence context of length 'margin' around each window
    seq_ch.append(referenceChr[window_lb:window_ub])
  if i == validation_chrom:
    seq_val = seq_ch
  else:
    seq.extend(seq_ch)
  del seq_ch

del reference, referenceChr

order = [x for x in range(len(seq))]
random.shuffle(order)
seq = np.array([seq[i] for i in order]) # Shuffle the training data, so we can easily choose a random subset for testing
num_indels = np.array([num_indels[i] for i in order])

x_train = np.array(seq[:num_train_ex])
y_train = np.array(num_indels[:num_train_ex])
x_test = np.array(seq_val)
y_test = np.array(num_indels_val)

#np.save(data_dir + 'RegrKerasTestSeq' + str(validation_chrom) + str(complexity_thresh) +  '.npy', x_test)
np.save(data_dir + 'RegrKerasTestLab' + str(validation_chrom) + str(complexity_thresh) + '.npy', y_test)

print('Mean # indels per window: {}'.format(float(sum(y_train))/len(y_train)))

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

use_lrelu = False
activation = 'linear' if use_lrelu else 'relu' # make linear to enable advanced activations

model = Sequential()
model.add(Conv1D(40, kernel_size=5, activation=activation, input_shape=(expanded_window_size, 4)))#, kernel_regularizer=l2(0.0001))) # Convolutional layer
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Conv1D(100, kernel_size=8, activation=activation))
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(1024, activation=activation))#, kernel_regularizer=l2(0.01))) # FC hidden layer
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1, activation='relu')) # Output layer. ReLU activation because we are trying to predict a nonnegative value!

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse', 'mae']) # Minimize the MSE. Also report mean absolute error

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

# Predictions on the test set
y_pred = utils.flatten(model.predict(x_test, batch_size=batch_size, verbose=1))
np.save(data_dir + 'RegrKerasTestLabPred' + str(validation_chrom) + str(complexity_thresh) + '.npy', y_pred)
#model.save(data_dir + 'RegrKerasModel' + str(validation_chrom) + str(complexity_thresh) + '.h5')

from scipy import stats
from sklearn import linear_model

# Compute the correlation between the test set predictions and the true values
r, p = stats.pearsonr(y_test, y_pred)
print('')
print('r value: {}'.format(r))
print('p value: {}'.format(p))

bin_preds, bin_trues = [], []
for i in range(len(y_test)//windows_per_bin):
  pred_agg, true_agg = 0, 0
  for j in range(i*windows_per_bin, i*windows_per_bin + windows_per_bin):
    pred_agg += y_pred[j]
    true_agg += y_test[j]
  bin_preds.append(pred_agg)
  bin_trues.append(true_agg)

bin_preds, bin_trues = np.array(bin_preds), np.array(bin_trues)

mae = np.mean(np.abs(bin_preds - bin_trues))
rms = np.sqrt(np.mean(np.square(bin_preds - bin_trues)))
r_bin, p_bin = stats.pearsonr(bin_trues, bin_preds)
avg_pred = np.mean(bin_preds)
avg_true = np.mean(bin_trues)

print('Bin size: {}, MAE: {}, RMS error: {}, r: {}, p-value: {}, average indels predicted: {}, average indels actual: {}'.format(windows_per_bin*window_size, mae, rms, r_bin, p_bin, avg_pred, avg_true))


regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(bin_trues, axis=1), bin_preds)
reg_pred = regr.predict(np.expand_dims(bin_trues, axis=1))

plt.scatter(bin_trues, bin_preds)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation counts ($r = {:.2f}'.format(r) + (', p =$ {:.2g})'.format(p) if p else ', p < 10^{-15})$'))
line_x = np.arange(min(np.amax(bin_preds), np.amax(bin_trues)))
plt.plot(line_x, line_x, color='m', linewidth=2.5)
plt.plot(bin_trues, reg_pred, color='g', linewidth=2.5)
plt.savefig('indel_rate_pred_keras' + str(validation_chrom) + str(complexity_thresh) + '.png')

#np.save(data_dir + 'RegrKerasBinPred' + str(validation_chrom) + '.npy', bin_preds)
#np.save(data_dir + 'RegrKerasBinTrue' + str(validation_chrom) + '.npy', bin_trues)
