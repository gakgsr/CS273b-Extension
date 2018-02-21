'''Splits the selected chromosomes into contiguous "buckets" of a specified size and directly attempts to predict the number of
   indel mutations within each bucket. Uses two convolutional layers followed by two fully-connected layers'''

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
import load_coverage as lc
import load_recombination as lr

data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

# TODO: Can augment training set with overlapping windows (i.e. starting at random positions)

use_coverage = True
use_recombination = False # Doesn't currently work
forbidden_chroms = [1, 2]
validation_chrom = 5#np.random.choice(20) + 3
k = 200
window_size = 2*k+1
windows_per_bin = 50
margin = 15
expanded_window_size = window_size + 2*margin
batch_size = 50
num_train_ex = 250000
epochs = 12

num_indels = []
seq = []
for i in range(4,6):#(1, 24):
  if i in forbidden_chroms:
    continue
  if i == 23:
    ch = 'X'
    continue # Recombination data is in multiple files for chromosome X; we skip it for now
  else:
    ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr) # Total chromosome length
  num_windows = (c_len-2*margin)//window_size
  num_indels_ch = [0]*num_windows # True number of indels in each window

  insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
  deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
  indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1
  if use_coverage: coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(ch))
  if use_recombination: recombination = lr.load_recombination(data_dir + "recombination_map/genetic_map_chr{}_combined_b37.txt".format(ch)) # TODO: Why does this have fewer entries than the others...?

  for il in indelLocations:
    if il < margin: continue
    if il >= num_windows*window_size: break
    num_indels_ch[il // window_size] += 1

  if i == validation_chrom:
    num_indels_val = num_indels_ch
  else:
    num_indels.extend(num_indels_ch)
  del num_indels_ch, insertionLocations, deletionLocations, indelLocations # Preserve memory
  seq_ch = []
  for w in range(num_windows):
    # First window predictions start at index margin, but we include sequence context of length 'margin' around it, so its input array starts at index 0
    window_lb, window_ub = w*window_size, (w+1)*window_size + 2*margin # Include additional sequence context of length 'margin' around each window
    next_window = referenceChr[window_lb:window_ub]
    if use_coverage: next_window = np.concatenate((next_window, coverage[window_lb:window_ub]), axis=1)
    if use_recombination: next_window = np.concatenate((next_window, recombination[window_lb:window_ub]), axis=1)
    seq_ch.append(next_window)
  if i == validation_chrom:
    seq_val = seq_ch
  else:
    seq.extend(seq_ch)
  del seq_ch

del reference, referenceChr
if use_coverage: del coverage
if use_recombination: del recombination

order = [x for x in range(len(seq))]
random.shuffle(order)
seq = np.array([seq[i] for i in order]) # Shuffle the training data, so we can easily choose a random subset for testing
num_indels = np.array([num_indels[i] for i in order])

x_train = np.array(seq[:num_train_ex])
y_train = np.array(num_indels[:num_train_ex])
x_test = np.array(seq_val)
y_test = np.array(num_indels_val)

print('Mean # indels per window: {}'.format(float(sum(y_train))/len(y_train)))

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

use_lrelu = False
activation = 'linear' if use_lrelu else 'relu' # make linear to enable advanced activations
reg_param = 0.0001
reg = l2(reg_param)

model = Sequential()
# First convolutional layer. Input channels: [A,C,G,T] and coverage or recomb if applicable
model.add(Conv1D(40, kernel_size=5, activation=activation, input_shape=(expanded_window_size, 4 + use_coverage + use_recombination), kernel_regularizer=reg))
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Conv1D(100, kernel_size=8, activation=activation, kernel_regularizer=reg))
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(1024, activation=activation, kernel_regularizer=reg)) # FC hidden layer
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Dense(1, activation='relu', kernel_regularizer=reg)) # Output layer. ReLU activation because we are trying to predict a nonnegative value!

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse', 'mae']) # Minimize the MSE. Also report mean absolute error

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

# Predictions on the test set
y_pred = utils.flatten(model.predict(x_test, batch_size=batch_size, verbose=1))

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

plt.scatter(bin_trues, bin_preds)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation counts ($r = {:.2f}'.format(r) + (', p =$ {:.2g})'.format(p) if p else ', p < 10^{-15})$'))
line_x = np.arange(min(np.amax(bin_preds), np.amax(bin_trues)))
plt.plot(line_x, line_x, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred_keras.png')

np.save(data_dir + 'RegrKerasBinPred.npy', bin_preds)
np.save(data_dir + 'RegrKerasBinTrue.npy', bin_trues)
