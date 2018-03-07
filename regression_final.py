'''Splits the selected chromosomes into contiguous "buckets" of a specified size and directly attempts to predict the number of
   indel mutations within each bucket. Uses two convolutional layers followed by two fully-connected layers'''

# TODO: Compare correlation with that of random selected examples
# TODO: Can augment training set with overlapping windows (i.e. starting at random positions)
import time
start_time = time.time()
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from math import sqrt
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

#random.seed(1)
use_coverage = True
use_recombination = True
scalarize = True # When true, coverage and/or recombination are included as scalars instead
filter_indels = False
forbidden_chroms = [1, 2]
validation_chrom = 8#np.random.choice(20) + 3
k = 200
window_size = 2*k+1
windows_per_bin = 50
margin = 16
expanded_window_size = window_size + 2*margin
batch_size = 100
num_train_ex = 600000
epoch_val_frac = 0.1 # Fraction of examples to validate with after each epoch (out of total validation data)
epochs = 5

if not (use_coverage or use_recombination): scalarize = False

num_indels = []
seq = []
if scalarize: scalar_data = []
for i in range(1, 24):
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

  if filter_indels:
    indel_data_load = np.load(data_dir + "indelLocationsFiltered{}.npy".format(ch))
    indelLocations = indel_data_load[indel_data_load[:, 2] == 1, 0] - 1
    indelLocations = np.asarray(indelLocations, dtype=int)
    del indel_data_load
  else:
    insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
    deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
    indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1
    del insertionLocations, deletionLocations
  if use_coverage: coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(ch)) # Note: For chromosome 6? the coverage data length is actually 1 shorter than the chromosome length
  if use_recombination: recombination = lr.load_recombination(data_dir + "recombination_map/genetic_map_chr{}_combined_b37.txt".format(ch), c_len)

  for il in indelLocations:
    if il < margin: continue
    if il >= num_windows*window_size: break
    num_indels_ch[il // window_size] += 1

  if i == validation_chrom:
    num_indels_val = num_indels_ch
  else:
    num_indels.extend(num_indels_ch)
  del num_indels_ch, indelLocations # Preserve memory

  seq_ch = []
  if scalarize: scalar_data_ch = []
  for w in range(num_windows):
    # First window predictions start at index margin, but we include sequence context of length 'margin' around it, so its input array starts at index 0
    window_lb, window_ub = w*window_size, (w+1)*window_size + 2*margin # Include additional sequence context of length 'margin' around each window
    next_window = referenceChr[window_lb:window_ub]
    if scalarize: scalar_datum = []
    if use_coverage:
      if scalarize:
        scalar_datum.append(np.mean(coverage[window_lb:window_ub]))
      else:
        next_window = np.concatenate((next_window, coverage[window_lb:window_ub]), axis=1)
    if use_recombination:
      if scalarize:
        scalar_datum.append(np.mean(recombination[window_lb:window_ub]))
      else:
        next_window = np.concatenate((next_window, recombination[window_lb:window_ub]), axis=1)
    seq_ch.append(next_window)
    if scalarize: scalar_data_ch.append(scalar_datum)
  if i == validation_chrom:
    seq_val = seq_ch
    if scalarize: scalar_data_val = scalar_data_ch
  else:
    seq.extend(seq_ch)
    if scalarize: scalar_data.extend(scalar_data_ch)
  del seq_ch
  if use_coverage: del coverage
  if use_recombination: del recombination

del reference, referenceChr

order = [x for x in range(len(seq))]
random.shuffle(order)
seq = np.asarray([seq[i] for i in order]) # Shuffle the training data, so we can easily choose a random subset for testing
if scalarize: scalar_data = np.asarray([scalar_data[i] for i in order])
num_indels = np.asarray([num_indels[i] for i in order])

x_train = np.asarray(seq[:num_train_ex])
y_train = np.asarray(num_indels[:num_train_ex])
x_test = np.asarray(seq_val)
y_test = np.asarray(num_indels_val)
if scalarize:
  s_train = scalar_data[:num_train_ex]
  s_test = np.asarray(scalar_data_val)
epoch_val_indices = np.random.choice(len(x_test), int(len(x_test) * epoch_val_frac), replace=False)

print('Mean # indels per window: {}'.format(float(sum(y_train))/len(y_train)))

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Dropout, Flatten, Lambda, Merge
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential
from keras.callbacks import *
import warnings

use_lrelu = False
activation = 'linear' if use_lrelu else 'relu' # make linear to enable advanced activations
reg_param = 0.0001
reg = l2(reg_param)

model = Sequential()
# First convolutional layer. Input channels: [A,C,G,T] and coverage or recomb if applicable
model.add(Conv1D(40, kernel_size=5, activation=activation, input_shape=(expanded_window_size, 4 + (not scalarize)*(use_coverage + use_recombination)), kernel_regularizer=reg))
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Conv1D(100, kernel_size=8, activation=activation, kernel_regularizer=reg))
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
model.add(Dense(1024, activation=activation, kernel_regularizer=reg)) # FC hidden layer 1
if use_lrelu: model.add(LeakyReLU(alpha=0.1))
if scalarize: # Add the scalar data in right before the output layer
  scalar_branch = Sequential()
  Identity = Lambda(lambda x: x + 0, input_shape=(use_coverage + use_recombination,))
  scalar_branch.add(Identity) # Hack to avoid doing any transformation on the scalar data
  def Concatenate(x):
    with warnings.catch_warnings(): # keras.layers.Merge is deprecated
      warnings.simplefilter('ignore')
      return Merge(x, mode='concat')
  merged = Concatenate([model, scalar_branch])
  final_model = Sequential()
  final_model.add(merged)
else:
  final_model = model
final_model.add(Dense(1, activation='relu')) # Output layer. ReLU activation because we are trying to predict a nonnegative value!

def rms(y_true, y_pred): return keras.backend.sqrt(keras.losses.mean_squared_error(y_true, y_pred)) # TODO: For some reason rms is totally different than sqrt(mse)... why??

final_model.compile(loss=keras.losses.mean_squared_error,
                    optimizer=keras.optimizers.Adam(),
                    metrics=['mse', 'mae']) # Minimize the MSE. Report mean absolute error and mean squared error.

setup_time = time.time()
print('Total setup time: {} sec'.format(setup_time - start_time))

logfile = 'KerasTrainingLog.csv'
if scalarize:
  train_input = [x_train, s_train]
  test_input = [x_test, s_test]
  val_input = ([x_test[epoch_val_indices], s_test[epoch_val_indices]], y_test[epoch_val_indices])
else:
  train_input = x_train
  test_input = x_test # This is really validation also, but we aren't "testing" yet
  val_input = (x_test[epoch_val_indices], y_test[epoch_val_indices])
final_model.fit(train_input, y_train,
                batch_size=batch_size,
                epochs=epochs,
                verbose=1,
                validation_data=val_input,
                callbacks=[CSVLogger(logfile)])

train_time = time.time()
print('Total training time: {} sec'.format(train_time - setup_time))

if scalarize:
  output_weights = np.squeeze(final_model.get_layer(index=-1).get_weights()[0]) # get_weights returns [weights, biases]
  # Print linear regression coefficients
  ind = -1
  if use_recombination:
    print('Recombination coefficient: {}'.format(output_weights[ind]))
    ind -= 1
  if use_coverage:
    print('Coverage coefficient: {}'.format(output_weights[ind]))

# Predictions on the test set
y_pred = utils.flatten(final_model.predict(test_input, batch_size=batch_size, verbose=1))
np.save(data_dir + 'RegrKerasWindowPred.npy', y_test)
np.save(data_dir + 'RegrKerasWindowPred.npy', y_pred)

from scipy import stats
from sklearn import linear_model

# Compute the correlation between the test set predictions and the true values
r, p = stats.pearsonr(y_test, y_pred)
outstr = '\nWindow predictions r value: {}, p value: {}'.format(r, p)
with open(logfile, 'a') as f: f.write(outstr + '\n')

bin_preds, bin_trues = [], []
for i in range(len(y_test)//windows_per_bin):
  pred_agg, true_agg = 0, 0
  for j in range(i*windows_per_bin, i*windows_per_bin + windows_per_bin):
    pred_agg += y_pred[j]
    true_agg += y_test[j]
  bin_preds.append(pred_agg)
  bin_trues.append(true_agg)

bin_preds, bin_trues = np.asarray(bin_preds), np.asarray(bin_trues)

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

test_time = time.time()
print('Total testing time: {} sec'.format(test_time - train_time))
print('Overall time: {} sec'.format(test_time - start_time))
