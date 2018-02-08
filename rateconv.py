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

data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

bsize = 5000 # Size of each bucket over which we predict the total number of indels

num_indels = []
seq = []
for i in range(22,23):#(1, 24):
  if i == 23:
    ch = 'X'
  else:
    ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr) # Total chromosome length
  
  insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
  deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
  indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1
  num_buckets = c_len//bsize
  num_indels_ch = [0]*num_buckets # True number of indels in each bucket
  for il in indelLocations:
    if il//bsize >= len(num_indels_ch): break
    num_indels_ch[il // bsize] += 1
  num_indels.extend(num_indels_ch)
  del num_indels_ch, insertionLocations, deletionLocations, indelLocations # Preserve memory
  seq_ch = np.array_split(referenceChr[:num_buckets*bsize], num_buckets) # Discard the last chunk of the chromosome, if it is smaller than a normal bucket
  seq.extend(seq_ch)

del reference, referenceChr, seq_ch

from random import shuffle

order = [x for x in range(0, len(seq))]
shuffle(order)
seq = np.array([seq[i] for i in order])
num_indels = np.array([num_indels[i] for i in order]) # Shuffle the buckets data, so we can easily choose a random subset for testing

ntest = len(seq) // 6 # Number of testing examples (currently set to 1/6 of the total)
x_train = seq[:ntest]
y_train = num_indels[:ntest]
x_test = seq[ntest:]
y_test = num_indels[ntest:]

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

model = Sequential()
model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=(bsize, 4)))#, kernel_regularizer=l2(0.0001))) # Convolutional layer
model.add(Flatten())
model.add(Dense(500, activation='relu'))#, kernel_regularizer=l2(0.01))) # FC hidden layer
model.add(Dense(1, activation='relu')) # Output layer. ReLU activation because we are trying to predict a nonnegative value!

model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.Adam(),
              metrics=['mse', 'mae']) # Minimize the MSE. Also report mean absolute error

batch_size = 5 # Need to use small batch size when buckets are very large, due to memory limitations
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=6,
          verbose=1)
          #validation_data=(x_test, y_test))

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# Predictions on the test set
ytestout = utils.flatten(model.predict(x_test, batch_size=5, verbose=1))
#print('')
#print(list(zip(y_test, ytestout)))

from scipy import stats
from sklearn import linear_model

# Compute the correlation between the test set predictions and the true values
r, p = stats.pearsonr(y_test, ytestout)
print(r)
print(p)

# Scatterplot of predicted vs. true values
plt.scatter(y_test, ytestout)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-10}$)')
plt.plot(y_test, y_test, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred.png')
