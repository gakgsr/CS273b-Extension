import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

import numpy as np
import utils
import cs273b
from random import shuffle

from scipy import stats
from sklearn import metrics

import keras
from keras.regularizers import l2
from keras.layers import Conv1D, Dense, Flatten, Dropout
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential

# Loading reference chromosome and setting random seed
np.random.seed(1)
data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

# Size of each bucket over which we predict the total number of indels
bsize = 10000
# Test on a separate chromosome, hold out chromosome 1, train on the rest
test_chrom = np.random.choice(range(2, 23), 1, replace=False)
print "Test chromosome is %d" % test_chrom

# Populate train and test data
train_num_indels = []
train_seq = []
test_num_indels = []
test_seq = []
for i in range(2,23):
  ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr)

  # Load indel data
  indel_data_load = np.load(data_dir + "indelLocationsFiltered" + str(i) + ".npy")
  # Filter by sequence complexity and filter value around 20 sized window and complexity threshold
  indelLocations = indel_data_load[indel_data_load[:, 2] == 1, 0] - 1

  num_buckets = min(5000, c_len//bsize)
  num_indels_ch = [0]*num_buckets
  for il in indelLocations:
    if il//bsize >= num_buckets: break
    num_indels_ch[il// bsize] += 1
  del indelLocations, indel_data_load

  seq_ch = np.array_split(referenceChr[:num_buckets*bsize], num_buckets)
  if i == test_chrom:
    test_seq.extend(seq_ch)
    test_num_indels.extend(num_indels_ch)
  else:
    train_seq.extend(seq_ch)
    train_num_indels.extend(num_indels_ch)

  del num_indels_ch, seq_ch

del reference, referenceChr

# Shuffle the data
order = [x for x in range(0, len(train_seq))]
shuffle(order)

x_train = np.array([train_seq[i] for i in order])
y_train = np.array([train_num_indels[i] for i in order])
x_test = np.array(test_seq)
y_test = np.array(test_num_indels)

del train_num_indels, test_num_indels, train_seq, test_seq


# Neural network model
model = Sequential()
# Convolutional layer
model.add(Conv1D(10, kernel_size=5, activation='relu', input_shape=(bsize, 4)))#, kernel_regularizer=l2(0.0001)))
model.add(Flatten())
# FC hidden layers
model.add(Dense(500, activation='relu'))#, kernel_regularizer=l2(0.01)))
model.add(Dense(200, activation='relu'))
# Output layer. ReLU activation because we are trying to predict a nonnegative value!
model.add(Dense(1, activation = 'relu'))

# Minimize the MSE. Also report mean absolute error
model.compile(loss = keras.losses.mean_squared_error, optimizer = keras.optimizers.Adam(), metrics = ['mae'])

# Need to use small batch size when buckets are very large, due to memory limitations
batch_size = 10
model.fit(x_train, y_train, batch_size = batch_size, epochs = 6, verbose = 1)#validation_data=(x_test, y_test))

#score = model.evaluate(x_test, y_test, verbose=0)
#print('Test loss:', score[0])
#print('Test accuracy:', score[1])

# Predictions on the test set
ytestout = utils.flatten(model.predict(x_test, batch_size = 10, verbose = 1))

# Compute the correlation between the test set predictions and the true values
r, p = stats.pearsonr(y_test, ytestout)
print "\nMetrics of the model for bin size %d" % bsize
print(r)
print(p)
print metrics.mean_squared_error(y_test, ytestout)
print metrics.mean_absolute_error(y_test, ytestout)

# Scatterplot of predicted vs. true values
plt.scatter(y_test, ytestout)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation rates ($r = {:.2f}'.format(r) + ', p < 10^{-10}$)')
plt.plot(y_test, y_test, color='m', linewidth=2.5)
plt.savefig('indel_rate_pred_full_chrom' + str(bsize) + '.png')
