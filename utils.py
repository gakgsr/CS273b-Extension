import tensorflow as tf
import numpy as np

## DNA Processing
letters = ['A', 'C', 'G', 'T']

# Convert a length-4 list containing 0s and at most one 1 into a DNA base, e.g. [0, 1, 0, 0] becomes 'C'. '?' if all zero.
# See https://stackoverflow.com/questions/19502378/python-find-first-instance-of-non-zero-number-in-list
def to_letter(lst):
  x = next((i for i, x in enumerate(lst) if x), None)
  if x is not None: return letters[x]
  return '?'

# Convert a one-hot encoded numpy array representing a chunk of DNA sequence to a DNA string
def onehot_to_str(elem):
  elem = list(elem)
  ret = []
  for l in elem:
    ret.append(to_letter(list(l)))
  return ''.join(ret)

# Convert a batch of one-hot encoded numpy arrays representing DNA to a list of DNA strings
def batch_to_strs(batch_x):
  b = list(batch_x)
  ret = []
  for elem in b:
    ret.append(onehot_to_str(elem))
  return ret

def flatten(arr):
  return np.reshape(arr, -1)

# Convert categorical labels to one-hot labels
def to_onehot(labels, num_categories):
  rv = np.zeros((len(labels), num_categories), dtype=np.uint8)
  for i, label in enumerate(labels):
    rv[i][label] = 1
  return rv

## Tensor flow helper methods
def weight_variable(shape):
  # Xavier initialization
  initializer = tf.contrib.layers.xavier_initializer()
  return tf.Variable(initializer(shape))

# Weight variable with L2 regularization with coefficient beta
def weight_variable_reg(name, shape, beta):
  initializer = tf.contrib.layers.xavier_initializer()
  if beta:
    return tf.get_variable(name=name, shape=shape, initializer=initializer, regularizer=tf.contrib.layers.l2_regularizer(beta))
  return tf.get_variable(name=name, shape=shape, initializer=initializer)

# Bias variable of the given shape, initialized to all 0.1 [TODO: Uniform small random initialization instead?]
def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

# 1D convolution with stride 1 and zero padding
def conv1d(x, W):
  return tf.nn.conv1d(x, W, stride=1, padding='SAME')

# Cross entropy loss. Higher weight_falsepos will penalize false positives more relative to false negatives
def cross_entropy(y_pred, y_true, weight_falsepos=1):
  # Add eps to prevent errors in rare cases of 0 input to log
  eps = 1e-12
  return tf.reduce_mean(-y_true * tf.log(y_pred + eps) - weight_falsepos * (1-y_true) * tf.log(1-y_pred + eps))

# Leaky ReLU
def lrelu(x, alpha=0.01):
  return tf.maximum(x, alpha * x)

# Adam optimizer for given loss, optionally with decayed learning rate (although decay is basically irrelevant for Adam...)
def adam_opt(loss, start_lr, decay_every_num_batches=0, decay_base=0.98):
  adam = tf.train.AdamOptimizer
  if not decay_every_num_batches:
    return adam(start_lr).minimize(loss)
  global_step = tf.Variable(0, trainable=False)
  lr = tf.train.exponential_decay(start_lr, global_step, decay_every_num_batches,
                                  decay_base, staircase=True)
  return adam(lr).minimize(loss, global_step=global_step)

# Compute the percentage of predictions that are correct
def compute_accuracy(y_pred, y_true):
  if y_pred.get_shape().as_list()[-1] == 1:
    correct_prediction = tf.equal(tf.round(y_pred), y_true)
  else: # multiclass: we are dealing with logits, not probabilities
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=-1), tf.argmax(y_true, axis=-1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  return accuracy

def coverage_placeholder(length):
  return tf.placeholder(tf.float32, shape=None)
  #return tf.placeholder(tf.float32, shape=[None, length]) # Confused why this statement is not the one being used (TODO understand why it still works)

def dna_placeholder(length):
  # 4 as last dimension because the data is one-hot encoded
  return tf.placeholder(tf.float32, shape=[None, length, 4])
