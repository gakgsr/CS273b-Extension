# Trains a simple CNN and then uses it to make predictions on every location in a chromosome.
# These predictions are later used to estimate the total number of indel mutations within specified regions of the genome (see process_genome_preds.py)

import tensorflow as tf
import numpy as np
import random
from sys import argv
import indel_model
#import custom_load_dataset
import load_full_dataset_sample_per_chrom
import utils

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 20#50
    strlen = 2*window+1
    batch_size = 100
    test_batch_size = 500
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 5
    print_every = 100 # print accuracy every 100 steps

class SimpleConv(indel_model.IndelModel):
    def add_prediction_op(self):
        fs = [5, 5] # filter sizes
        cs = [4, 40, 80] # cs[i] is output number of channels from layer i [where layer 0 is input layer]

        # First conv layer
        W_conv1 = utils.weight_variable([fs[0], cs[0], cs[1]])
        b_conv1 = utils.bias_variable([cs[1]])

        h_conv1 = utils.lrelu(utils.conv1d(self.x, W_conv1) + b_conv1)

        # Second conv layer
        W_conv2 = utils.weight_variable([fs[1], cs[1], cs[2]])
        b_conv2 = utils.bias_variable([cs[2]])

        h_conv2 = utils.lrelu(utils.conv1d(h_conv1, W_conv2) + b_conv2)

        # First fully connected layer. Reshape the convolution output to 1D vector
        W_fc1 = utils.weight_variable([self.config.strlen * cs[2], 1024])
        b_fc1 = utils.bias_variable([1024])

        h_conv2_flat = tf.reshape(h_conv2, [-1, self.config.strlen * cs[2]])
        h_fc1 = utils.lrelu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # Dropout (should be added to earlier layers too...)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Final fully-connected layer
        W_fc2 = utils.weight_variable([1024, 1])
        b_fc2 = utils.bias_variable([1])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_out = tf.sigmoid(y_conv)
        is_zero = tf.clip_by_value(tf.reduce_sum(self.x), 0, 1) # basically will be 1 iff at least one entry of x is nonzero
        y_out = tf.multiply(y_out, is_zero)
        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

config = Config()
#loader = load_dataset.DatasetLoader(chromosome=chromosome, windowSize=config.window, #custom_load_dataset.DatasetLoader(chromosome=chromosome, windowSize=config.window,
#                                    testBatchSize=config.test_batch_size,
#                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False, load_entire=True)
loader = load_full_dataset_sample_per_chrom.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)

chromosome = loader.test_chrom
conv_net = SimpleConv(config, loader)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
losses, val_accuracies = conv_net.fit(sess, save=True)

conv_net.predictAll(sess, save=True)
print("test accuracy %g" % conv_net.test(sess))

auroc = conv_net.calc_roc(sess, 'conv_auroc.png')
print("ROC AUC: %g" % auroc)
auprc = conv_net.calc_auprc(sess, 'conv_auprc.png')
print("PR AUC: %g" % auprc)
print("f1 score: %g" % conv_net.calc_f1(sess))
conv_net.print_confusion_matrix(sess)

loader.load_chromosome_window_data()
tb = 1000 # Test batch size
numBatches = (len(loader.referenceChr) + tb - 1) // tb
predNums = [0]*numBatches
print('{} batches'.format(numBatches))

numTest = 0
maxNumTest = 1000
fullPreds = None
realIndels = []
indices = []
for i in range(numBatches):
  X, indelList, indexList = loader.load_chromosome_window_batch_modified(window_size=config.window, batch_size=tb)
  if i % 1000 == 0:
    print('Batch {}'.format(i))
  if np.sum(X) == 0: # No indels in the entire bucket. Skip for brevity
    continue
  preds = utils.flatten(conv_net.predict(sess, X))
  sumpreds = sum(preds.round()) # Sum of predicted probabilities for the current window
  predNums[i] = sumpreds
  numTest += 1
  #if numTest > maxNumTest: break # To truncate the program and predict on just a subset of the chromosome.
  #indelList = [x in loader.setOfIndelLocations for x in range(lb, ub)] # Whether each x in [lb, ub) is an indel or not.
  if fullPreds is None:
    fullPreds = utils.flatten(preds)
  else:
    fullPreds = np.concatenate((fullPreds, utils.flatten(preds)))
  realIndels.extend(indelList)
  indices.extend(indexList) # The actual indices that were examined

from sklearn import metrics
print(metrics.average_precision_score(realIndels, fullPreds)) # Most positions are not indels, so we report the average precision score (similar to AUPRC)

# Save the results
arr = np.concatenate((np.expand_dims(indices, axis=1), np.expand_dims(realIndels, axis=1), np.expand_dims(fullPreds, axis=1)), axis=1)
np.save("/datadrive/project_data/genomeIndelPredictionsUnSpaced.npy", arr)
