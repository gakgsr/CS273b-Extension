import tensorflow as tf
import numpy as np
import random
import indel_model
import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils
from sys import argv

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    def __init__(self, windowSize):
      self.window = windowSize
      self.strlen = 2*self.window+1
      self.batch_size = 100
      self.test_batch_size = 500
      self.lr = 1e-4
      self.dropout_prob = 0.5
      self.num_epochs = 1
      self.print_every = 100 # print accuracy every 100 steps

class SimpleConv(indel_model.IndelModel):
#    def add_placeholders(self):
#        self.x = utils.dna_placeholder(2*self.config.window+1)
#        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        left_half, right_half = tf.split(self.x, [self.config.window, self.config.window+1], axis=1)
        # First conv layer
        W_convleft1 = utils.weight_variable([5, 4, 40])
        b_convleft1 = utils.bias_variable([40])

        W_convright1 = utils.weight_variable([5, 4, 40])
        b_convright1 = utils.bias_variable([40])

        h_convleft1 = utils.lrelu(utils.conv1d(left_half, W_convleft1) + b_convleft1)
        h_convright1 = utils.lrelu(utils.conv1d(right_half, W_convright1) + b_convright1)

        # Second conv layer
        W_convleft2 = utils.weight_variable([5, 40, 80])
        b_convleft2 = utils.bias_variable([80])

        W_convright2 = utils.weight_variable([5, 40, 80])
        b_convright2 = utils.bias_variable([80])

        h_convleft2 = utils.lrelu(utils.conv1d(h_convleft1, W_convleft2) + b_convleft2)
        h_convright2 = utils.lrelu(utils.conv1d(h_convright1, W_convright2) + b_convright2)

        h_convout = tf.concat([h_convleft2, h_convright2], 1)

        # First fully connected layer. Reshape the convolution output to 1D vector
        fc_dim_1 = int(self.config.strlen * 80 / 7.89)
        W_fc1 = utils.weight_variable([self.config.strlen * 80, fc_dim_1])
        b_fc1 = utils.bias_variable([fc_dim_1])

        h_conv_flat = tf.reshape(h_convout, [-1, self.config.strlen*80])
        #h_conv_flat = tf.nn.dropout(h_conv_flat, self.keep_prob)
        h_fc1 = utils.lrelu(tf.matmul(h_conv_flat, W_fc1) + b_fc1)
        h_fc1 = tf.nn.dropout(h_fc1, self.keep_prob)

        # Final fully-connected layer
        W_fc2 = utils.weight_variable([fc_dim_1, 1])
        b_fc2 = utils.bias_variable([1])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        y_out = tf.sigmoid(y_conv)
        #TODO: Add separate filter with unshared weights that looks at center?

        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

if len(argv) < 2:
  window = 50
else:
  window = int(argv[1])

print("window {}".format(window))
config = Config(window)
loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)
#loader = load_full_dataset_sample_one_pos.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, test_frac=0.05, load_coverage=False)

conv_net = SimpleConv(config, loader, plotTrain=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
#losses, val_accuracies = conv_net.fit(sess, save=True)

#conv_net.predictAll(sess, save=True)

#all_results = conv_net.hard_examples(sess)
#hard_positives = [x for x in all_results if x[1]]
#print(all_results[:100])
#print(hard_positives[:100])

conv_net.print_metrics(sess, 'splitconv_window{}'.format(window), 'splitconv_window{}_results.txt'.format(window))
