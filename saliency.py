# Generates saliency map visualizations. TODO: Change from color-based plots to the plots with varying letter sizes (using DragoNN or similar?)

import tensorflow as tf
import numpy as np
import random
from sys import argv
import indel_model
import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib import cm as CM

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 8#10
    strlen = 2*window+1
    batch_size = 1000#100
    test_batch_size = 500
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 1#5
    print_every = 100 # print accuracy every 100 steps

class SimpleConv(indel_model.IndelModel):
    def add_placeholders(self):
        self.x = utils.dna_placeholder(2*self.config.window+1)
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        fs = [5, 5] # filter sizes
        cs = [4, 40, 80] # cs[i] is output number of channels from layer i [where layer 0 is input layer]

        # First conv layer
        W_conv1 = utils.weight_variable([fs[0], cs[0], cs[1]])
        b_conv1 = utils.bias_variable([cs[1]])

        h_conv1 = tf.nn.relu(utils.conv1d(self.x, W_conv1) + b_conv1)

        # Second conv layer
        W_conv2 = utils.weight_variable([fs[1], cs[1], cs[2]])
        b_conv2 = utils.bias_variable([cs[2]])

        h_conv2 = tf.nn.relu(utils.conv1d(h_conv1, W_conv2) + b_conv2)

        # First fully connected layer. Reshape the convolution output to 1D vector
        W_fc1 = utils.weight_variable([self.config.strlen * cs[2], 1024])
        b_fc1 = utils.bias_variable([1024])

        h_conv2_flat = tf.reshape(h_conv2, [-1, self.config.strlen * cs[2]])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv2_flat, W_fc1) + b_fc1)

        # Dropout (should be added to earlier layers too...)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Final fully-connected layer
        W_fc2 = utils.weight_variable([1024, 1])
        b_fc2 = utils.bias_variable([1])

        y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        y_out = tf.sigmoid(y_conv)
        self.grads = tf.squeeze(tf.gradients(tf.squeeze(y_conv), self.x))
        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

    def get_grads(self, sess, X):
        return np.squeeze(sess.run(tf.gradients(self.pred, self.x), feed_dict={self.x: X, self.y_: np.zeros((len(X), 1)), self.keep_prob: 1.0}))

if len(argv) > 1:
  chromosome = int(argv[1])
else:
  chromosome=21
config = Config()
loader = load_dataset.DatasetLoader(chromosome=chromosome, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)

conv_net = SimpleConv(config, loader)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
losses, val_accuracies = conv_net.fit(sess, save=True)

conv_net.predictAll(sess, save=True)
print("test accuracy %g" % conv_net.test(sess))

all_results = conv_net.hard_examples(sess, make_strs=False)
hard_positives = [x for x in all_results if x[1]]
hard_negatives = [x for x in all_results if not x[1]]
#print(hard_positives[-10:]) # easiest positives

matplotlib.rcParams['figure.figsize'] = [20., 3.0]
plt.rc('xtick',labelsize=17.5)

# Heat map: More positive values are redder, more negative values are bluer.
# Positive means that that base pair contributes positively to the predicted probability of indel. Analogous for negative.
def heatmap(A, s, name):
    # Normalize A to get all values within [-1, 1] (is this appropriate?? TODO)
    A = np.expand_dims(A, axis=0) / np.amax(np.abs(A))
    fig, ax = plt.subplots()
    heatmap = ax.pcolor(A, vmin=-1, vmax=1, cmap=CM.bwr)
    plt.colorbar(heatmap)

    # put the major ticks at the middle of each cell
    ax.set_xticks(np.arange(A.shape[1]) + 0.5, minor=False)
    ax.invert_yaxis()
    ax.xaxis.tick_top()

    column_labels = list(s)
    column_labels[len(column_labels)//2] += '*' # Star the center position to make it easy to pick out
    ax.set_xticklabels(column_labels, minor=False, rotation = 'horizontal')
    ax.set_yticklabels([])
    plt.savefig(name)
    plt.clf()

def flatten(grad, inp):
    return [x[np.argmax(l)] for x, l in zip(grad, inp)] # Pick out the gradient with respect to the actual base pair at each position, ignoring the bases that don't appear

# Find the gradients for the easiest negative example and easiest positive example
easy_neg_grad = conv_net.get_grads(sess, np.array([list(x[2]) for x in hard_negatives[-1:]]))
easy_pos_grad = conv_net.get_grads(sess, np.array([list(x[2]) for x in hard_positives[-1:]]))

neg_flat = flatten(easy_neg_grad, hard_negatives[-1][2])
pos_flat = flatten(easy_pos_grad, hard_positives[-1][2])

#print(easy_pos_grad)
print(utils.onehot_to_str(hard_positives[-1][2]))
print(pos_flat)
print(utils.onehot_to_str(hard_negatives[-1][2]))
#print(flatten(easy_pos_grad, hard_positives[-1][2]))

# Plot the saliency maps
heatmap(neg_flat, utils.onehot_to_str(hard_negatives[-1][2]), 'saliency_negative.png')
heatmap(pos_flat, utils.onehot_to_str(hard_positives[-1][2]), 'saliency_positive.png')
