import tensorflow as tf
import numpy as np
import random
import indel_model
#import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils
import load_full_dataset_sample_per_chrom

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 20
    strlen = 2*window+1
    batch_size = 100
    test_batch_size = 500
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 5
    print_every = 100 # print accuracy every 100 steps

class SimpleConvCoverage(indel_model.IndelModel):
#    def add_placeholders(self):
#        self.x = utils.dna_placeholder(2*self.config.window+1)
#        self.c = utils.coverage_placeholder(2*self.config.window+1)
#        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        fs = [5, 5] # filter sizes
        cs = [4, 80, 160] # cs[i] is output number of channels from layer i [where layer 0 is input layer]

        # First conv layer
        W_conv1 = utils.weight_variable([fs[0], cs[0], cs[1]])
        b_conv1 = utils.bias_variable([cs[1]])

        h_conv1 = utils.lrelu(utils.conv1d(self.x, W_conv1) + b_conv1)

	# Middle conv layer
        W_conv_mid = utils.weight_variable([fs[1], cs[1], cs[1]])
        b_conv_mid = utils.bias_variable([cs[1]])

        h_conv_mid = utils.lrelu(utils.conv1d(h_conv1, W_conv_mid) + b_conv_mid)

        # Second conv layer
        W_conv2 = utils.weight_variable([fs[1], cs[1], cs[2]])
        b_conv2 = utils.bias_variable([cs[2]])

        h_conv2 = utils.lrelu(utils.conv1d(h_conv_mid, W_conv2) + b_conv2)
	
        # Conv layer on top of the coverage
        W_conv_coverage = utils.weight_variable([fs[0], 1, cs[2]])
        b_conv_coverage = utils.bias_variable([cs[2]])

        conv_c = tf.expand_dims(self.c, -1)
        #print(conv_c.shape, W_conv_coverage.shape, b_conv_coverage.shape)
        h_conv_coverage = utils.lrelu(utils.conv1d(conv_c, W_conv_coverage) + b_conv_coverage)

        h_concatenated = tf.concat([h_conv2, h_conv_coverage], axis = -1)
        # First fully connected layer. Reshape the convolution output to 1D vector

        orig_shape = h_concatenated.get_shape().as_list()
        flat_shape = np.prod(orig_shape[1:])
        new_shape = [-1,] + [flat_shape]
        h_concatenated_flat = tf.reshape(h_concatenated, new_shape)
        h_concat_drop = tf.nn.dropout(h_concatenated_flat, self.keep_prob)
        fc1_in = h_concatenated_flat.get_shape().as_list()[-1]
        W_fc1 = utils.weight_variable([fc1_in, 1024])
        b_fc1 = utils.bias_variable([1024])
        h_fc1 = utils.lrelu(tf.matmul(h_concat_drop, W_fc1) + b_fc1)

        # Fully-connected layer on top of the coverage
        #W_fc_coverage = utils.weight_variable([self.config.strlen, cs[2]])
        #b_fc_coverage = utils.bias_variable([cs[2]])

        #h_fc_coverage = tf.nn.relu(tf.matmul(self.c, W_fc_coverage) + b_fc_coverage)
        #h_concatenated = tf.concat([h_fc1, h_fc_coverage], axis = -1)

        # Dropout (should be added to earlier layers too...)
        #h_concatenated_drop = tf.nn.dropout(h_concatenated, self.keep_prob)
        h_fc1_drop = tf.nn.dropout(h_fc1, self.keep_prob)

        # Final fully-connected layer
        W_fc2 = utils.weight_variable([1024, 1])
        b_fc2 = utils.bias_variable([1])

        y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
        y_out = tf.sigmoid(y_conv)

        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

config = Config()
#loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
#                                    batchSize=config.batch_size,
#                                    testBatchSize=config.test_batch_size,
#                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=True)
loader = load_full_dataset_sample_per_chrom.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, pos_frac=0.5, load_coverage=True, complexity_threshold=1.1)

conv_net = SimpleConvCoverage(config, loader, include_coverage = True, plotTrain=True)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
'''losses, val_accuracies = conv_net.fit(sess, save=True)

conv_net.predictAll(sess, save=True)
print("test accuracy %g" % conv_net.test(sess))

all_results = conv_net.hard_examples(sess)
hard_positives = [x for x in all_results if x[1]]
#print(all_results[:100])
#print(hard_positives[:100])

auroc = conv_net.calc_roc(sess, 'conv_auroc.png')
print("ROC AUC: %g" % auroc)
auprc = conv_net.calc_auprc(sess, 'conv_auprc.png')
print("PR AUC: %g" % auprc)
print("f1 score: %g" % conv_net.calc_f1(sess))
conv_net.print_confusion_matrix(sess)

conv_net.plot_val_accuracies('conv_val.png')'''
conv_net.print_metrics(sess, 'conv', 'simple_conv_results.txt')
conv_net.plot_freq_by_confusion(sess, 'CNNCoverage')
