import tensorflow as tf
import numpy as np
import random
import indel_model
#import load_dataset # See load_dataset script to observe how the training and test data is loaded
import load_full_dataset_sample_one_pos
import utils

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 50
    strlen = 2*window+1
    batch_size = 100
    test_batch_size = 500
    lr = 1e-4
    dropout_prob = 0.25
    num_epochs = 5
    print_every = 200

class SimpleFCN(indel_model.IndelModel):
#    def add_placeholders(self):
#        self.x = utils.dna_placeholder(2*self.config.window+1)
#        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        # Indices 1 onwards are hidden layer widths
        layer_widths = [config.strlen * 4, config.strlen * 16, config.strlen * 8, config.strlen]

        # Flatten input
        h_prev = tf.reshape(self.x, [-1, config.strlen * 4])
        for i in range(1, len(layer_widths)):
            W_fc = utils.weight_variable([layer_widths[i-1], layer_widths[i]])
            b_fc = utils.bias_variable([layer_widths[i]])
            h_prev = utils.lrelu(tf.matmul(h_prev, W_fc) + b_fc)

        W_last = utils.weight_variable([layer_widths[-1], 1])
        b_last = utils.bias_variable([1])
        y_out = tf.sigmoid(tf.matmul(h_prev, W_last) + b_last)

        # Dropout
        #keep_prob = tf.placeholder(tf.float32)
        #h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

        return y_out

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

config = Config()

'''loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)'''

loader = load_full_dataset_sample_one_pos.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, test_frac=0.05, load_coverage=False)

fc_net = SimpleFCN(config, loader)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
losses, val_accuracies = fc_net.fit(sess, save=True)

fc_net.print_metrics(sess, 'fcn', 'simple_fcn_results.txt')
