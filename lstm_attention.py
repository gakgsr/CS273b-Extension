import tensorflow as tf
import numpy as np
import random
import indel_model
import load_dataset # See load_dataset script to observe how the training and test data is loaded
import utils

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 50
    strlen = 2*window+1
    batch_size = 100
    test_batch_size = 500
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 6
    num_hidden = 128
    print_every = 100 # print accuracy every 100 steps

class SimpleLSTM(indel_model.IndelModel):
#    def add_placeholders(self):
#        self.x = utils.dna_placeholder(2*self.config.window+1)
#        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    def add_prediction_op(self):
        W = tf.get_variable("W", [2*self.config.num_hidden, 1], initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable("b", [1], initializer=tf.contrib.layers.xavier_initializer())
        fw_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.config.num_hidden, forget_bias=1.0)
        bw_cell_1 = tf.contrib.rnn.BasicLSTMCell(self.config.num_hidden, forget_bias=1.0)
	fw_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.config.num_hidden, forget_bias=1.0)
        bw_cell_2 = tf.contrib.rnn.BasicLSTMCell(self.config.num_hidden, forget_bias=1.0)

        #x = tf.unstack(self.x, 2*self.config.window+1, 1)
	outputs, output_state_fw, output_state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn([fw_cell_1, fw_cell_2], [bw_cell_1, bw_cell_2], self.x, dtype=tf.float32)
	outputs = tf.unstack(outputs, axis=1)
	print(len(outputs))
	print(outputs[-1].get_shape())
	#W_a =  tf.get_variable("W_a", [2*self.config.num_hidden, 1], initializer=tf.contrib.layers.xavier_initializer())
        #b_a = tf.get_variable("b_a", [1], initializer=tf.contrib.layers.xavier_initializer())

	#e  = tf.tanh(tf.matmul(outputs, W_a) + b_a)
	#attention = tf.nn.softmax(e)
	#context = tf.matmul(outputs, attention)
	
	y_lstm = tf.matmul(outputs[-1], W) + b
	print(y_lstm.get_shape())
        return tf.sigmoid(y_lstm)

    def add_loss_op(self, pred):
        loss = utils.cross_entropy(pred, self.y_)
        return loss

    def add_training_op(self, loss):
        train_op = utils.adam_opt(loss, self.config.lr, self.loader.num_trainbatches(), 0.98)
        return train_op

config = Config()
loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
                                    batchSize=config.batch_size,
                                    testBatchSize=config.test_batch_size,
                                    seed=1, test_frac=0.025, pos_frac=0.5, load_coverage=False)

lstm_net = SimpleLSTM(config, loader)
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
losses, val_accuracies = lstm_net.fit(sess, save=True)
lstm_net.predictAll(sess, save=True)

print("test accuracy %g" % lstm_net.test(sess))
auroc = lstm_net.calc_roc(sess, 'lstm_auroc.png')
print("ROC AUC: %g" % auroc)
auprc = lstm_net.calc_auprc(sess, 'lstm_auprc.png')
print("PR AUC: %g" % auprc)
print("f1 score: %g" % lstm_net.calc_f1(sess))
lstm_net.print_confusion_matrix(sess)
lstm_net.plot_val_accuracies('lstm_val.png')
