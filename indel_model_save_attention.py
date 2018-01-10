import time
import matplotlib
import pdb
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
import utils

class IndelModel(object):
    """Base model class for neural network indel classifiers."""

    def __init__(self, config, loader, include_coverage=False, include_entropy = False, multiclass=False):
        self.config = config
        self.loader = loader
        self.batches_per_epoch = 1 #loader.num_trainbatches()
        self.num_test_batches = loader.num_testbatches()
        self.predictions = None
        self.val_accuracies = None
        self.include_coverage = include_coverage
        self.include_entropy = include_entropy
        self.multiclass = multiclass
        self.build()

    def add_placeholders(self):
        raise NotImplementedError("Each Model must re-implement this method.")

    # This function doesn't seem to be used anywhere?
    def create_feed_dict(self, inputs_batch, labels_batch=None):
        if labels_batch is None:
            feed_dict = {self.x: inputs_batch}
        else:
            feed_dict = {self.x: inputs_batch,
                         self.y_: labels_batch}
            return feed_dict

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input data into a batch of predictions."""
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        raise NotImplementedError("Each Model must re-implement this method.")

    def run_epoch(self, sess, epoch_num, validate=True):
        """Runs an epoch of training.
           Args:
                sess: tf.Session() object
                inputs: np.ndarray of shape (n_samples, n_features)
                labels: np.ndarray of shape (n_samples, n_classes)
           Returns:
                average_loss: scalar. Average minibatch loss of model on epoch.
        """
        total_loss = 0
        accuracies = []
        for i in range(self.batches_per_epoch):
            batch = self.loader.get_batch()
            if self.config.print_every and i % self.config.print_every == 0:
                if validate:
                    val_accuracy = self.eval_validation_accuracy()
                    print("step {}, validation accuracy {:.3f}".format(i, val_accuracy))
                    accuracies.append((i + epoch_num * self.batches_per_epoch, val_accuracy))
                else:
                    if self.include_coverage and self.include_entropy:
                        train_accuracy = self.eval_accuracy_on_batch(batch[0], batch[1], batch[2], batch[3])
                    elif self.include_coverage:
                        train_accuracy = self.eval_accuracy_on_batch(batch[0], batch[1], batch[2])
                    elif self.include_entropy:
                        train_accuracy = self.eval_accuracy_on_batch(batch[0], batch[1], batch[2])
                    else:
                        train_accuracy = self.eval_accuracy_on_batch(batch[0], batch[1])
                    print("step {}, training accuracy {:.3f}".format(i, train_accuracy))
            
            if self.include_coverage and self.include_entropy:
                _, loss_val = sess.run([self.train_op, self.loss],
                                     feed_dict={self.x: batch[0], self.c: batch[1], self.e: batch[2], self.y_: batch[2], 
                                                self.keep_prob: 1-self.config.dropout_prob})
            elif self.include_coverage:
                _, loss_val = sess.run([self.train_op, self.loss],
                                     feed_dict={self.x: batch[0], self.c: batch[1], self.y_: batch[2], 
                                                self.keep_prob: 1-self.config.dropout_prob})
            elif self.include_entropy:
                _, loss_val = sess.run([self.train_op, self.loss],
                                     feed_dict={self.x: batch[0], self.e: batch[1], self.y_: batch[2], 
                                                self.keep_prob: 1-self.config.dropout_prob})
            else:
                attention, _, loss_val = sess.run([self.attention, self.train_op, self.loss],
                                         feed_dict={self.x: batch[0], self.y_: batch[1],
                                                    self.keep_prob: 1-self.config.dropout_prob})
		pdb.set_trace()
		np.savetxt("a.csv", attention[0], delimiter=",")
            total_loss += loss_val

        return total_loss / self.batches_per_epoch, accuracies

    def fit(self, sess, save=True):
        """Fit model on provided data.

           Args:
                sess: tf.Session()
                inputs: np.ndarray of shape (n_samples, n_features)
                labels: np.ndarray of shape (n_samples, n_classes)
           Returns:
                losses: list of loss per epoch
        """
        losses = []
        val_accuracies = []
        for epoch in range(self.config.num_epochs):
            start_time = time.time()
            average_loss, epoch_accuracies = self.run_epoch(sess, epoch)
            val_accuracies.extend(epoch_accuracies)
            duration = time.time() - start_time
            print('Epoch {:}: loss = {:.2f} ({:.3f} sec)'.format(epoch, average_loss, duration))
            losses.append(average_loss)
        if save: self.val_accuracies = val_accuracies
        return losses, val_accuracies

    def predict(self, sess, X, C = None, E = None):
        if self.include_coverage and self.include_entropy:
            return self.pred.eval(feed_dict={self.x: X, self.c: C, self.e: E, self.keep_prob: 1.0})
        elif self.include_coverage:
            return self.pred.eval(feed_dict={self.x: X, self.c: C, self.keep_prob: 1.0})
        elif self.include_entropy:
            return self.pred.eval(feed_dict={self.x: X, self.e: E, self.keep_prob: 1.0})
        else:
            return self.pred.eval(feed_dict={self.x: X, self.keep_prob: 1.0})

    # Get model output for all test examples
    def predictAll(self, sess, save=False):
        if self.predictions is not None: return self.predictions
        predictions = None
        for i in range(self.num_test_batches):
            testbatch = self.loader.get_testbatch()
            if self.include_coverage and self.include_entropy:
                preds = self.predict(sess, testbatch[0], testbatch[1], testbatch[2])
            elif self.include_coverage:
                preds = self.predict(sess, testbatch[0], testbatch[1])
            elif self.include_entropy:
                preds = self.predict(sess, testbatch[0], E = testbatch[1])
            else:
                preds = self.predict(sess, testbatch[0])
            if not self.multiclass:
                preds = utils.flatten(preds)
            if predictions is None:
                predictions = preds
            else:
                predictions = np.concatenate((predictions, preds))
        if save: self.predictions = predictions
        return predictions

    # Test on all test examples
    def test(self, sess):
        if self.predictions is not None:
            if self.multiclass:
                return metrics.accuracy_score(np.argmax(self.loader.test_data[-1], axis=-1), np.argmax(self.predictions.round(), axis=-1))
            else:
                return metrics.accuracy_score(utils.flatten(self.loader.test_data[-1]), self.predictions.round())
        test_acc = 0
        num_test_ex = 0
        for i in range(self.num_test_batches):
            testbatch = self.loader.get_testbatch()
            cur_size = len(testbatch[1])
            if self.include_coverage and self.include_entropy:
                batch_acc = cur_size * self.eval_accuracy_on_batch(testbatch[0], testbatch[1], testbatch[2], testbatch[3])
            elif self.include_coverage:
                batch_acc = cur_size * self.eval_accuracy_on_batch(testbatch[0], testbatch[1], testbatch[2])
            elif self.include_entropy:
                batch_acc = cur_size * self.eval_accuracy_on_batch(testbatch[0], testbatch[1], testbatch[2])
            else:
                batch_acc = cur_size * self.eval_accuracy_on_batch(testbatch[0], testbatch[1])
            test_acc += batch_acc
            num_test_ex += cur_size
        return test_acc / num_test_ex

    # Sort the examples by how badly the prediction was off from the true value (in decreasing order)
    def hard_examples(self, sess, make_strs=True, reverse_offset=False):
        len_testdata = self.loader.len_testdata()
        if self.predictions is not None:
            if make_strs:
              strs = utils.batch_to_strs(self.loader.test_data[0])
            else:
              strs = self.loader.test_data[0]
            pred_list = list(utils.flatten(self.predictions))
            label_list = list(utils.flatten(self.loader.test_data[-1]))
            indices_list = list(utils.flatten(self.loader.indices[-len_testdata:]))
            nearby_indices_list = list(utils.flatten(self.loader.nearby_indels[-len_testdata:])) # not accurate if nearby flag is false; can be ignored
            if reverse_offset:
              indices_list = [x + self.loader.offset for x in indices_list]
              nearby_indices_list = [x + self.loader.offset for x in nearby_indices_list]
            all_results = list(zip(pred_list, label_list, strs, indices_list, nearby_indices_list))
        else:
            all_results = [None]*len_testdata
            index = 0
            for i in range(self.num_test_batches):
                testbatch = self.loader.get_testbatch()
                if self.include_coverage and self.include_entropy:
                    yp = list(utils.flatten(self.predict(sess, testbatch[0], testbatch[1], testbatch[2])))
                elif self.include_coverage:
                    yp = list(utils.flatten(self.predict(sess, testbatch[0], testbatch[1])))
                elif self.include_entropy:
                    yp = list(utils.flatten(self.predict(sess, testbatch[0], testbatch[1])))
                else:
                    yp = list(utils.flatten(self.predict(sess, testbatch[0])))
                if make_strs:
                  y_both = list(zip(yp, list(utils.flatten(testbatch[-1])), utils.batch_to_strs(testbatch[0])))
                else:
                  y_both = list(zip(yp, list(utils.flatten(testbatch[-1])), testbatch[0]))
                for j in range(index, index + len(y_both)):
                    all_results[j] = y_both[j - index]
                index += len(y_both)
        all_results.sort(key=lambda x: -abs(x[0] - x[1]))
        return all_results

    def calc_roc(self, sess, plotName=None):
        predictions = self.predictAll(sess)
        print(utils.flatten(self.loader.test_data[-1]))
        print(self.predictions)
        fpr, tpr, thresholds = metrics.roc_curve(utils.flatten(self.loader.test_data[-1]), self.predictions)
        roc_auc = metrics.auc(fpr, tpr)
        if plotName:
            plt.plot(list(reversed(1-fpr)), list(reversed(tpr)), color='darkorange',
                     lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
            plt.plot([0, 1], [1, 0], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.01])
            plt.ylim([0.0, 1.01])
            plt.xlabel('Sensitivity')
            plt.ylabel('Specificity')
            plt.title('Receiver operating characteristic')
            plt.legend(loc='lower left')
            plt.savefig(plotName)
            plt.clf()
        return roc_auc

    def calc_auprc(self, sess, plotName=None):
        predictions = self.predictAll(sess)
        precision, recall, thresholds = metrics.precision_recall_curve(utils.flatten(self.loader.test_data[-1]), self.predictions)
        pr_auc = metrics.average_precision_score(utils.flatten(self.loader.test_data[-1]), self.predictions)
        if plotName:
            plt.plot(recall, precision, color='darkorange', lw=2,
                     label='PR curve (area = %0.2f)' % pr_auc)
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.ylim([0.0, 1.05])
            plt.xlim([0.0, 1.0])
            plt.title('Precision-Recall curve')
            plt.legend(loc='lower left')
            plt.savefig(plotName)
            plt.clf()
        return pr_auc
    
    def calc_f1(self, sess):
        predictions = self.predictAll(sess)
        return metrics.f1_score(utils.flatten(self.loader.test_data[-1]), self.predictions.round())

    def print_confusion_matrix(self, sess):
        predictions = self.predictAll(sess)
        tn, fp, fn, tp = metrics.confusion_matrix(utils.flatten(self.loader.test_data[-1]), self.predictions.round()).ravel()
        outstr = 'Confusion Matrix:\n'
        outstr += '\t\t\tLabeled Positive\tLabeled Negative\n'
        outstr += 'Predicted Positive\t\t{}\t\t{}\n'.format(tp, fp)
        outstr += 'Predicted Negative\t\t{}\t\t{}'.format(fn, tn)
        print(outstr)
        return outstr

    def plot_val_accuracies(self, plotName):
        if self.val_accuracies is None:
            print('Error: Must train with save=True and validate=True first.')
            return
        steps = [x[0] for x in self.val_accuracies]
        accuracies = [x[1] for x in self.val_accuracies]
        plt.plot(steps, accuracies, color='g', lw=1, label='Validation Accuracy')
        plt.title('Validation Accuracy over time')
        plt.xlabel('Batch number')
        plt.ylabel('Fraction of validation set classified correctly')
        plt.savefig(plotName)
        plt.clf()

    def eval_accuracy_on_batch(self, x_batch, y_batch):
        if self.include_coverage and self.include_entropy:
            return self.accuracy.eval(feed_dict={self.x: x_batch, self.c: c_batch, self.c: e_batch, self.y_: y_batch, self.keep_prob: 1.0})
        elif self.include_coverage:
            return self.accuracy.eval(feed_dict={self.x: x_batch, self.c: c_batch, self.y_: y_batch, self.keep_prob: 1.0})
        elif self.include_entropy:
            return self.accuracy.eval(feed_dict={self.x: x_batch, self.e: e_batch, self.y_: y_batch, self.keep_prob: 1.0})
        else:
            return self.accuracy.eval(feed_dict={self.x: x_batch, self.y_: y_batch, self.keep_prob: 1.0})

    def eval_validation_accuracy(self):
        if self.include_coverage and self.include_entropy:
            x_val, c_val, e_val, y_val = self.loader.val_set()
            return self.accuracy.eval(feed_dict={self.x: x_val, self.c: c_val, self.e: e_val, self.y_: y_val, self.keep_prob: 1.0})
        elif self.include_coverage:
            x_val, c_val, y_val = self.loader.val_set()
            return self.accuracy.eval(feed_dict={self.x: x_val, self.c: c_val, self.y_: y_val, self.keep_prob: 1.0})
        elif self.include_entropy:
            x_val, e_val, y_val = self.loader.val_set()
            return self.accuracy.eval(feed_dict={self.x: x_val, self.e: e_val, self.y_: y_val, self.keep_prob: 1.0})
        else:
            x_val, y_val = self.loader.val_set()
            return self.accuracy.eval(feed_dict={self.x: x_val, self.y_: y_val, self.keep_prob: 1.0})

    def build(self):
        self.add_placeholders()
        self.keep_prob = tf.placeholder(tf.float32)
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.attention = self.get_attention()
	self.accuracy = utils.compute_accuracy(self.pred, self.y_)

    def print_metrics(self, sess, plot_prefix, output_stats_file):
        losses, val_accuracies = self.fit(sess, save=True)
        self.predictAll(sess, save=True)
        test_acc = self.test(sess)
        print("test accuracy %g" % test_acc)
        auroc = self.calc_roc(sess, plot_prefix + '_auroc.png')
        print("ROC AUC: %g" % auroc)
        auprc = self.calc_auprc(sess, plot_prefix + '_auprc.png')
        print("PR AUC: %g" % auprc)
        f1 = self.calc_f1(sess)
        print("f1 score: %g" % f1)
        conf_str = self.print_confusion_matrix(sess)
        self.plot_val_accuracies(plot_prefix + '_val.png')
        with open(output_stats_file, 'w') as f:
            f.write("Test accuracy: %g\n" % test_acc)
            f.write("ROC AUC: %g\n" % auroc)
            f.write("PR AUC: %g\n" % auprc)
            f.write("f1 score: %g\n" % f1)
            f.write(conf_str + "\n")
