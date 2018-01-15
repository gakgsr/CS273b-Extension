import time
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import tensorflow as tf
import utils
import entropy

class IndelModel(object):
    """Base model class for neural network indel classifiers."""

    def __init__(self, config, loader, include_coverage=False, include_entropy=False, include_recombination=False, multiclass=False, plotTrain=False):
        self.config = config # Config object contains model hyperparameters
        self.loader = loader # Dataset loader. Must satisfy basic API to load training and testing batches, etc. For instance see load_dataset.py
        self.batches_per_epoch = loader.num_trainbatches()
        self.num_test_batches = loader.num_testbatches()
        self.predictions = None # Test predictions will be stored here
        self.val_accuracies = None # Validation accuracies during training will be stored here
        self.include_coverage = include_coverage
        self.include_entropy = include_entropy
        self.include_recombination = include_recombination
        self.multiclass = multiclass # Whether it is a multiclass model
        self.plotTrain = plotTrain # Whether to plot training accuracies
        self.build()

    # Initializes the model TensorFlow placeholders. (Note: Not all necessarily need to be used)
    def add_placeholders(self):
        self.x = utils.dna_placeholder(2*self.config.window+1)
        self.c = utils.coverage_placeholder(2*self.config.window+1)
        self.e = utils.coverage_placeholder(2*self.config.window+1)
        self.r = tf.placeholder(tf.float32, shape=None)
        self.y_ = tf.placeholder(tf.float32, shape=[None, 1])

    # Helper method for feed_dict creation
    def get_feed_dict_values(self, batch):
        feedDictValues = [batch[0]] # batch[0] is the sequence data
        currIdx = 1
        featureFlags = [self.include_coverage, self.include_entropy, self.include_recombination]
        for i in range(len(featureFlags)):
            if featureFlags[i]: # Include this feature
                feedDictValues.append(batch[currIdx])
                currIdx += 1
            else: # Just use "None" for this feature, since we don't use it
                feedDictValues.append(None)
        feedDictValues.append(batch[currIdx]) # The labels
        return tuple(feedDictValues)
    
    # feed dict for the batch
    def create_feed_dict(self, batch, keep_prob):
        feedDictValues = self.get_feed_dict_values(batch)
        return {self.x:feedDictValues[0],self.c:feedDictValues[1],self.e:feedDictValues[2],self.r:feedDictValues[3],
                                                                                self.y_:feedDictValues[4],self.keep_prob:keep_prob}

    def add_prediction_op(self):
        """Adds the core transformation for this model which transforms a batch of input data into a batch of predictions."""
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_loss_op(self, pred):
        # Loss function for the model
        raise NotImplementedError("Each Model must re-implement this method.")

    def add_training_op(self, loss):
        # Training optimizer (Adam recommended)
        raise NotImplementedError("Each Model must re-implement this method.")
    
    def run_epoch(self, sess, epoch_num, validate=True):
        """Runs an epoch of training.
           Args:
                sess: tf.Session() object
                labels: np.ndarray of shape (n_samples, n_classes)
           Returns:
                average loss (scalar) and validation accuracies every 'print_every' steps of training
        """
        total_loss = 0
        accuracies = []
        trainaccs = [] # Training set accuracies
        for i in range(self.batches_per_epoch):
            batch = self.loader.get_batch() # Next training batch
            if self.config.print_every and i % self.config.print_every == 0: # Print some summaries
                if validate: # Evaluate validation set accuracy
                    val_accuracy = self.eval_validation_accuracy()
                    print("step {}, validation accuracy {:.3f}".format(i, val_accuracy))
                    accuracies.append((i + epoch_num * self.batches_per_epoch, val_accuracy))
                if self.plotTrain: # Output and store training accuracy
                    train_accuracy = self.eval_accuracy_on_batch(batch)
                    print("step {}, training accuracy {:.3f}".format(i, train_accuracy))
                    trainaccs.append((i + epoch_num * self.batches_per_epoch, train_accuracy))

            _, loss_val = sess.run([self.train_op, self.loss], feed_dict = self.create_feed_dict(batch,1-self.config.dropout_prob)) # Run training step, store loss
            total_loss += loss_val
        self.trainaccs = trainaccs
        return total_loss / self.batches_per_epoch, accuracies

    def fit(self, sess, save=True):
        """Fit model on provided data.

           Args:
                sess: tf.Session()
                save: Whether to save the validation set accuracies during training
           Returns:
                losses: list of average loss values per epoch
                val_accuracies: validation set accuracies during training
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

    # Return predictions for given input
    def predict(self, sess, X, C = None, E = None, R = None):
        return self.pred.eval(feed_dict={self.x: X, self.c: C, self.e: E, self.r: R, self.keep_prob: 1.0})

    # Get model output for all test examples
    def predictAll(self, sess, save=False):
        if self.predictions is not None: return self.predictions
        predictions = None
        for i in range(self.num_test_batches):
            testbatch = self.loader.get_testbatch()
            X, C, E, R, Y = self.get_feed_dict_values(testbatch)
            preds = self.predict(sess, X, C, E, R)
            if not self.multiclass:
                preds = utils.flatten(preds) # Flatten labels in the binary classification case, since they are just 1 or 0
            if predictions is None:
                predictions = preds # First batch
            else:
                predictions = np.concatenate((predictions, preds))
        if save: self.predictions = predictions # Save the predictions
        return predictions

    # Evaluate accuracy on all test examples
    def test(self, sess):
        if self.predictions is not None:
            # Evaluate the accuracy of the saved predictions, based on the true test labels
            if self.multiclass:
                return metrics.accuracy_score(np.argmax(self.loader.test_data[-1], axis=-1), np.argmax(self.predictions.round(), axis=-1))
            else:
                return metrics.accuracy_score(utils.flatten(self.loader.test_data[-1]), self.predictions.round())
        test_acc = 0
        num_test_ex = 0
        for i in range(self.num_test_batches):
            testbatch = self.loader.get_testbatch()
            cur_size = len(testbatch[1])
            batch_acc = curr_size * self.eval_accuracy_on_batch(testbatch) # Accuracy on the batch
            test_acc += batch_acc
            num_test_ex += cur_size # Final accuracy needs to be weighted average of batch accuracies, weighted by the size of each batch (since the last batch may be smaller)
        return test_acc / num_test_ex

    # Sort the examples by how badly the prediction was off from the true value (in decreasing order). Only works for binary task right now. (TODO: extend to multiclass case)
    def hard_examples(self, sess, make_strs=True, reverse_offset=False):
        len_testdata = self.loader.len_testdata()
        if self.predictions is not None:
            if make_strs: # Convert the onehot encoded input sequence data into DNA strings, for readability
              strs = utils.batch_to_strs(self.loader.test_data[0])
            else:
              strs = self.loader.test_data[0]
            pred_list = list(utils.flatten(self.predictions))
            label_list = list(utils.flatten(self.loader.test_data[-1])) # True test labels
            indices_list = list(utils.flatten(self.loader.indices[-len_testdata:])) # The (center) index of each test example within the chromosome
            nearby_indices_list = list(utils.flatten(self.loader.nearby_indels[-len_testdata:])) # not meaningful if 'nearby' flag is false; can be ignored in that case
            if reverse_offset: # If an offset was used when loading, need to reverse it
              indices_list = [x + self.loader.offset for x in indices_list]
              nearby_indices_list = [x + self.loader.offset for x in nearby_indices_list]
            all_results = list(zip(pred_list, label_list, strs, indices_list, nearby_indices_list))
        else: # Similar to above, but calculates predictions first, except doesn't include indices. Really, should just always make and save predictions first, so there is no need to deal with this case. (TODO)
            all_results = [None]*len_testdata
            index = 0
            for i in range(self.num_test_batches):
                testbatch = self.loader.get_testbatch()
                X, C, E, R = self.get_feed_dict_values(testbatch)
                yp = list(utils.flatten(self.predict(sess, X, C, E, R))) # Predictions
                if make_strs:
                  y_both = list(zip(yp, list(utils.flatten(testbatch[-1])), utils.batch_to_strs(testbatch[0])))
                else:
                  y_both = list(zip(yp, list(utils.flatten(testbatch[-1])), testbatch[0]))
                for j in range(index, index + len(y_both)):
                    all_results[j] = y_both[j - index]
                index += len(y_both)
        all_results.sort(key=lambda x: -abs(x[0] - x[1])) # Sort by decreasing difference of the predicted indel probability with the actual 0-1 label
        return all_results

    # Calculate and plot the ROC curve and AUROC.
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

    # Calculate and plot the precision-recall curve (PRC) and AUPRC.
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
    
    # Calculate the model's F1-score.
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

    # Plot the validation accuracies, and training accuracies as well if desired, during training. Must actually run training first.
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
        if self.plotTrain:
          plt.plot(steps, self.trainaccs, color='darkorange', lw=1, label='Training Batch Accuracy')
        plt.savefig(plotName)
        plt.clf()

    # Evaluate the model's accuracy on a batch of examples.
    def eval_accuracy_on_batch(self, batch):
        X, C, E, R, Y = self.get_feed_dict_values(batch)
        return self.accuracy.eval(feed_dict = {self.x: X, self.c: C, self.e: E, self.r: R, self.y_: Y, self.keep_prob: 1.0})

    # Evaluate the accuracy on the whole validation set.
    def eval_validation_accuracy(self):
        validation_batch = self.loader.val_set()
        x_val, c_val, e_val, r_val, y_val = self.get_feed_dict_values(validation_batch)
        return self.accuracy.eval(feed_dict = {self.x: x_val, self.c: c_val, self.e: e_val, self.r: r_val, self.y_: y_val, self.keep_prob: 1.0})

    # Setup the model.
    def build(self):
        self.add_placeholders()
        self.keep_prob = tf.placeholder(tf.float32)
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self.loss)
        self.accuracy = utils.compute_accuracy(self.pred, self.y_)

    def plot_complexity(self, sess):
        predictions = self.predictions.round()
        testLabels = self.loader.test_data[-1]
        testLabels = utils.flatten(testLabels)
        testEntropy = entropy.entropySequence(self.loader.test_data[0])
        print predictions.shape, testLabels.shape, testEntropy.shape
        # TODO: Remove hardcoded image names (e.g. "CNN...")
        # Modified TODO: Remove this function if it is no longer in use
        plt.hist(testEntropy[np.logical_and(testLabels == 1, predictions == 1)])
        plt.title('Histogram of sequence entropy for true positives')
        plt.savefig('CNNComplexityTruePositives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 0, predictions == 1)])
        plt.title('Histogram of sequence entropy for false positives')
        plt.savefig('CNNComplexityFalsePositives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 0, predictions == 0)])
        plt.title('Histogram of sequence entropy for true negatives')
        plt.savefig('CNNComplexityTrueNegatives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(testLabels == 1, predictions == 0)])
        plt.title('Histogram of sequence entropy for false negatives')
        plt.savefig('CNNComplexityFalseNegatives.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 1, predictions == 1), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for true positives')
        plt.savefig('CNNComplexityTruePositivesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 0, predictions == 1), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for false positives')
        plt.savefig('CNNComplexityFalsePositivesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 0, predictions == 0), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for true negatives')
        plt.savefig('CNNComplexityTrueNegativesGEQ1.png')
        plt.clf()
        plt.hist(testEntropy[np.logical_and(np.logical_and(testLabels == 1, predictions == 0), testEntropy >= 1)])
        plt.title('Histogram of sequence entropy for false negatives')
        plt.savefig('CNNComplexityFalseNegativesGEQ1.png')
        plt.clf()

    def print_metrics_for_binned(self, testLabels, predictions, validIndices):
        print "Accuracy Score: %f" % metrics.accuracy_score(testLabels[validIndices], predictions[validIndices])
        print "F1 Score: %f" % metrics.f1_score(testLabels[validIndices], predictions[validIndices])
        print "ROCAUC Score: %f" % metrics.roc_auc_score(testLabels[validIndices], predictions[validIndices])
        fpr, tpr, thresholds = metrics.roc_curve(testLabels[validIndices], predictions[validIndices])
        print "PRAUC Score: %f" % metrics.average_precision_score(testLabels[validIndices], predictions[validIndices])
        print "Confusion Matrix:"
        print metrics.confusion_matrix(testLabels[validIndices], predictions[validIndices])

    def print_binned_accuracy(self, sess):
        predictions = self.predictions.round()
        testLabels = self.loader.test_data[-1]
        testAC = self.loader.allele_count_test
        testLabels = utils.flatten(testLabels)
        testEntropy = entropy.entropySequence(self.loader.test_data[0])
        minEntropy = [0.9, 1.0, 1.1, 1.2, 1.3]
        maxEntropy = [1.0, 1.1, 1.2, 1.3, 1.4]
        print "Model Prediction in different entropy bins\n"
        for minval, maxval in zip(minEntropy, maxEntropy):
            print "%f <= Entropy < %f" % (minval, maxval)
            validIndices = np.logical_and(testEntropy >= minval, testEntropy < maxval)
            self.print_metrics_for_binned(testLabels, predictions, validIndices)
        print "End of Model Prediction in different entropy bins\n"
        print "Model Prediction in AC = 1 set\n"
        validIndices = np.logical_or(testAC == 1, testLabels == 0)
        self.print_metrics_for_binned(testLabels, predictions, validIndices)
        print "Model Prediction in AC > 1 set\n"
        validIndices = np.logical_or(testAC > 1, testLabels == 0)
        self.print_metrics_for_binned(testLabels, predictions, validIndices)

    # Print all of the import metrics and plots. Also write sthese results to a text file with the specified name.
    def print_metrics(self, sess, plot_prefix, output_stats_file):
        losses, val_accuracies = self.fit(sess, save=True)
        self.predictAll(sess, save=True)
        #self.plot_complexity(sess)
        #self.print_binned_accuracy(sess)
        print("Validation Chromosome: {}".format(self.loader.val_chrom))
        print("Test Chromosome: {}".format(self.loader.test_chrom))
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
