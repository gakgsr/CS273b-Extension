import numpy as np
import random
import indel_model
#import load_dataset # See load_dataset script to observe how the training and test data is loaded
import load_full_dataset_sample_per_chrom
import utils
from sklearn import linear_model
import sklearn.metrics
import entropy

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 50
    strlen = 2*window+1
    batch_size = 200
    test_batch_size = 200
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 2
    print_every = 100 # print accuracy every 100 steps



config = Config()
loader = load_full_dataset_sample_per_chrom.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, test_frac=0.025, load_coverage=False)
#loader = load_dataset.DatasetLoader(chromosome=21, windowSize=config.window,
#                                    batchSize=config.batch_size,
#                                    testBatchSize=config.test_batch_size,
#                                    seed=1, test_frac=0.05, load_coverage=False)


datset = loader.dataset
labls = utils.flatten(loader.labels)
allele_count_test = utils.flatten(loader.allele_count)
#print(labls.shape)
#print len(loader.ordering)
np.save("/datadrive/project_data/GenomePositions21.npy", loader.genome_positions)
entropyMatrix = entropy.entropyVector(datset)
np.save("/datadrive/project_data/EntropyMatrix21.npy", entropyMatrix)
#entropy.logisticRegression(entropyMatrix, labls, range(loader.num_train_examples+1), range(loader.num_train_examples+1, loader.dataset.shape[0]), plot_complexity=False, testAC = allele_count_test)
print("Validation Chromosome: {}".format(loader.val_chrom))
print("Test Chromosome: {}".format(loader.test_chrom))
entropy.logisticRegression(entropyMatrix, labls, range(loader.num_train_examples), range(loader.num_train_examples+loader.num_val_examples, loader.dataset.shape[0]), plot_complexity=False, testAC = allele_count_test)
