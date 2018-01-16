import numpy as np
#import random
#import indel_model
import load_full_dataset_sample_per_chrom
import utils
#from sklearn import linear_model
#import sklearn.metrics
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
loader = load_full_dataset_sample_per_chrom.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, load_coverage=False, complexity_threshold=1.1)

datset = loader.dataset
labls = utils.flatten(loader.labels)
entropyMatrix = entropy.entropyVector(datset)
print("Validation Chromosome: {}".format(loader.val_chrom))
print("Test Chromosome: {}".format(loader.test_chrom))
entropy.logisticRegression(entropyMatrix, labls, loader.train_indices, loader.test_indices, testAC = loader.allele_count[loader.test_indices])
