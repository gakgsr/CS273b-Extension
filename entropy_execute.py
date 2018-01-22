import numpy as np
import load_full_dataset_sample_per_chrom
import utils
import entropy
import sequence_analysis

class Config(object):
    """Holds model hyperparams and data information.
       Model objects are passed a Config() object at instantiation."""
    window = 20
    strlen = 2*window+1
    batch_size = 200
    test_batch_size = 200
    lr = 1e-4
    dropout_prob = 0.5
    num_epochs = 2
    print_every = 100 # print accuracy every 100 steps



config = Config()
loader = load_full_dataset_sample_per_chrom.DatasetLoader(windowSize=config.window, batchSize=config.batch_size, testBatchSize=config.test_batch_size, seed=1, load_coverage=False, complexity_threshold=1.2, pos_frac = 0.5)

datset = loader.dataset
labls = utils.flatten(loader.labels)
entropyMatrix = entropy.entropyVector(datset)
freq_count, freq_matrix = sequence_analysis.sequence_2_mer_generate(datset)
print("Validation Chromosome: {}".format(loader.val_chrom))
print("Test Chromosome: {}".format(loader.test_chrom))
print "Entropy Model"
log_reg_model_entrpy = entropy.logisticRegression(entropyMatrix, labls, loader.train_indices, loader.test_indices, testAC = loader.allele_count[loader.test_indices])
print "Frequency Model"
log_reg_model_freq = sequence_analysis.logistic_regression_2_mer(freq_matrix, labls, loader.train_indices, loader.test_indices)

'''
loader.load_chromosome_window_data(loader.val_chrom)
tb = 1000 # Test batch size
numBatches = (len(loader.referenceChr) + tb - 1) // tb
print('{} batches'.format(numBatches))

fullPreds = None
realIndels = []
indices = []
for i in range(100000):#range(numBatches):
  X, indelList, indexList = loader.load_chromosome_window_batch_modified(window_size=config.window, batch_size=tb)
  if i % 1000 == 0:
    print('Batch {}'.format(i))
  if np.sum(X) == 0: # No indels in the entire bucket. Skip for brevity
    continue
  preds = utils.flatten(log_reg_model.predict(entropy.entropyVector(X)))
  if fullPreds is None:
    fullPreds = preds
  else:
    fullPreds = np.concatenate((fullPreds, preds))
  realIndels.extend(indelList)
  indices.extend(indexList) # The actual indices that were examined

# Save the results
arr = np.concatenate((np.expand_dims(indices, axis=1), np.expand_dims(realIndels, axis=1), np.expand_dims(fullPreds, axis=1)), axis=1)
np.save("/datadrive/project_data/genomeIndelPredictionsValChromEntrpy.npy", arr)


loader.load_chromosome_window_data(loader.test_chrom)
tb = 1000 # Test batch size
numBatches = (len(loader.referenceChr) + tb - 1) // tb
predNums = [0]*numBatches
print('{} batches'.format(numBatches))

numTest = 0
maxNumTest = 1000
fullPreds = None
realIndels = []
indices = []
for i in range(100000):#range(numBatches):
  X, indelList, indexList = loader.load_chromosome_window_batch_modified(window_size=config.window, batch_size=tb)
  if i % 1000 == 0:
    print('Batch {}'.format(i))
  if np.sum(X) == 0: # No indels in the entire bucket. Skip for brevity
    continue
  preds = utils.flatten(log_reg_model.predict(entropy.entropyVector(X)))
  if fullPreds is None:
    fullPreds = preds
  else:
    fullPreds = np.concatenate((fullPreds, preds))
  realIndels.extend(indelList)
  indices.extend(indexList) # The actual indices that were examined

# Save the results
arr = np.concatenate((np.expand_dims(indices, axis=1), np.expand_dims(realIndels, axis=1), np.expand_dims(fullPreds, axis=1)), axis=1)
np.save("/datadrive/project_data/genomeIndelPredictionsTestChromEntrpy.npy", arr)
'''
