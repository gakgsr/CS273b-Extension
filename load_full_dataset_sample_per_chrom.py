# TODO notes for Ananth: Add the recombination data
# Also work on the nearby_indels feature
import cs273b
import load_coverage as lc
import numpy as np
from math import ceil
import utils
import entropy
import pandas as pd
import time

# Base location of all input data files
data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
  def __init__(self, _kw=0, windowSize=100, batchSize=100, testBatchSize=500, seed=1, pos_frac=0.5, load_coverage=True, load_entropy=False, triclass=False, nearby=0, offset=0, complexity_threshold=0):
    ##
    # If window size k, means we read k base pairs before the center and k after, for a total of 2k+1 base pairs in the input
    self.window = windowSize
    # Number of training examples per batch
    self.batchSize = batchSize
    # Number of testing examples per test batch (we can't test everything at once due to memory)
    self.testBatchSize = testBatchSize
    ##
    # Whether to do tri-class classification (Insertion, Deletion, or neither) as opposed to binary (Indel or non-indel)
    self.triclass = triclass
    # If nearby is nonzero, negative examples are only sampled from within 'nearby' of some positive example. Otherwise, they are sampled at random from the genome.
    self.nearby = nearby
    # Either 0 or 1, to handle 1-indexing of the gnomad_indels.tsv file. Technically should be 1, but in practice 0 seems to work just as well??
    self.offset = offset
    ##
    # Whether to use calculated sequence entropy as input to the model
    self.load_entropy = load_entropy
    # Whether to use coverage data as input to the model
    self.load_coverage = load_coverage
    # Whether to use recombination data as input to the model
    #self.load_recombination = load_recombination
    ##
    # The minimum complexity of the sequence needed to be a part of our train/test/val sets
    self.complexity_threshold = complexity_threshold
    ##
    # Load the reference genome
    self.referenceChrFull, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
    # Preserve memory
    del ambiguous_bases
    ##
    # Index of next training example (for batching purposes)
    self.cur_index = 0
    # Index of next testing example (for batching purposes)
    self.test_index = 0
    # Index of next location in the chromosome (for load_chromosome_window_batch function)
    self.chrom_index = 0
    ##
    if seed is not None:
      np.random.seed(seed)
    self.__initializeTrainData(pos_frac)
    self.__initializeTestData()


  # Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
  # Also includes desired number of negative training examples (positions not listed as indels)
  # In this module, we only look at high complexity sequences, ie. locations which have complexity (window size 20 to measure it) >= 1.1
  def __initializeTrainData(self, frac_positives):
    ##
    # for brevity
    k = self.window
    # The window size used to compute sequence complexity
    k_seq_complexity = 20
    # We use chromosomes 2-22, we won't use chromosome 1 until the very end
    num_chrom_used = 21
    ##
    # Number of indels in the entire dataset used to train/test/val
    lengthIndels = 20000*num_chrom_used
    # Number of non-indels in the entire dataset
    num_negatives = int(int((1./frac_positives-1) * lengthIndels)/num_chrom_used)*num_chrom_used
    # Number of locations in the entire dataset
    total_length = lengthIndels + num_negatives
    ##
    # Number of indels in the entire dataset per chromosome
    num_negatives_per_chrom = int(num_negatives/num_chrom_used)
    # Number of non-indels in the entire dataset per chromosome
    lengthIndels_per_chrom = int(lengthIndels/num_chrom_used)
    # Number of locations in the entire dataset per chromosome
    total_length_per_chrom = lengthIndels_per_chrom + num_negatives_per_chrom
    ##
    # one-hot encoded sequences of size 2*k + 1 around each location
    dataset = np.zeros((total_length, 2*k + 1, 4))
    # coverage corresponding to each location in the dataset
    coverageDataset = np.zeros((total_length, 2*k + 1))
    # entropy of expanding windows in the dataset
    entropyDataset = np.zeros((total_length, 2*k + 1))
    # indices on the genome of the locations in the dataset
    indices = np.zeros(total_length, dtype=np.uint32)
    # allele count values for indels, 0 for non-indels
    allele_count = np.zeros(total_length, dtype=np.uint32)
    nearby_indels = np.zeros(total_length, dtype=np.uint32)
    # label is either a bool or an int depending on the number of classes
    if self.triclass:
      labeltype = np.uint8
    else:
      labeltype = np.bool
    # 0 for non-indels 1 (and 2) in case of indels
    labels = np.zeros(total_length, dtype=labeltype)
    # seems to be the same as indices, ToDo does it neet to be there???
    genome_positions = np.zeros(total_length, dtype=np.uint32)
    # the chromosome number corresponding to each location
    chrom_num = np.zeros(total_length, dtype=np.uint32)

    # Load data from chromosomes 2-22
    # populate dataset and related variables per chromosome
    for chromosome in range(2, 23):
      ##
      # Load the chromosome from the full genome
      referenceChr = self.referenceChrFull[str(chromosome)]
      ## Load and process the positive (indels) dataset
      # This is a 6 column data: indel locations, allele count, filter value, 50 window, and 20 window sequence complexity, insertion (1) or deletion (0)
      indel_data_load = pd.read_csv(data_dir + "indelLocationsFiltered" + str(chromosome) + ".txt", delimiter = ' ', header = None)
      # Even non-filtered ones are a part of indelLocationsFull, so that we don't put these in the negative examples even by chance
      indelLocationsFull = np.array(indel_data_load.iloc[:, 0])
      # Filter by sequence complexity and filter value around 20 sized window and complexity threshold
      total_indices = np.arange(len(indelLocationsFull))
      filtered_indices = np.logical_and(np.array(indel_data_load.iloc[:, 2] == 1), np.array(indel_data_load.iloc[:, 4] >= self.complexity_threshold))
      # Add an additional filter for allele count = 1
      filtered_indices = np.logical_and(np.array(indel_data_load.iloc[:, 1] == 1), filtered_indices)

      # Sample the indels, taking into consideration the classification problem in hand
      if self.triclass:
        filtered_indices_insert = np.logical_and(np.array(indel_data_load.iloc[:, 5] == 1), filtered_indices)
        filtered_indices_insert = total_indices[filtered_indices_insert]
        filtered_indices_delete = np.logical_and(np.array(indel_data_load.iloc[:, 5] == 0), filtered_indices)
        filtered_indices_delete = total_indices[filtered_indices_delete]
        insertionLocations = np.random.choice(filtered_indices_insert, size = int(lengthIndels_per_chrom/2), replace = False)
        deletionLocations = np.random.choice(filtered_indices_delete, size = lengthIndels_per_chrom - int(lengthIndels_per_chrom/2), replace = False)
        indel_indices = np.concatenate((insertionLocations, deletionLocations))
        del filtered_indices_insert, filtered_indices_delete, insertionLocations, deletionLocations
      else:
        filtered_indices = total_indices[filtered_indices]
        indel_indices = np.random.choice(filtered_indices, size = lengthIndels_per_chrom, replace = False)
      ##
      indelLocations = np.array(indel_data_load.iloc[indel_indices, 0])
      allele_count_val = np.array(indel_data_load.iloc[indel_indices, 1])
      del indel_data_load, indel_indices, filtered_indices, total_indices
      indelLocations = indelLocations - self.offset

      ## Load the coverage data if needed
      coverage = None
      if self.load_coverage:
        coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(chromosome))

      ## Create the negative dataset
      rel_size_neg_large = 2
      neg_positions_large = np.loadtxt(data_dir + "nonindelLocationsSampled" + str(chromosome) + '.txt').astype(int)
      neg_positions_large = np.random.choice(neg_positions_large, size = rel_size_neg_large*num_negatives_per_chrom, replace = False)
      # Remove those that have complexity below the threshold
      neg_sequence_indices = np.arange(2*k_seq_complexity + 1) - k_seq_complexity
      neg_sequence_indices = np.repeat(neg_sequence_indices, len(neg_positions_large), axis = 0)
      neg_sequence_indices = np.reshape(neg_sequence_indices, [-1, len(neg_positions_large)])
      neg_sequence_indices += np.transpose(neg_positions_large)
      neg_sequence_complexity = entropy.entropySequence(referenceChr[neg_sequence_indices.transpose(), :])
      neg_positions_large = neg_positions_large[neg_sequence_complexity >= self.complexity_threshold]
      del neg_sequence_indices, neg_sequence_complexity
      ##
      if self.nearby:
        # Create a list of all permissible nearby locations
        nearby_locations = np.arange(-self.nearby, self.nearby + 1)
        nearby_locations = np.repeat(nearby_locations, len(indelLocations), axis = 0)
        nearby_locations = np.reshape(nearby_locations, [-1, len(indelLocations)])
        nearby_locations += np.transpose(indelLocations)
        nearby_locations = np.reshape(nearby_locations, -1)
        # Remove all indel locations and low-complexity non-indel locations from nearby locations
        nearby_locations = np.array((set(nearby_locations) - set(indelLocationsFull)) & set(neg_positions_large))
        if len(nearby_locations) >= num_negatives_per_chrom:
          neg_positions = np.random.choice(nearby_locations, size = num_negatives_per_chrom, replace = False)
        else:
          # Else sample the remaining from the negative positions- this is the best that can be done, try increasing the nearby size
          print "Try increasing nearby or rel_size_neg_large. Not enough nearby-non-indels could be sampled in chromosome {}".format(chromosome)
          num_neg_needed = num_negatives_per_chrom - len(nearby_locations)
          not_nearby = np.random.choice(list((set(neg_positions_large) - set(indelLocationsFull)) - set(nearby_locations)), size = num_neg_needed, replace = False)
          neg_positions = np.concatenate((nearby_locations, not_nearby))
      else:
        neg_positions = np.random.choice(neg_positions_large, size = num_negatives_per_chrom, replace = False)

      for i in range(lengthIndels_per_chrom + num_negatives_per_chrom):
        if i < lengthIndels_per_chrom:
          if not self.triclass:
            label = 1 # standard binary classification labels
          elif i < int(lengthIndels_per_chrom/2):
            label = 1 # insertions will be labeled as 1
          else:
            label = 2 # deletions will be labeled as 2
          pos = indelLocations[i]
          allele_count[total_length_per_chrom*(chromosome - 2) + i] = allele_count_val[i]
        else:
          label = 0
          pos = neg_positions[i - lengthIndels_per_chrom]
          # Compute the true value of nearby_indels TODO
          #if self.nearby:
        indices[total_length_per_chrom*(chromosome - 2) + i] = pos
        coverageWindow = np.zeros(2*k + 1)
        # get k base pairs before and after the position
        window = referenceChr[pos - k : pos + k + 1]
        if coverage is not None:
          coverageWindow = utils.flatten(coverage[pos - k : pos + k + 1])
        dataset[total_length_per_chrom*(chromosome - 2) + i] = window
        coverageDataset[total_length_per_chrom*(chromosome - 2) + i] = coverageWindow
        labels[total_length_per_chrom*(chromosome - 2) + i] = label
        genome_positions[total_length_per_chrom*(chromosome - 2) + i] = pos
        chrom_num[total_length_per_chrom*(chromosome - 2) + i] = chromosome
    if self.load_entropy:
      entropyDataset[:, k+1:2*k+1] = entropy.entropyVector(dataset)
    ##
    # Randomly choose the validation and test chromosome
    self.val_chrom, self.test_chrom = np.random.choice(range(2, 23), 2, replace=False)
    # Set the number of training examples, and the respective set indices
    self.num_train_examples = total_length_per_chrom*(num_chrom_used - 2)
    self.train_indices = np.logical_and(chrom_num != self.val_chrom, chrom_num != self.test_chrom)
    self.test_indices = np.array(chrom_num == self.test_chrom)
    self.val_indices = np.array(chrom_num == self.val_chrom)
    ##
    # Set the respective variables
    self.dataset = dataset
    self.coverageDataset = coverageDataset
    self.entropyDataset = entropyDataset
    self.indices = indices
    self.allele_count = allele_count
    self.nearby_indels = nearby_indels
    self.genome_positions = genome_positions
    self.labels = labels
    del dataset, coverageDataset, entropyDataset, indices, allele_count, nearby_indels, genome_positions, labels
  def get_batch(self):
    return self.get_randbatch() # default: random batch

  # Get the next batch of training examples. The training data is shuffled after every epoch.
  def get_randbatch(self, batchSize=0):
    if batchSize == 0: batchSize = self.batchSize
    # Randomize the order of examples, if we are at the beginning of the next epoch
    if self.cur_index == 0:
      np.random.shuffle(self.train_indices)
    start, end = self.cur_index, self.cur_index + batchSize
    batch_indices = self.train_indices[start : end] # Indices of the training examples that will make up the batch
    retval = [self.dataset[batch_indices]]
    if self.load_coverage: # Load all additional data sources that are requested
      retval.append(self.coverageDataset[batch_indices])
    if self.load_entropy:
      retval.append(self.entropyDataset[batch_indices])
    #if self.load_recombination:
    #  retval.append(self.recombinationDataset[batch_indices])
    retval.append(self.labels[batch_indices])
    self.cur_index = end # Start of next batch
    if end >= self.num_train_examples:
      self.cur_index = 0 # Epoch just ended
    return tuple(retval)

  def __initializeTestData(self):
    self.test_data = [self.dataset[self.test_indices]]
    if self.load_coverage:
      self.test_data.append(self.coverageDataset[self.test_indices])
    if self.load_entropy:
      self.test_data.append(self.entropyDataset[self.test_indices])
    self.test_data.append(self.labels[self.test_indices])
    self.test_data = tuple(self.test_data)
    print("Number of test examples: {}".format(len(self.test_indices)))

  # Total number of training batches based on given batch size.
  def num_trainbatches(self):
    return int(ceil(float(self.num_train_examples) / self.batchSize))

  # Total number of test examples
  def len_testdata(self):
    return len(self.test_data[1])

  # Total number of test batches based on given test batch size.
  def num_testbatches(self):
    return int(ceil(float(len(self.test_data[1])) / self.testBatchSize))

  # Needed if manually testing batch by batch, and doing so multiple times
  def reset_test_index(self):
    self.test_index = 0

  # Get the next batch of testing data
  def get_testbatch(self):
    if self.test_index < len(self.test_data[1]):
      rv = [self.test_data[0][self.test_index:self.test_index+self.testBatchSize]]
      if self.load_coverage:
        rv.append(self.test_data[1][self.test_index:self.test_index+self.testBatchSize])
      if self.load_entropy:
        rv.append(self.test_data[-2][self.test_index:self.test_index+self.testBatchSize])
      rv.append(self.test_data[-1][self.test_index:self.test_index+self.testBatchSize])
    else:
      raise RuntimeError("test index is {}, only {} examples".format(self.test_index, len(self.test_data[1])))
    self.test_index += self.testBatchSize
    return tuple(rv)

  # Get the validation set
  def val_set(self, length=1000):
    val_indices = self.val_indices[:length]
    retval = [self.dataset[val_indices, :, :]]
    if self.load_coverage:
      retval.append(self.coverageDataset[val_indices, :])
    if self.load_entropy:
      retval.append(self.entropyDataset[val_indices, :])
    retval.append(self.labels[val_indices])
    return tuple(retval)

  #TODO Review the file from here
  def load_chromosome_window_data(self):
    chromosome = self.test_chrom
    referenceChr = self.referenceChrFull[str(chromosome)]
    if self.triclass:
      self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
      self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
      self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations)) - self.offset
    else:
      indel_data_load = np.loadtxt(data_dir + "indelLocationsFiltered{}".format(chromosome) + ext).astype(int) # This is a 5 column data: indel locations, allele count, filter value, 50 window, and 20 window sequence complexity
      indelLocationsFull = indel_data_load[:, 0] # Even non-filtered ones are a part of indelLocationsFull, so that we don't put these in the negative examples even by chance
      total_indices = np.arange(indel_data_load.shape[0])
      # Filter by sequence complexity around 20 sized window and complexity threshold
      filtered_indices = np.logical_and(np.array(indel_load_data[:, 2] == 1) and np.array(indel_load_data[:, 4] >= self.complexity_threshold))
      filtered_indices = total_indices[filtered_indices]
      indel_indices = np.random.choice(filtered_indices, size=lengthIndels_per_chrom, replace=False)
      self.indelLocations = indel_data_load[indel_indices, 0]
      del indel_data_load, indel_indices, filtered_indices, total_indices
      self.setOfIndelLocations = set(self.indelLocations)

  def load_chromosome_window_batch(self, window_size, batch_size):
    lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
    ub = min(len(referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
    num_ex = ub - lb
    X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
    self.chrom_index = ub
    labels = [ex in self.setOfIndelLocations for ex in range(lb, ub)]
    return referenceChr[X+Y, :], labels, lb, ub

  def load_chromosome_window_batch_modified(self, window_size, batch_size):
    lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
    ub = min(len(referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
    num_ex = ub - lb
    X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
    self.chrom_index = ub
    k = window_size
    k_seq_complexity = 20
    indelList = [x in self.setOfIndelLocations for x in range(lb, ub)]
    seqDataset = referenceChr[X+Y, :]
    complexity = entropy.entropySequence(seqDataset[:, k - k_seq_complexity : k + k_seq_complexity + 1, :])
    indexList = [complexity[x - lb] >= self.complexity_threshold for x in range(lb, ub)]
    seqDataset = seqDataset[indexList]
    return seqDataset, indelList, indexList
