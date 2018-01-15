# ToDo notes for Ananth: Add the recombination data
# Also work on the self.triclass is true cases
# Also work on the self.nearby_indel is true cases
import cs273b
import load_coverage as lc
import numpy as np
from math import ceil
import utils
import entropy
import pandas as pd

# Base location of all input data files
data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
  def __init__(self, _kw=0, windowSize=100, batchSize=100, testBatchSize=500, seed=1, pos_frac=0.5, load_coverage=True, load_entropy=False, triclass=False, nearby=0, offset=0, complexity_threshold=0):
    self.window = windowSize # If window size k, means we read k base pairs before the center and k after, for a total of 2k+1 base pairs in the input
    self.batchSize = batchSize # Number of training examples per batch
    self.testBatchSize = testBatchSize # Number of testing examples per test batch (we can't test everything at once due to memory)
    self.triclass = triclass # Whether to do tri-class classification (Insertion, Deletion, or neither) as opposed to binary (Indel or non-indel)
    self.nearby = nearby # If nearby is nonzero, negative examples are only sampled from within 'nearby' of some positive example. Otherwise, they are sampled at random from the genome.
    self.offset = offset # Either 0 or 1, to handle 1-indexing of the gnomad_indels.tsv file. Technically should be 1, but in practice 0 seems to work just as well??
    self.load_entropy = load_entropy # Whether to use calculated sequence entropy as input to the model
    self.load_coverage = load_coverage # Whether to use coverage data as input to the model
    self.complexity_threshold = complexity_threshold # The minimum complexity of the sequence needed to be a part of our train/test/val sets
    self.referenceChrFull, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp") # Load the reference genome
    del ambiguous_bases # Preserve memory
    self.cur_index = 0 # Index of next training example (for batching purposes)
    self.test_index = 0 # Index of next testing example (for batching purposes)
    self.chrom_index = 0 # Index of next location in the chromosome (for load_chromosome_window_batch function)
    if seed is not None:
      np.random.seed(seed)
    self.__initializeTrainData(pos_frac)
    self.__initializeTestData()
    #self.load_recombination = load_recombination # Whether to use recombination data as input to the model

  # Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
  # Also includes desired number of negative training examples (positions not listed as indels)
  # In this module, we only look at high complexity sequences, ie. locations which have complexity (window size 20 to measure it) >= 1.1
  def __initializeTrainData(self, frac_positives):
    k = self.window # for brevity
    k_seq_complexity = 20 # The window size used to compute sequence complexity
    self.indelLocations = np.loadtxt(data_dir + "indelLocations21.txt").astype(int) # This has been loaded just to set the number of examples per chromosome
    num_chrom_used = 21 # We use chromosomes 2-22, we won't use chromosome 1 until the very end
    lengthIndels = int(len(self.indelLocations)/16)*num_chrom_used # Number of indels in the entire dataset
    num_negatives = int(int((1./frac_positives-1) * lengthIndels)/16)*num_chrom_used # Number of non-indels in the entire dataset
    total_length = lengthIndels + num_negatives # Number of locations in the entire dataset
    num_negatives_per_chrom = int(num_negatives/num_chrom_used) # Number of indels in the entire dataset per chromosome
    lengthIndels_per_chrom = int(lengthIndels/num_chrom_used) # Number of non-indels in the entire dataset per chromosome
    total_length_per_chrom = lengthIndels_per_chrom + num_negatives_per_chrom # Number of locations in the entire dataset per chromosome
    dataset = np.zeros((total_length, 2*k + 1, 4)) # one-hot encoded sequences of size 2*k + 1 around each location
    coverageDataset = np.zeros((total_length, 2*k + 1)) # coverage corresponding to each location in the dataset
    entropyDataset = np.zeros((total_length, 2*k + 1)) # entropy of expanding windows in the dataset
    indices = np.zeros(total_length, dtype=np.uint32) # indices on the genome of the locations in the dataset
    allele_count = np.zeros(total_length, dtype=np.uint32) # allele count values for indels, 0 for non-indels
    nearby_indels = np.zeros(total_length, dtype=np.uint32)
    if self.triclass:
      labeltype = np.uint8
    else:
      labeltype = np.bool
    labels = np.zeros(total_length, dtype=labeltype) # 0 for non-indels 1 (and 2) in case of indels
    genome_positions = np.zeros(total_length, dtype=np.uint32) # seems to be the same as indices, ToDo does it neet to be there???
    chrom_num = np.zeros(total_length, dtype=np.uint32) # the chromosome number corresponding to each location

    # Load data from chromosomes 2-22
    # populate dataset and related variables per chromosome
    for chromosome in range(2, 23):
      self.referenceChr = self.referenceChrFull[str(chromosome)]
      self.refChrLen = len(self.referenceChr)
      ext = ".txt"
      if self.triclass:
        self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
        self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
        self.indelLocationsFull = np.concatenate((self.insertionLocations, self.deletionLocations))
        self.insertLocations = np.random.choice(self.insertLocations, size=int(lengthIndels_per_chrom/2), replace=False)
        self.deletionLocations = np.random.choice(self.deletionLocations, size=lengthIndels_per_chrom - int(lengthIndels_per_chrom/2), replace=False)
        self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations))
        self.indelLocations = self.indelLocations - self.offset
      else:
        #self.indelLocationsFull = np.loadtxt(data_dir + "indelLocations{}".format(chromosome) + ext).astype(int)
        indel_data_load = pd.read_csv(data_dir + "indelLocationsFiltered" + str(chromosome) + ".txt", delimiter = ' ', header = None) # This is a 5 column data: indel locations, allele count, filter value, 50 window, and 20 window sequence complexity
        self.indelLocationsFull = indel_data_load.iloc[:, 0].tolist() # Even non-filtered ones are a part of indelLocationsFull, so that we don't put these in the negative examples even by chance
        total_indices = np.arange(len(self.indelLocationsFull))
        # Filter by sequence complexity around 20 sized window and complexity threshold
        filtered_indices = np.logical_and(np.array(indel_load_data.iloc[:, 2] == 1) and np.array(indel_load_data.iloc[:, 4] >= self.complexity_threshold))
        filtered_indices = total_indices[filtered_indices]
        indel_indices = np.random.choice(filtered_indices, size=lengthIndels_per_chrom, replace=False)
        self.indelLocations = indel_data_load.iloc[indel_indices, 0]
        allele_count_val = indel_data_load.iloc[indel_indices, 1]
        del indel_data_load, indel_indices, filtered_indices, total_indices
        self.indelLocations = self.indelLocations - self.offset
      self.nonzeroLocationsRef = np.where(np.any(self.referenceChr != 0, axis = 1))[0]
      if self.nearby:
        self.zeroLocationsRef = np.where(np.all(self.referenceChr == 0, axis = 1))[0]
        self.setOfZeroLocations = set(self.zeroLocationsRef)
      self.coverage = None
      if self.load_coverage:
        self.coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(chromosome))
      self.setOfIndelLocations = set(self.indelLocations)
      self.prevChosenRefLocations = set()
      nearby_indels[total_length_per_chrom*(chromosome - 1) : total_length_per_chrom*(chromosome - 1) + lengthIndels_per_chrom] = self.indelLocations

      # dataset should have all the indels as well as random negative training samples
      if self.nearby:
        neg_positions = np.random.choice(self.indelLocations, size=num_negatives_per_chrom)
        nearby_indels[total_length_per_chrom*(chromosome - 1) + lengthIndels_per_chrom : total_length_per_chrom*chromosome] = neg_positions
        offset = np.multiply(np.random.randint(1, self.nearby+1, size=num_negatives_per_chrom), np.random.choice([-1, 1], size=num_negatives_per_chrom))
        neg_positions = neg_positions + offset # locations that are offset from indels by some amount
      else:
        neg_positions = np.random.choice(self.nonzeroLocationsRef, size=num_negatives_per_chrom)
        self.nearby_indels = neg_positions # to prevent error if this is undefined
      for i in range(lengthIndels_per_chrom + num_negatives_per_chrom):
        if i < lengthIndels_per_chrom:
          if not self.triclass:
            label = 1 # standard binary classification labels
          elif i < len(self.insertionLocations):
            label = 1 # insertions will be labeled as 1
          else:
            label = 2 # deletions will be labeled as 2
          pos = self.indelLocations[i]
          allele_count[total_length_per_chrom*(chromosome - 1) + i] = allele_count_val[i]
        else:
          label = 0
          pos = neg_positions[i - lengthIndels_per_chrom]
          if self.nearby:
            niter = 0
            while (pos in self.prevChosenRefLocations) or (pos in self.setOfZeroLocations) or (pos in self.setOfIndelLocations) and niter < 1001:
              nearby_indels[total_length_per_chrom*(chromosome - 1) + i] = np.random.choice(self.indelLocations)
              pos = nearby_indels[total_length_per_chrom*(chromosome - 1) + i] + np.random.randint(1, self.nearby+1) * np.random.choice([-1, 1])
              niter += 1
          else:
            while (pos in self.prevChosenRefLocations) or (pos in self.setOfIndelLocations) or entropy.entropySequence(self.referenceChr[pos - k_seq_complexity : pos + k_seq_complexity + 1]) < self.complexity_threshold:
              pos = np.random.choice(self.nonzeroLocationsRef)
          self.prevChosenRefLocations.add(pos)
        indices[total_length_per_chrom*(chromosome - 1) + i] = pos
        coverageWindow = np.zeros(2*k + 1)
        # get k base pairs before and after the position
        window = self.referenceChr[pos - k : pos + k + 1]
        coverageWindow = None
        if self.coverage is not None:
          coverageWindow = utils.flatten(self.coverage[pos - k : pos + k + 1])
        dataset[total_length_per_chrom*(chromosome - 1) + i] = window
        coverageDataset[total_length_per_chrom*(chromosome - 1) + i] = coverageWindow
        labels[total_length_per_chrom*(chromosome - 1) + i] = label
        genome_positions[total_length_per_chrom*(chromosome - 1) + i] = pos
        chrom_num[total_length_per_chrom*(chromosome - 1) + i] = chromosome
    if self.load_entropy:
      entropyDataset[:, k+1:2*k+1] = entropy.entropyVector(dataset)
    # Set the number of training examples, validation examples (then the test example numbers is automatically set)
    self.num_train_examples = total_length_per_chrom*(num_chrom_used - 2)
    self.num_val_examples = total_length_per_chrom
    # Randomly choose the validation and test chromosome
    self.val_chrom, self.test_chrom = np.random.choice(23, 2, replace=False) + 1
    # Shuffle the list and then populate dataset and other variables
    # The first self.num_train_examples are the training examples, the next self.num_val_examples are validation examples, and the remaining are test examples
    # This is very verbose, maybe it can be written in a neater manner
    indices_train = np.logical_and(chrom_num != self.val_chrom, chrom_num != self.test_chrom)
    rawZipped = zip(list(dataset[indices_train, :, :]), list(coverageDataset[indices_train, :]), list(labels[indices_train]), list(genome_positions[indices_train]), list(indices[indices_train]), list(nearby_indels[indices_train]), list(entropyDataset[indices_train, :]), list(allele_count[indices_train]))
    del indices_train
    np.random.shuffle(rawZipped)
    a, b, c, d, e, f, g, h = zip(*rawZipped)
    self.dataset = np.zeros((total_length, 2*k + 1, 4))
    self.coverageDataset = np.zeros((total_length, 2*k + 1))
    self.entropyDataset = np.zeros((total_length, 2*k + 1))
    self.indices = np.zeros(total_length, dtype=np.uint32)
    self.allele_count = np.zeros(total_length, dtype=np.uint32)
    self.nearby_indels = np.zeros(total_length, dtype=np.uint32)
    self.genome_positions = np.zeros(total_length, dtype=np.uint32)
    self.dataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :, :] = dataset[chrom_num == self.val_chrom, :, :]
    self.dataset[self.num_train_examples + self.num_val_examples:, :, :] = dataset[chrom_num == self.test_chrom, :, :]
    self.dataset[:self.num_train_examples, :, :] = np.array(a)
    del dataset, a
    self.coverageDataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :] = coverageDataset[chrom_num == self.val_chrom, :]
    self.coverageDataset[self.num_train_examples + self.num_val_examples:, :] = coverageDataset[chrom_num == self.test_chrom, :]
    self.coverageDataset[:self.num_train_examples, :] = np.array(b)
    del coverageDataset, b
    self.entropyDataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :] = entropyDataset[chrom_num == self.val_chrom, :]
    self.entropyDataset[self.num_train_examples + self.num_val_examples:, :] = entropyDataset[chrom_num == self.test_chrom, :]
    self.entropyDataset[:self.num_train_examples, :] = np.array(g)
    del entropyDataset, g
    labels_new = np.zeros(total_length, dtype=labeltype)
    labels_new[self.num_train_examples:self.num_train_examples + self.num_val_examples] = labels[chrom_num == self.val_chrom]
    labels_new[self.num_train_examples + self.num_val_examples:] = labels[chrom_num == self.test_chrom]
    labels_new[:self.num_train_examples] = np.array(c)
    if self.triclass:
      self.labels = utils.to_onehot(labels_new, 3)
    else:
      self.labels = np.expand_dims(labels_new, axis=1)
    del labels_new, labels, c
    self.genome_positions[self.num_train_examples:self.num_train_examples + self.num_val_examples] = genome_positions[chrom_num == self.val_chrom]
    self.genome_positions[self.num_train_examples + self.num_val_examples:] = genome_positions[chrom_num == self.test_chrom]
    self.genome_positions[:self.num_train_examples] = np.array(d)
    del genome_positions, d
    self.indices[self.num_train_examples:self.num_train_examples + self.num_val_examples] = indices[chrom_num == self.val_chrom]
    self.indices[self.num_train_examples + self.num_val_examples:] = indices[chrom_num == self.test_chrom]
    self.indices[:self.num_train_examples] = np.array(e)
    del indices, e
    self.nearby_indels[self.num_train_examples:self.num_train_examples + self.num_val_examples] = nearby_indels[chrom_num == self.val_chrom]
    self.nearby_indels[self.num_train_examples + self.num_val_examples:] = nearby_indels[chrom_num == self.test_chrom]
    self.nearby_indels[:self.num_train_examples] = np.array(f)
    del nearby_indels, f
    self.allele_count[self.num_train_examples:self.num_train_examples + self.num_val_examples] = allele_count[chrom_num == self.val_chrom]
    self.allele_count[self.num_train_examples + self.num_val_examples:] = allele_count[chrom_num == self.test_chrom]
    self.allele_count[:self.num_train_examples] = np.array(h)
    self.allele_count_test = self.allele_count[self.num_train_examples + self.num_val_examples:]
    del allele_count, h
    self.ordering = list(range(0, self.num_train_examples))

  def get_batch(self):
    return self.get_randbatch() # default: random batch

  # Get the next batch of training examples. The training data is shuffled after every epoch.
  def get_randbatch(self, batchSize=0):
    if batchSize == 0: batchSize = self.batchSize
    # Randomize the order of examples, if we are at the beginning of the next epoch
    if self.cur_index == 0:
      np.random.shuffle(self.ordering)
    start, end = self.cur_index, self.cur_index + batchSize
    batch_indices = self.ordering[start : end] # Indices of the training examples that will make up the batch
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
    test_data_x = self.dataset[self.num_train_examples + self.num_val_examples:]
    test_data_y = self.labels[self.num_train_examples + self.num_val_examples:]
    if self.coverage is not None and self.load_entropy:
      test_data_coverage = self.coverageDataset[self.num_train_examples + self.num_val_examples:]
      test_data_entropy = self.entropyDataset[self.num_train_examples + self.num_val_examples:]
      self.test_data = test_data_x, test_data_coverage, test_data_entropy, test_data_y
    elif self.coverage is not None:
      test_data_coverage = self.coverageDataset[self.num_train_examples + self.num_val_examples:]
      self.test_data = test_data_x, test_data_coverage, test_data_y
    elif self.load_entropy:
      test_data_entropy = self.entropyDataset[self.num_train_examples + self.num_val_examples:]
      self.test_data = test_data_x, test_data_entropy, test_data_y
    else:
      self.test_data = test_data_x, test_data_y
    print("Number of test examples: {}".format(len(test_data_y)))

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
      if self.coverage is not None and self.load_entropy:
        rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[2][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[3][self.test_index:self.test_index+self.testBatchSize]
      elif self.coverage is not None:
        rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[2][self.test_index:self.test_index+self.testBatchSize]
      elif self.load_entropy:
        rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[1][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[2][self.test_index:self.test_index+self.testBatchSize]
      else:
        rv = self.test_data[0][self.test_index:self.test_index+self.testBatchSize], \
           self.test_data[1][self.test_index:self.test_index+self.testBatchSize]
    else:
      raise RuntimeError("test index is {}, only {} examples".format(self.test_index, len(self.test_data[1])))
    self.test_index += self.testBatchSize
    return rv

  # Get the validation set
  def val_set(self, length=1000):
    val_indices = range(self.num_train_examples, self.num_train_examples + length)
    if self.coverage is not None and self.load_entropy:
      return self.dataset[val_indices, :, :], self.coverageDataset[val_indices, :], self.entropyDataset[val_indices, :], self.labels[val_indices]
    elif self.coverage is not None:
      return self.dataset[val_indices, :, :], self.coverageDataset[val_indices, :], self.labels[val_indices]
    elif self.load_entropy:
      return self.dataset[val_indices, :, :], self.entropyDataset[val_indices, :], self.labels[val_indices]
    else:
      return self.dataset[val_indices, :, :], self.labels[val_indices]

  def load_chromosome_window_data(self):
    chromosome = self.test_chrom
    self.referenceChr = self.referenceChrFull[str(chromosome)]
    if self.triclass:
      self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
      self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
      self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations)) - self.offset
    else:
      indel_data_load = np.loadtxt(data_dir + "indelLocationsFiltered{}".format(chromosome) + ext).astype(int) # This is a 5 column data: indel locations, allele count, filter value, 50 window, and 20 window sequence complexity
      self.indelLocationsFull = indel_data_load[:, 0] # Even non-filtered ones are a part of indelLocationsFull, so that we don't put these in the negative examples even by chance
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
    ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
    num_ex = ub - lb
    X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
    self.chrom_index = ub
    labels = [ex in self.setOfIndelLocations for ex in range(lb, ub)]
    return self.referenceChr[X+Y, :], labels, lb, ub

  def load_chromosome_window_batch_modified(self, window_size, batch_size):
    lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
    ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
    num_ex = ub - lb
    X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
    self.chrom_index = ub
    k = window_size
    k_seq_complexity = 20
    indelList = [x in loader.setOfIndelLocations for x in range(lb, ub)]
    seqDataset = self.referenceChr[X+Y, :]
    complexity = entropy.entropySequence(seqDataset[:, k - k_seq_complexity : k + k_seq_complexity + 1, :])
    indexList = [complexity[x - lb] >= self.complexity_threshold for x in range(lb, ub)]
    seqDataset = seqDataset[indexList]
    return seqDataset, indelList, indexList