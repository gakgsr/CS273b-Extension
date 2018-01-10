import cs273b
import load_coverage as lc
import load_recombination as lr
import numpy as np
from math import ceil
import utils
import entropy

# Base location of all input data files
data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
  def __init__(self, _kw=0, chromosome=21, windowSize=100, batchSize=100, testBatchSize=500, seed=1, test_frac=0.05, pos_frac=0.5, load_coverage=True, load_entropy=False, load_recombination=False, include_filtered=True, triclass=False, nearby=0, offset=0, load_entire=True, delref=True):
    self.window = windowSize # If window size k, means we read k base pairs before the center and k after, for a total of 2k+1 base pairs in the input
    self.batchSize = batchSize # Number of training examples per batch
    self.testBatchSize = testBatchSize # Number of testing examples per test batch (we can't test everything at once due to memory)
    self.test_frac = test_frac # Fraction of data used for testing
    self.triclass = triclass # Whether to do tri-class classification (Insertion, Deletion, or neither) as opposed to binary (Indel or non-indel)
    self.nearby = nearby # If nearby is nonzero, negative examples are only sampled from within 'nearby' of some positive example. Otherwise, they are sampled at random from the genome.
    self.offset = offset # Either 0 or 1, to handle 1-indexing of the gnomad_indels.tsv file. Technically should be 1, but in practice 0 seems to work just as well??
    self.load_entropy = load_entropy # Whether to use calculated sequence entropy as input to the model
    self.load_coverage = load_coverage # Whether to use coverage data as input to the model
    self.load_recombination = load_recombination # Whether to use recombination data as input to the model
    reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp") # Load the reference genome
    self.referenceChr = reference[str(chromosome)] # Pick out the sequence data for the chromosome of interest
    self.refChrLen = len(self.referenceChr)
    del reference, ambiguous_bases # Preserve memory
    ext = ".txt"
    if not include_filtered: ext = "_filtered" + ext # If include_filtered is false, filtered examples are excluded from the set of positive indel examples
    if self.triclass:
      self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
      self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
      self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations))
    else:
      self.indelLocations = np.loadtxt(data_dir + "indelLocations{}".format(chromosome) + ext).astype(int)
    self.nonzeroLocationsRef = np.where(np.any(self.referenceChr != 0, axis = 1))[0] # Locations where the reference is nonzero (if zero, means that that base is missing/uncertain)
    if nearby:
      self.zeroLocationsRef = np.where(np.all(self.referenceChr == 0, axis = 1))[0] # Locations where the reference sequence is zero
      self.setOfZeroLocations = set(self.zeroLocationsRef)
    self.indelLocations = self.indelLocations - offset
    self.coverage = None
    if load_coverage:
      self.coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(chromosome))
    self.recombination = None
    if load_recombination:
      self.recombination = lr.load_recombination(data_dir + "recombination_map/genetic_map_chr{}_combined_b37.txt".format(chromosome))
    self.setOfIndelLocations = set(self.indelLocations)
    self.prevChosenRefLocations = set()
    self.cur_index = 0 # Index of next training example (for batching purposes)
    self.test_index = 0 # Index of next testing example (for batching purposes)
    self.chrom_index = 0 # Index of next location in the chromosome (for load_chromosome_window_batch function)
    if seed is not None:
      np.random.seed(seed)
    self.__initializeTrainData(pos_frac)
    self.__initializeTestData()
    if not load_entire: # If we don't need the sequence data once our train and test set are initialized, we can delete it
      del self.referenceChr
      del self.nonzeroLocationsRef

  # Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
  # Also includes desired number of negative training examples (positions not listed as indels)
  def __initializeTrainData(self, frac_positives):
    k = self.window # for brevity
    lengthIndels = len(self.indelLocations) # Total number of indels
    num_negatives = int((1./frac_positives-1) * lengthIndels) # Total number of negative training examples we need, based on the desired fraction of positive examples
    total_length = lengthIndels + num_negatives # Total number of examples [both training and testing!]
    dataset = np.zeros((total_length, 2*k + 1, 4))
    coverageDataset = np.zeros((total_length, 2*k + 1))
    entropyDataset = np.zeros((total_length, 2*k + 1))
    recombinationDataset = np.zeros((total_length, 1))
    #recombinationDataset = np.zeros((total_length, 2*k + 1))
    if self.triclass:
      labeltype = np.uint8 # Three distinct labels in this case
    else:
      labeltype = np.bool
    labels = np.zeros(total_length, dtype=labeltype)
    genome_positions = np.zeros(total_length, dtype=np.uint32)

    # dataset should have all the indels as well as random negative training samples
    if self.nearby:
      neg_positions = np.random.choice(self.indelLocations, size=num_negatives) # First choose a random number of examples among known indels
      self.nearby_indels = neg_positions # Store the locations of these selected indels
      offset = np.multiply(np.random.randint(1, self.nearby+1, size=num_negatives), np.random.choice([-1, 1], size=num_negatives)) # Offset by a random nonzero amount <= to self.nearby
      neg_positions = neg_positions + offset # These locations that are offset from indels by some amount are [roughly] our negative examples; but see for loop below
    else:
      neg_positions = np.random.choice(self.nonzeroLocationsRef, size=num_negatives) # Select random nonzero locations from the reference genomes
      self.nearby_indels = neg_positions # to prevent error if this is undefined (value should not be used as it is meaningless in this case)
    self.indices = neg_positions # Locations of the negative training examples
    for i in range(lengthIndels + num_negatives): # Loop over all examples
      if i < lengthIndels: # Positive example
        if not self.triclass:
          label = 1 # standard binary classification labels
        elif i < len(self.insertionLocations):
          label = 1 # insertions will be labeled as 1
        else:
          label = 2 # deletions will be labeled as 2
        pos = self.indelLocations[i]
      else: # Negative example (not an indel)
        label = 0
        pos = neg_positions[i - lengthIndels] # Get corresponding entry of neg_positions, which stores the tentative positions of all negative examples. However, we may need to update this position. We still predefine them and update if needed simply for efficiency's sake.
        if self.nearby: # Position must be near a known indel
          niter = 0 # Avoid infinite loops (probably should still make sure the selected position is not at an indel location (or zero?) regardless of # iterations in the below condition, though)
          while (pos in self.prevChosenRefLocations) or (pos in self.setOfZeroLocations) or (pos in self.setOfIndelLocations) and niter < 1001:
            # Avoid choosing an already selected position, a zero location (unknown reference base), or an actual indel
            self.nearby_indels[i - lengthIndels] = np.random.choice(self.indelLocations) # Select again using the same procedure, until we get a valid negative example
            pos = self.nearby_indels[i - lengthIndels] + np.random.randint(1, self.nearby+1) * np.random.choice([-1, 1])
            niter += 1
        else: # Position simply just has to not be previously selected, and not a positive (i.e. indel) example
          while (pos in self.prevChosenRefLocations) or (pos in self.setOfIndelLocations):
            pos = np.random.choice(self.nonzeroLocationsRef)
        self.indices[i - lengthIndels] = pos # True position of the negative example
        self.prevChosenRefLocations.add(pos) # Store this position, so we don't reuse it
      # get the k base pairs before and after the position, and the position itself
      window = self.referenceChr[pos - k : pos + k + 1]
      coverageWindow = None # Coverage window corresponding to the input base pairs (loaded only if necessary)
      if self.coverage is not None:
        coverageWindow = utils.flatten(self.coverage[pos - k : pos + k + 1])
      recombWindowAverage = None
      if self.recombination is not None: # Recombination window, if needed
        recombWindow = np.zeros((2*k + 1, 1))
        recombWindowIndices = np.arange(pos - k, pos + k + 1).reshape((2*k + 1, 1))
        recombInBounds = recombWindowIndices[np.where(recombWindowIndices < len(self.recombination))]
        recombWindow[recombInBounds - (pos - k)] = self.recombination[recombInBounds]
        recombOutOfBounds = recombWindowIndices[np.where(recombWindowIndices >= len(self.recombination))]
        recombWindow[recombOutOfBounds - (pos - k)] = self.recombination[-1] 
        recombWindowAverage = np.mean(recombWindow)
        #recombWindowAverage = utils.flatten(recombWindow)
      dataset[i] = window # Store the data for this example in the overall data structure
      coverageDataset[i] = coverageWindow
      recombinationDataset[i] = recombWindowAverage
      labels[i] = label
      genome_positions[i] = pos # This might be the same as self.indices?
    self.indices = np.concatenate((self.indelLocations, self.indices)) # Indices for positive examples are simply in self.indelLocations
    self.nearby_indels = np.concatenate((self.indelLocations, self.nearby_indels)) # "Nearby" indel for a positive example is the indel itself
    if self.load_entropy:
      entropyDataset[:, k+1:2*k+1] = entropy.entropyVector(dataset) # Create the entropy vectors, if needed
    rawZipped = zip(list(dataset), list(coverageDataset), list(labels), list(genome_positions), list(self.indices), list(self.nearby_indels), list(entropyDataset), list(recombinationDataset))
    # Shuffle the data
    np.random.shuffle(rawZipped)
    a, b, c, d, e, f, g, h = zip(*rawZipped)
    dataset = np.array(a)
    coverageDataset = np.array(b)
    entropyDataset = np.array(g)
    recombinationDataset = np.array(h)
    labels = np.array(c, dtype=labeltype)
    genome_positions = np.array(d, dtype=np.uint32)
    self.indices = np.array(e, dtype=np.uint32)
    self.nearby_indels = np.array(f, dtype=np.uint32)
    self.dataset = dataset
    self.coverageDataset = coverageDataset
    self.entropyDataset = entropyDataset
    self.recombinationDataset = recombinationDataset
    if self.triclass:
      self.labels = utils.to_onehot(labels, 3)
    else:
      self.labels = np.expand_dims(labels, axis=1) # Make labels n by 1 (for convenience)
    self.genome_positions = genome_positions
    self.num_train_examples = int(round(total_length * (1-self.test_frac))) # Number of examples to use for training (as opposed to testing)
    self.ordering = list(range(0, self.num_train_examples)) # Order in which we go through the training examples (will be changed)

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
    if self.load_recombination:
      retval.append(self.recombinationDataset[batch_indices])
    retval.append(self.labels[batch_indices])

    self.cur_index = end # Start of next batch
    if end >= self.num_train_examples:
      self.cur_index = 0 # Epoch just ended
    return tuple(retval)

  # Fairly self-explanatory: initialize the test data.
  def __initializeTestData(self):
    # Get all non-training examples
    test_data_x = self.dataset[self.num_train_examples+1:]
    test_data_y = self.labels[self.num_train_examples+1:]
    self.test_data = [test_data_x]
    if self.load_coverage:
      self.test_data.append(self.coverageDataset[self.num_train_examples+1:])
    if self.load_entropy:
      self.test_data.append(self.entropyDataset[self.num_train_examples+1:])
    if self.load_recombination:
      self.test_data.append(self.recombinationDataset[self.num_train_examples+1:])
    self.test_data.append(test_data_y)
    self.test_data = tuple(self.test_data)
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
      rv = [self.test_data[i][self.test_index:self.test_index+self.testBatchSize] for i in range(len(self.test_data))]
      rv = tuple(rv)
    else:
      raise RuntimeError("test index is {}, only {} examples".format(self.test_index, len(self.test_data[1])))
    self.test_index += self.testBatchSize
    return rv

  # Validation set. Definitely should not overlap the test set!!
  def val_set(self, length=1000):
    retval = [self.test_data[i][:length] for i in range(len(self.test_data))]
    return tuple(retval)

  # This method is used to load the entire chromosome, batch by batch. This will load the next batch. Currently it only loads the genome, not any of the additional data sources.
  def load_chromosome_window_batch(self, window_size, batch_size):
    lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)?
    ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, note that ub is not inclusive
    num_ex = ub - lb
    X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
    self.chrom_index = ub
    labels = [ex in self.setOfIndelLocations for ex in range(lb, ub)]
    return self.referenceChr[X+Y, :], labels, lb, ub
