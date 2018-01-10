import cs273b
import load_coverage as lc
import numpy as np
from math import ceil
import utils
import entropy

data_dir = "/datadrive/project_data/"

class DatasetLoader(object):
	def __init__(self, _kw=0, windowSize=100, batchSize=100, testBatchSize=500, seed=1, test_frac=0.05, pos_frac=0.5, load_coverage=True, load_entropy=False, include_filtered=True, triclass=False, nearby=0, offset=0, load_entire = True):
		self.window = windowSize
		self.batchSize = batchSize
		self.testBatchSize = testBatchSize
		self.test_frac = test_frac
		self.triclass = triclass
		self.nearby = nearby
		self.offset = offset
		self.include_filtered = include_filtered
		self.load_entropy = load_entropy
		self.load_coverage = load_coverage
		self.referenceChrFull, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
		del ambiguous_bases
		self.cur_index = 0
		self.test_index = 0
		self.chrom_index = 0
		if seed is not None:
			np.random.seed(seed)
		self.__initializeTrainData(pos_frac)
		self.__initializeTestData()
		# Nimit ToDo -- in this new context, do the next 3 lines need to exist?
		if not load_entire:
			del self.referenceChrFull
			del self.nonzeroLocationsRef

	# Returns dataset in which each element is a 2D array: window of size k around indels: [(2k + 1) * 4 base pairs one-hot encoded]
	# Also includes desired number of negative training examples (positions not listed as indels)
	def __initializeTrainData(self, frac_positives):
		k = self.window # for brevity
		self.indelLocations = np.loadtxt(data_dir + "indelLocations21.txt").astype(int)
		lengthIndels = int(len(self.indelLocations)/22)*22# Does this need to be increases?
		num_negatives = int(int((1./frac_positives-1) * lengthIndels)/22)*22
		total_length = lengthIndels + num_negatives
		num_negatives_per_chrom = int(num_negatives/22)
		lengthIndels_per_chrom = int(lengthIndels/22)
		total_length_per_chrom = lengthIndels_per_chrom + num_negatives_per_chrom
		dataset = np.zeros((total_length, 2*k + 1, 4))
		coverageDataset = np.zeros((total_length, 2*k + 1))
		entropyDataset = np.zeros((total_length, 2*k + 1))
		indices = np.zeros(total_length, dtype=np.uint32)
		allele_count = np.zeros(total_length, dtype=np.uint32)
		nearby_indels = np.zeros(total_length, dtype=np.uint32)
		if self.triclass:
		  labeltype = np.uint8
		else:
		  labeltype = np.bool
		labels = np.zeros(total_length, dtype=labeltype)
		genome_positions = np.zeros(total_length, dtype=np.uint32)
		chrom_num = np.zeros(total_length, dtype=np.uint32)

		for chromosome in range(1, 23):
			self.referenceChr = self.referenceChrFull[str(chromosome)]
			self.refChrLen = len(self.referenceChr)
			ext = ".txt"
			if not self.include_filtered: ext = "_filtered" + ext
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
				indel_data_load = np.loadtxt(data_dir + "indelLocationsFiltered{}".format(chromosome) + ext).astype(int)
				self.indelLocationsFull = indel_data_load[:, 0]
				indel_indices = np.random.choice(np.arange(indel_data_load.shape[0]), size=lengthIndels_per_chrom, replace=False)
				self.indelLocations = indel_data_load[indel_indices, 0]
				allele_count_val = indel_data_load[indel_indices, 1]
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
					  while (pos in self.prevChosenRefLocations) or (pos in self.setOfIndelLocations):
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
		self.num_train_examples = total_length_per_chrom*20
		self.num_val_examples = total_length_per_chrom
		self.val_chrom, self.test_chrom = np.random.choice(23, 2, replace=False) + 1
		indices_train = np.logical_and(chrom_num != self.val_chrom, chrom_num != self.test_chrom)
		rawZipped = zip(list(dataset[indices_train, :, :]), list(coverageDataset[indices_train, :]), list(labels[indices_train]), list(genome_positions[indices_train]), list(indices[indices_train]), list(nearby_indels[indices_train]), list(entropyDataset[indices_train, :]), list(allele_count[indices_train]))
		# Shuffle the list
		np.random.shuffle(rawZipped)
		a, b, c, d, e, f, g, h = zip(*rawZipped)
		self.dataset = np.zeros((total_length, 2*k + 1, 4))
		self.coverageDataset = np.zeros((total_length, 2*k + 1))
		self.entropyDataset = np.zeros((total_length, 2*k + 1))
		self.indices = np.zeros(total_length, dtype=np.uint32)
		self.allele_count = np.zeros(total_length, dtype=np.uint32)
		self.nearby_indels = np.zeros(total_length, dtype=np.uint32)
		labels_new = np.zeros(total_length, dtype=labeltype)
		self.genome_positions = np.zeros(total_length, dtype=np.uint32)
		self.dataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :, :] = dataset[chrom_num == self.val_chrom, :, :]
		self.dataset[self.num_train_examples + self.num_val_examples:, :, :] = dataset[chrom_num == self.test_chrom, :, :]
		self.dataset[:self.num_train_examples, :, :] = np.array(a)
		self.coverageDataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :] = coverageDataset[chrom_num == self.val_chrom, :]
		self.coverageDataset[self.num_train_examples + self.num_val_examples:, :] = coverageDataset[chrom_num == self.test_chrom, :]
		self.coverageDataset[:self.num_train_examples, :] = np.array(b)
		self.entropyDataset[self.num_train_examples:self.num_train_examples + self.num_val_examples, :] = entropyDataset[chrom_num == self.val_chrom, :]
		self.entropyDataset[self.num_train_examples + self.num_val_examples:, :] = entropyDataset[chrom_num == self.test_chrom, :]
		self.entropyDataset[:self.num_train_examples, :] = np.array(g)
		labels_new[self.num_train_examples:self.num_train_examples + self.num_val_examples] = labels[chrom_num == self.val_chrom]
		labels_new[self.num_train_examples + self.num_val_examples:] = labels[chrom_num == self.test_chrom]
		labels_new[:self.num_train_examples] = np.array(c)
		print np.sum(labels_new == 0)
		print np.sum(labels_new != 0)
		self.genome_positions[self.num_train_examples:self.num_train_examples + self.num_val_examples] = genome_positions[chrom_num == self.val_chrom]
		self.genome_positions[self.num_train_examples + self.num_val_examples:] = genome_positions[chrom_num == self.test_chrom]
		self.genome_positions[:self.num_train_examples] = np.array(d)
		self.indices[self.num_train_examples:self.num_train_examples + self.num_val_examples] = indices[chrom_num == self.val_chrom]
		self.indices[self.num_train_examples + self.num_val_examples:] = indices[chrom_num == self.test_chrom]
		self.indices[:self.num_train_examples] = np.array(e)
		self.nearby_indels[self.num_train_examples:self.num_train_examples + self.num_val_examples] = nearby_indels[chrom_num == self.val_chrom]
		self.nearby_indels[self.num_train_examples + self.num_val_examples:] = nearby_indels[chrom_num == self.test_chrom]
		self.nearby_indels[:self.num_train_examples] = np.array(f)
		self.allele_count[self.num_train_examples:self.num_train_examples + self.num_val_examples] = allele_count[chrom_num == self.val_chrom]
		self.allele_count[self.num_train_examples + self.num_val_examples:] = allele_count[chrom_num == self.test_chrom]
		self.allele_count[:self.num_train_examples] = np.array(h)
		if self.triclass:
			self.labels = utils.to_onehot(labels_new, 3)
		else:
			self.labels = np.expand_dims(labels_new, axis=1)
		self.ordering = list(range(0, self.num_train_examples))
		self.allele_count_test = allele_count[self.num_train_examples + self.num_val_examples:]

	def get_batch(self):
		return self.get_randbatch() # default: random

	def get_randbatch(self, batchSize=0):
		if batchSize == 0: batchSize = self.batchSize
		# Randomize the order of batches
		if self.cur_index == 0:
			np.random.shuffle(self.ordering)
		start, end = self.cur_index, self.cur_index + batchSize
		batch_indices = self.ordering[start : end]
		if self.coverage is not None and self.load_entropy:
			retval = (self.dataset[batch_indices], self.coverageDataset[batch_indices], self.entropyDataset[batch_indices], self.labels[batch_indices])
		elif self.coverage is not None:
			retval = (self.dataset[batch_indices], self.coverageDataset[batch_indices], self.labels[batch_indices])
		elif self.load_entropy:
			retval = (self.dataset[batch_indices], self.entropyDataset[batch_indices], self.labels[batch_indices])
		else:
			retval = (self.dataset[batch_indices], self.labels[batch_indices])
		self.cur_index = end
		if end >= self.num_train_examples:
			self.cur_index = 0
		return retval

	def __initializeTestData(self):
		# Get all non-training examples
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

	def num_trainbatches(self):
		return int(ceil(float(self.num_train_examples) / self.batchSize))

	def len_testdata(self):
		return len(self.test_data[1])

	def num_testbatches(self):
		return int(ceil(float(len(self.test_data[1])) / self.testBatchSize))

	def reset_test_index(self):
		self.test_index = 0

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

	def load_chromosome_window_batch(self, window_size, batch_size):
		if self.chrom_index == 0:
			chromosome = self.test_chrom
			self.referenceChr = self.referenceChrFull[str(chromosome)]
			ext = ".txt"
			if not self.include_filtered: ext = "_filtered" + ext
			if self.triclass:
				self.insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins".format(chromosome) + ext).astype(int)
				self.deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del".format(chromosome) + ext).astype(int)
				self.indelLocations = np.concatenate((self.insertionLocations, self.deletionLocations)) - self.offset
			else:
				#self.indelLocationsFull = np.loadtxt(data_dir + "indelLocations{}".format(chromosome) + ext).astype(int)
				indel_data_load = np.loadtxt(data_dir + "indelLocationsFiltered{}".format(chromosome) + ext).astype(int)
				self.indelLocations = indel_data_load[:, 0] - self.offset
		lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
		ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
		num_ex = ub - lb
		X, Y = np.mgrid[lb - window_size : lb - window_size + num_ex, 0 : 2*window_size + 1]
		self.chrom_index = ub
		labels = [ex in self.setOfIndelLocations for ex in range(lb, ub)]
		return self.referenceChr[X+Y, :], labels, lb, ub

	def load_chromosome_window_batch_spaced(self, window_size, batch_size):
		lb = max(window_size, self.chrom_index) # we should probably instead pad with random values (actually may not be needed)
		ub = min(len(self.referenceChr) - window_size, lb + batch_size) # ditto to above. also, ub is not inclusive
		num_ex = ub - lb
		X, Y = np.mgrid[lb - window_size : 2*window_size + 1 : lb - window_size + num_ex, 0 : 2*window_size + 1]
		self.chrom_index = ub
		labels = [ex in self.setOfIndelLocations for ex in range(lb, 2*window_size + 1, ub)]
		return self.referenceChr[X+Y, :], labels, lb, ub