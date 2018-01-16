import pandas as pd
import numpy as np
import cs273b
import entropy

data_dir = "/datadrive/project_data/"

# Read the reference chromosome
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases
# Read the gnomad indels data file
indels_file = pd.read_csv(data_dir + "gnomad_indels.tsv", delimiter = '\t')
# Convert true or false filter value to 1 or 0
indels_file[indels_file.columns[4]] = indels_file[indels_file.columns[4]].astype(int)
# Output file names
filename_base = data_dir + "indelLocationsFiltered"
filename_base_2 = data_dir + "nonindelLocationsSampled"
# Windows to compute entropy. The smaller window is for defining sequence complexity, larger for input to CNN
window1 = 50
window2 = 20

# Process per chromosome, from 1-22
for chromosome in range(1, 23):
  ##
  # Read the indels in this chromosome
  locations_indels = np.array(indels_file[indels_file.columns[0]] == chromosome)
  indel_indices = np.array(indels_file.iloc[locations_indels, 1])
  # Reference chromosome
  reference_chrom = reference[str(chromosome)]
  ##
  # Create respective windows and measure complexity
  len_indel_indices = len(indel_indices)
  sequence_indices = np.arange(2*window2 + 1) - window2
  sequence_indices = np.repeat(sequence_indices, len_indel_indices, axis = 0)
  sequence_indices = np.reshape(sequence_indices, [-1, len_indel_indices])
  neg_sequence_indices += np.transpose(indel_indices)
  small_window_entropy = entropy.entropySequence(reference_chrom[sequence_indices, :])
  #
  sequence_indices = np.arange(2*window1 + 1) - window1
  sequence_indices = np.repeat(sequence_indices, len_indel_indices, axis = 0)
  sequence_indices = np.reshape(sequence_indices, [-1, len_indel_indices])
  neg_sequence_indices += np.transpose(indel_indices)
  large_window_entropy = entropy.entropySequence(reference_chrom[sequence_indices, :])
  ##
  # Store insertions or deletions information 
  ins_del = np.zeros(len_indel_indices, dtype = int)
  for i in range(len_indel_indices):
    loc = locations_indels[i]
    if(len(indels_file.iloc[loc, 3]) > len(indels_file.iloc[loc, 2])): # 'Alt' longer than 'ref': insertion
      ins_del[i] = 1
  ##
  # Save the indels data
  indels_for_chromosome_filtered = indels_file[locations_indels]
  indel_locations_chromosome_filtered = indels_for_chromosome_filtered[[indels_for_chromosome_filtered.columns[1], indels_for_chromosome_filtered.columns[7], indels_for_chromosome_filtered.columns[4]]]
  indel_locations_chromosome_filtered['LargeWindowComplexity'] = pd.Series(large_window_entropy, index=indel_locations_chromosome_filtered.index)
  indel_locations_chromosome_filtered['SmallWindowComplexity'] = pd.Series(small_window_entropy, index=indel_locations_chromosome_filtered.index)
  indel_locations_chromosome_filtered['InsertionDeletion'] = pd.Series(small_window_entropy, index=indel_locations_chromosome_filtered.index)
  indel_locations_chromosome_filtered.to_csv(filename_base + str(chromosome) + '.txt', sep = ' ', header = False, index = False)
  ##
  # Sample and save a set of strict non-indels
  nonzeroLocationsRef = np.where(np.any(reference_chrom != 0, axis = 1))[0]
  rel_size_neg_large = 2
  neg_positions_large = np.random.choice(list(set(nonzeroLocationsRef) - set(indelLocationsFull)), size = rel_size_neg_large*len_indel_indices, replace = False)
  np.savetxt(neg_positions_large, filename_base_2 + str(chromosome) + '.txt')