import pandas as pd
import numpy as np
import cs273b
import entropy

np.random.seed(1)

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
  print "Processing chromosome {}".format(chromosome)
  # Read the indels in this chromosome
  locations_indels = np.array(indels_file[indels_file.columns[0]] == chromosome)
  indel_indices = np.array(indels_file.iloc[locations_indels, 1], dtype = int)
  # Reference chromosome
  reference_chrom = reference[str(chromosome)]
  ##
  # Store insertions or deletions information 
  ins_del = np.zeros(len(indel_indices), dtype = int)
  ins_del_indices = np.arange(len(locations_indels))
  ins_del_indices = ins_del_indices[locations_indels]
  for i in range(len(indel_indices)):
    loc = ins_del_indices[i]
    if(len(list(indels_file.iloc[loc, 3])) > len(list(indels_file.iloc[loc, 2]))): # 'Alt' longer than 'ref': insertion
      ins_del[i] = 1
  ##
  # Save the indels data
  allele_count = np.array(indels_file.iloc[locations_indels, 7], dtype = int)
  filter_value = np.array(indels_file.iloc[locations_indels, 4], dtype = int)
  arr = np.concatenate((np.expand_dims(indel_indices, axis = 1), np.expand_dims(allele_count, axis = 1), np.expand_dims(filter_value, axis = 1), np.expand_dims(ins_del, axis = 1)), axis = 1)
  np.save(filename_base + str(chromosome) + '.npy', arr)
  ##
  # Sample and save a set of strict non-indels
  nonzeroLocationsRef = np.where(np.any(reference_chrom != 0, axis = 1))[0]
  rel_size_neg_large = 1.5
  # Remove positions that have any indels around their window
  window_containing_indel = np.arange(2*window1 + 1) - window1
  window_containing_indel = np.repeat(window_containing_indel, len(indel_indices), axis = 0)
  window_containing_indel = np.reshape(window_containing_indel, [-1, len(indel_indices)])
  window_containing_indel += np.transpose(indel_indices)
  neg_positions_large = np.random.choice(list(set(nonzeroLocationsRef) - set(np.reshape(window_containing_indel, -1))), size = int(rel_size_neg_large*len(indel_indices)), replace = False)
  np.save(filename_base_2 + str(chromosome) + '.npy', neg_positions_large)
