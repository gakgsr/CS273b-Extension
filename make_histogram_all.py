import numpy as np
from sys import argv
import cs273b

data_dir = '/datadrive/project_data/'
freqs_outer_full = {}
freqs_outer = {}
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
bucketsize = 10000000

for chromosome in range(23):
  if chromosome == 0:
    chromosome = 'X'

  print('Loading chromosome {}'.format(chromosome))
  referenceChr = reference[str(chromosome)]
  c_len = len(referenceChr)

  insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(chromosome)).astype(int)
  deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(chromosome)).astype(int)
  indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1

  num_buckets = (c_len + bucketsize - 1) // bucketsize
  num_indels = [0]*num_buckets
  bucketsizes = [bucketsize]*num_buckets
  bucketsizes[-1] = c_len % bucketsize

  for il in indelLocations:
    num_indels[il / bucketsize] += 1

  bucketranges = []
  csum = 0
  for i in range(len(bucketsizes)):
    temp = csum + bucketsizes[i]
    bucketranges.append('{}-{}'.format(csum, temp-1))
    csum = temp

  freqs = [(float(x)/y, x, y, z) for x, y, z in zip(num_indels, bucketsizes, bucketranges)]

  freqs_outer_full[chromosome] = freqs

for k in freqs_outer_full.keys():
  freqs_outer[k] = np.array([x[0] for x in freqs_outer_full[k]])
  print(k)
  print(freqs_outer[k])
  print('')
