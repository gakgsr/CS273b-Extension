import numpy as np
from sys import argv
import cs273b

data_dir = '/datadrive/project_data/'

insFreqs = []
delFreqs = []
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
for i in range(1, 24):
  if i == 23:
    ch = 'X'
  else:
    ch = str(i)
  print('Processing ' + ch)
  referenceChr = reference[ch]
  c_len = len(referenceChr)

  insertionLocations = np.loadtxt(data_dir + "indelLocations{}_ins.txt".format(ch)).astype(int)
  deletionLocations = np.loadtxt(data_dir + "indelLocations{}_del.txt".format(ch)).astype(int)
  #indelLocations = np.concatenate((insertionLocations, deletionLocations)) - 1

  insFreq = float(len(insertionLocations)) / c_len
  delFreq = float(len(deletionLocations)) / c_len
  insFreqs.append(insFreq)
  delFreqs.append(delFreq)
  continue

  bucketsize = 1000000
  num_buckets = (c_len + bucketsize - 1) // bucketsize
  num_indels = [0]*num_buckets
  bucketsizes = [bucketsize]*num_buckets
  bucketsizes[-1] = c_len % bucketsize

  for il in indelLocations:
    num_indels[il / bucketsize] += 1

  freqs = [float(x)/y for x, y in zip(num_indels, bucketsizes)]

  print(np.array(freqs))

  #print(num_indels)
  #print(bucketsizes)

print(insFreqs)
print(delFreqs)
