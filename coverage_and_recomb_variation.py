""" Measures the standard deviations of coverage and recombination in a window over the means of the corresponding measurements in the window,
    as a way of assessing the amount of variability in each of these metrics.
"""

import numpy as np
from sys import argv
import load_coverage as lc
import load_recombination as lr
import cs273b

data_dir = '/datadrive/project_data/'
reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
del ambiguous_bases

k = 200
window_size = 2*k+1
windows_per_bin = 50
margin = 16
expanded_window_size = window_size + 2*margin

rel_vars_cvr_overall = []
rel_vars_rec_overall = []

outfile = 'coverage_and_recomb_variation_results.txt'
f = open(outfile, 'w')
for i in range(1, 24):
  if i == 23:
    ch = 'X'
    continue # Recombination data is in multiple files for chromosome X; we skip it for now
  else:
    ch = str(i)

  print('Processing Chromosome {}'.format(ch))
  referenceChr = reference[ch]
  c_len = len(referenceChr) # Total chromosome length
  num_windows = (c_len-2*margin)//window_size

  coverage = lc.load_coverage(data_dir + "coverage/{}.npy".format(ch))
  recombination = lr.load_recombination(data_dir + "recombination_map/genetic_map_chr{}_combined_b37.txt".format(ch), c_len)

  rel_vars_cvr = []
  rel_vars_rec = []
  for w in range(num_windows):
    # First window predictions start at index margin, but we include sequence context of length 'margin' around it, so its input array starts at index 0
    window_lb, window_ub = w*window_size, (w+1)*window_size + 2*margin # Include additional sequence context of length 'margin' around each window
    next_window = referenceChr[window_lb:window_ub]
    cvr = coverage[window_lb:window_ub]
    rec = recombination[window_lb:window_ub]
    cm, rm = np.mean(cvr), np.mean(rec)
    if cm: rel_vars_cvr.append(np.std(cvr) / cm)
    if rm: rel_vars_rec.append(np.std(rec) / rm)
  
  rel_vars_cvr.sort()
  rel_vars_rec.sort()
  cl = len(rel_vars_cvr)
  rl = len(rel_vars_rec)
  str1 = '=====CHROMOSOME {} RELATIVE DECILES====='.format(ch)
  str2 = 'Coverage: ' + ', '.join(['{}: {:.3e}'.format(i, rel_vars_cvr[max(0,(i*cl)//100-1)]) for i in range(0, 101, 10)])
  str3 = 'Recombination: ' + ', '.join(['{}: {:.3e}'.format(i, rel_vars_rec[max(0,(i*rl)//100-1)]) for i in range(0, 101, 10)])
  print(str1)
  print(str2)
  print(str3)
  f.write('\n'.join([str1, str2, str3]) + '\n')
  rel_vars_cvr_overall.extend(rel_vars_cvr)
  rel_vars_rec_overall.extend(rel_vars_rec)

rel_vars_cvr_overall.sort()
rel_vars_rec_overall.sort()
str1 = '=====OVERALL RELATIVE DECILES====='
str2 = 'Coverage: ' + ', '.join(['{}: {:.3e}'.format(i, rel_vars_cvr_overall[max(0,(i*cl)//100-1)]) for i in range(0, 101, 10)])
str3 = 'Recombination: ' + ', '.join(['{}: {:.3e}'.format(i, rel_vars_rec_overall[max(0,(i*rl)//100-1)]) for i in range(0, 101, 10)])
print(str1)
print(str2)
print(str3)
f.write('\n'.join([str1, str2, str3]) + '\n')

f.close()
