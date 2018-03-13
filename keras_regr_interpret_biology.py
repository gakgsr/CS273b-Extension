import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
import pandas as pd
import cs273b

data_dir = '/datadrive/project_data/'
validation_chrom = 8
k = 200
window_size = 2*k+1
windows_per_bin = 1
margin = 15
expanded_window_size = window_size + 2*margin
complexity_thresh = 0

reference, ambiguous_bases = cs273b.load_bitpacked_reference(data_dir + "Homo_sapiens_assembly19.fasta.bp")
val_chrom_len = len(reference[str(validation_chrom)])
del reference, ambiguous_bases

#x_test = np.load(data_dir + 'RegrKerasTestSeq' + str(validation_chrom) + str(complexity_thresh) + '.npy')
y_test = np.load(data_dir + 'RegrKerasTestLab' + str(validation_chrom) + '.npy')
y_pred = np.load(data_dir + 'RegrKerasTestLabPred' + str(validation_chrom) + '.npy')

cdng = pd.read_csv(data_dir + "hinv70Coding.txt", sep = '\t', header = None)
cdng_val = cdng[cdng[1] == 'chr' + str(validation_chrom)]
del cdng

cdng_regions = np.zeros(val_chrom_len, dtype = bool)
for i in range(len(cdng_val)):
  strt = int(cdng_val.iloc[i, 2])
  cdng_lens = cdng_val.iloc[i, 11].split(',')
  cdng_strt = cdng_val.iloc[i, 12].split(',')
  for j in range(len(cdng_strt)-1):
    cdng_regions[strt+int(cdng_strt[j]):strt+int(cdng_strt[j])+int(cdng_lens[j])] = 1

cdng_frac = np.sum(cdng_regions, dtype = float)/val_chrom_len
print "Fraction of coding length is %f" % cdng_frac

cent_regions = np.zeros(val_chrom_len, dtype = bool) + 1
cent_regions[44033744:45877265] = 0
cent_frac = np.sum(cent_regions, dtype = float)/val_chrom_len
print "Fraction of centromere length is %f" % cent_frac

over_pred, over_pred_cdng, over_pred_cent = 0, 0, 0
undr_pred, undr_pred_cdng, undr_pred_cent = 0, 0, 0
c = ['b']*len(y_test)
f = 0.9
for i in range(len(y_test)):
  if(y_pred[i] > 1.1*y_test[i]):
    over_pred += 1
    if(np.sum(cdng_regions[i*window_size + margin:(i+1)*window_size + margin]) <= f*window_size):
      over_pred_cdng += 1
  if(y_pred[i] < 0.9*y_test[i]):
    undr_pred += 1
    if(np.sum(cdng_regions[i*window_size + margin:(i+1)*window_size + margin]) <= f*window_size):
      undr_pred_cdng += 1

for i in range(len(y_test)):
  if(np.sum(cent_regions[i*window_size + margin:(i+1)*window_size + margin]) <= f*window_size):
    if(y_pred[i] != 0 or y_test[i] != 0):
      print "Non zero element"
    if y_pred[i] > 1.1*y_test[i]:
      over_pred_cent += 1
    if y_pred[i] < 0.9*y_test[i]:
      undr_pred_cent += 1


print "Ratio of coding regions in over predictions is:"
print float(over_pred_cdng)/over_pred
print "Ratio of coding regions in under predictions is:"
print float(undr_pred_cdng)/undr_pred
print "Ratio of centromere regions in over predictions is:"
print float(over_pred_cent)/over_pred
print "Ratio of centromere regions in under predictions is:"
print float(undr_pred_cent)/undr_pred

regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(y_test, axis=1), y_pred)
reg_pred = regr.predict(np.expand_dims(y_test, axis=1))

plt.scatter(y_test, y_pred, c = c)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation counts')
line_x = np.arange(min(np.amax(y_pred), np.amax(y_pred)))
plt.plot(line_x, line_x, color='m', linewidth=2.5)
plt.plot(y_test, reg_pred, color='g', linewidth=2.5)
#plt.savefig('indel_rate_pred_keras_non_coding_rgn_cent' + str(validation_chrom) + str(complexity_thresh) + '_' + str(windows_per_bin) + '.png')
