import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from sklearn import linear_model
from sequence_analysis import sequenceLogos, sequence_2_mer_generate, plot_seq_2_mer_freq

data_dir = '/datadrive/project_data/'
validation_chrom = 8
k = 200
window_size = 2*k+1
windows_per_bin = 1
margin = 15
expanded_window_size = window_size + 2*margin
complexity_thresh = 1.1

#x_test = np.load(data_dir + 'RegrKerasTestSeq' + str(validation_chrom) + str(complexity_thresh) + '.npy')
y_test = np.load(data_dir + 'RegrKerasTestLab' + str(validation_chrom) + str(complexity_thresh) + '.npy')
y_pred = np.load(data_dir + 'RegrKerasTestLabPred' + str(validation_chrom) + str(complexity_thresh) + '.npy')

# Average number of indels in the 401 sized window as threshold
threshold = 3
threshold2 = 5

print np.sum(y_test, dtype = float)/len(y_test)
print np.sum(y_pred, dtype = float)/len(y_pred)

'''

# Create 4 classes of predictions
true_labels = np.arange(len(y_test))
true_pos = true_labels[np.logical_and(y_test >= threshold2, y_pred >= threshold2)]
true_neg = true_labels[np.logical_and(y_test < threshold, y_pred < threshold)]
false_pos = true_labels[np.logical_and(y_test < threshold, y_pred >= threshold2)]
false_neg = true_labels[np.logical_and(y_test >= threshold2, y_pred < threshold)]


sequenceLogos(x_test[true_pos], 'KerasTruePos')
sequenceLogos(x_test[true_neg], 'KerasTrueNeg')
sequenceLogos(x_test[false_pos], 'KerasFalsePos')
sequenceLogos(x_test[false_neg], 'KerasFalseNeg')

freq_val_true_pos, _ = sequence_2_mer_generate(x_test[true_pos])
freq_val_true_neg, _ = sequence_2_mer_generate(x_test[true_neg])
freq_val_false_pos, _ = sequence_2_mer_generate(x_test[false_pos])
freq_val_false_neg, _ = sequence_2_mer_generate(x_test[false_neg])
plot_seq_2_mer_freq(freq_val_true_pos, 'KerasTruePos')
plot_seq_2_mer_freq(freq_val_true_neg, 'KerasTrueNeg')
plot_seq_2_mer_freq(freq_val_false_pos, 'KerasFalsePos')
plot_seq_2_mer_freq(freq_val_false_neg, 'KerasFalseNeg')
'''

# Compute the correlation between the test set predictions and the true values
r, p = stats.pearsonr(y_test, y_pred)
print('')
print('r value: {}'.format(r))
print('p value: {}'.format(p))

bin_preds, bin_trues = [], []
for i in range(len(y_test)//windows_per_bin):
  pred_agg, true_agg = 0, 0
  for j in range(i*windows_per_bin, i*windows_per_bin + windows_per_bin):
    pred_agg += y_pred[j]
    true_agg += y_test[j]
  bin_preds.append(pred_agg)
  bin_trues.append(true_agg)

bin_preds, bin_trues = np.array(bin_preds), np.array(bin_trues)

mae = np.mean(np.abs(bin_preds - bin_trues))
rms = np.sqrt(np.mean(np.square(bin_preds - bin_trues)))
r_bin, p_bin = stats.pearsonr(bin_trues, bin_preds)
avg_pred = np.mean(bin_preds)
avg_true = np.mean(bin_trues)

print('Bin size: {}, MAE: {}, RMS error: {}, r: {}, p-value: {}, average indels predicted: {}, average indels actual: {}'.format(windows_per_bin*window_size, mae, rms, r_bin, p_bin, avg_pred, avg_true))


regr = linear_model.LinearRegression()
regr.fit(np.expand_dims(bin_trues, axis=1), bin_preds)
reg_pred = regr.predict(np.expand_dims(bin_trues, axis=1))

plt.scatter(bin_trues, bin_preds)
plt.xlabel('True number of indels')
plt.ylabel('Predicted number of indels')
plt.title('Predicted vs. actual indel mutation counts ($r = {:.2f}'.format(r) + (', p =$ {:.2g})'.format(p) if p else ', p < 10^{-15})$'))
line_x = np.arange(min(np.amax(bin_preds), np.amax(bin_trues)))
plt.plot(line_x, line_x, color='m', linewidth=2.5)
plt.plot(bin_trues, reg_pred, color='g', linewidth=2.5)
plt.savefig('indel_rate_pred_keras' + str(validation_chrom) + str(complexity_thresh) + '_' + str(windows_per_bin) + '.png')
