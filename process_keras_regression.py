import numpy as np
from sequence_analysis import sequenceLogos, sequence_2_mer_generate, plot_seq_2_mer_freq

data_dir = '/datadrive/project_data/'

validation_chrom = 8
x_test = np.load(data_dir + 'RegrKerasTestSeq' + str(validation_chrom) + '.npy')
y_test = np.load(data_dir + 'RegrKerasTestLab' + str(validation_chrom) + '.npy')
y_pred = np.load(data_dir + 'RegrKerasTestLabPred' + str(validation_chrom) + '.npy')

# Average number of indels in the 401 sized window as threshold
threshold = 4

# Create 4 classes of predictions
true_labels = np.arange(len(y_test))
true_pos = true_labels[np.logical_and(y_test >= threshold, y_pred >= threshold)]
true_neg = true_labels[np.logical_and(y_test < threshold, y_pred < threshold)]
false_pos = true_labels[np.logical_and(y_test < threshold, y_pred >= threshold)]
false_neg = true_labels[np.logical_and(y_test >= threshold, y_pred < threshold)]

sequenceLogos(x_test[true_pos], 'KerasTruePos')
sequenceLogos(x_test[true_neg], 'KerasTrueNeg')
sequenceLogos(x_test[false_pos], 'KerasFalsePos')
sequenceLogos(x_test[false_neg], 'KerasFalseNeg')
'''
freq_val_true_pos, _ = sequence_2_mer_generate(x_test[true_pos])
freq_val_true_neg, _ = sequence_2_mer_generate(x_test[true_neg])
freq_val_false_pos, _ = sequence_2_mer_generate(x_test[false_pos])
freq_val_false_neg, _ = sequence_2_mer_generate(x_test[false_neg])
plot_seq_2_mer_freq(freq_val_true_pos, 'KerasTruePos')
plot_seq_2_mer_freq(freq_val_true_neg, 'KerasTrueNeg')
plot_seq_2_mer_freq(freq_val_false_pos, 'KerasFalsePos')
plot_seq_2_mer_freq(freq_val_false_neg, 'KerasFalseNeg')
'''
