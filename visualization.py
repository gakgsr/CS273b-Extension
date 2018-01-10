import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
import pdb
def attention():
	a = genfromtxt('a.csv', delimiter=',')
	data = a
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(data.shape[1]) + 0.5, minor=False)
	ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
	ax.invert_yaxis()
	heatmap = ax.pcolor(data, cmap='Blues')

	cbar = plt.colorbar(heatmap)
	# plt.imshow(np.transpose(a[0:186,0:13]), cmap='Blues', interpolation='nearest')
	#row_labels = ['In', 'the', 'U.S.', 'News', '&', 'World', 'Report', "'s", '"', 'Americas', 'Best', 'Colleges', '"', '2016', 'issue', ',', 'KUs', 'School', 'of', 'Engineering', 'was', 'ranked', 'tied', 'for', '90th', 'among', 'national', 'universities', '.']
	#column_labels = ["Against", "what", "other", "kinds", "of", "institutions", "was","KU", "'s", "engineering", "school", "compared","?"]
	#ax.set_xticklabels(column_labels, minor=False)
	#ax.set_yticklabels(row_labels, minor=False)
	plt.show()

def predictions():
	b1 = genfromtxt('b1.csv', delimiter=',')
	b1 = b1[0:100]
	b1 = np.reshape(b1, (1, 100))
	b2 = genfromtxt('b2.csv', delimiter=',')
	b2 = b2[0:100]
	b2 = np.reshape(b2, (1, 100))
	fig, ax = plt.subplots()
	ax.set_xticks(np.arange(100), minor=False)
	ax.set_yticks(np.array([0]), minor=False)
	# ax.set_yticks(np.arange(data.shape[0]) + 0.5, minor=False)
	heatmap = ax.pcolor(b1, cmap='Blues')

	cbar = plt.colorbar(heatmap)
	row_labels = ['Nintendo', 'was', 'not', 'as', 'restrictive', 'as', 'Sega', ',', 'which', 'did', 'not', 'permit', 'third-party', 'publishing', 'until', 'Mediagenic', 'in', 'late', 'summer', '1988', '.', 'Nintendo', "'s", 'intention', ',', 'however', ',', 'was', 'to', 'reserve', 'a', 'large', 'part', 'of', 'NES', 'game', 'revenue', 'for', 'itself', '.', 'Nintendo', 'required', 'that', 'they', 'be', 'the', 'sole', 'manufacturer', 'of', 'all', 'cartridges', ',', 'and', 'that', 'the', 'publisher', 'had', 'to', 'pay', 'in', 'full', 'before', 'the', 'cartridges', 'for', 'that', 'game', 'be', 'produced', '.', 'Cartridges', 'could', 'not', 'be', 'returned', 'to', 'Nintendo', ',', 'so', 'publishers', 'assumed', 'all', 'the', 'risk', '.', 'As', 'a', 'result', ',', 'some', 'publishers', 'lost', 'more', 'money', 'due', 'to', 'distress', 'sales', 'of', 'remaining', 'inventory', 'at', 'the', 'end', 'of', 'the', 'NES', 'era', 'than', 'they', 'ever', 'earned', 'in', 'profits', 'from', 'sales', 'of', 'the', 'games', '.', 'Because', 'Nintendo', 'controlled', 'the', 'production', 'of', 'all', 'cartridges', ',', 'it', 'was', 'able', 'to', 'enforce', 'strict', 'rules', 'on', 'its', 'third-party', 'developers', ',', 'which', 'were', 'required', 'to', 'sign', 'a', 'contract', 'by', 'Nintendo', 'that', 'would', 'obligate', 'these', 'parties', 'to', 'develop', 'exclusively', 'for', 'the', 'system', ',', 'order', 'at', 'least', '10,000', 'cartridges', ',', 'and', 'only', 'make', 'five', 'games', 'per', 'year', '.', 'A', '1988', 'shortage', 'of', 'DRAM', 'and', 'ROM', 'chips', 'also', 'reportedly', 'caused', 'Nintendo', 'to', 'only', 'permit', '25', '%', 'of', 'publishers', "'", 'requests', 'for', 'cartridges', '.', 'This', 'was', 'an', 'average', 'figure', ',', 'with', 'some', 'publishers', 'receiving', 'much', 'higher', 'amounts', 'and', 'others', 'almost', 'none', '.', 'GameSpy', 'noted', 'that', 'Nintendo', "'s", '"', 'iron-clad', 'terms', '"', 'made', 'the', 'company', 'many', 'enemies', 'during', 'the', '1980s', '.', 'Some', 'developers', 'tried', 'to', 'circumvent', 'the', 'five', 'game', 'limit', 'by', 'creating', 'additional', 'company', 'brands', 'like', 'Konami', "'s", 'Ultra', 'Games', 'label', ';', 'others', 'tried', 'circumventing', 'the', '10NES', 'chip', '.']
	#row_labels = ['The', 'duplication', 'and', 'transmission', 'of', 'genetic', 'material', 'from', 'one', 'generation', 'of', 'cells', 'to', 'the', 'next', 'is', 'the', 'basis', 'for', 'molecular', 'inheritance', ',', 'and', 'the', 'link', 'between', 'the', 'classical', 'and', 'molecular', 'pictures', 'of', 'genes', '.', 'Organisms', 'inherit', 'the', 'characteristics', 'of', 'their', 'parents', 'because', 'the', 'cells', 'of', 'the', 'offspring', 'contain', 'copies', 'of', 'the', 'genes', 'in', 'their', 'parents', "'", 'cells', '.', 'In', 'asexually', 'reproducing', 'organisms', ',', 'the', 'offspring', 'will', 'be', 'a', 'genetic', 'copy', 'or', 'clone', 'of', 'the', 'parent', 'organism', '.', 'In', 'sexually', 'reproducing', 'organisms', ',', 'a', 'specialized', 'form', 'of', 'cell', 'division', 'called', 'meiosis', 'produces', 'cells', 'called', 'gametes', 'or', 'germ', 'cells', 'that', 'are', 'haploid', ',', 'or', 'contain', 'only', 'one', 'copy', 'of', 'each', 'gene', '.', ':20.2', 'The', 'gametes', 'produced', 'by', 'females', 'are', 'called', 'eggs', 'or', 'ova', ',', 'and', 'those', 'produced', 'by', 'males', 'are', 'called', 'sperm', '.', 'Two', 'gametes', 'fuse', 'to', 'form', 'a', 'diploid', 'fertilized', 'egg', ',', 'a', 'single', 'cell', 'that', 'has', 'two', 'sets', 'of', 'genes', ',', 'with', 'one', 'copy', 'of', 'each', 'gene', 'from', 'the', 'mother', 'and', 'one', 'from', 'the', 'father', '.', ':20']
	ax.set_xticklabels(row_labels, minor=False, rotation='vertical')
	plt.setp(ax.get_xticklabels(), rotation='vertical', fontsize=8)
	plt.show()

attention()
