"""cs237b.py: Contains helper functions to load genomics data for cs237b class."""

__author__      = "Laurent Francioli"

import logging
import numpy as np
from os import listdir
import pickle

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)

#Not necessary here, but reference sequence was one-hot encoded based on the indices of the bases in this array using LabelBinarizer as follows;
#BASES = ['A','C','G','T']
#from sklearn.preprocessing import LabelBinarizer
#dna_encoder = LabelBinarizer(sparse_output=True).fit(BASES)

def load_bitpacked_reference(in_path):
    """
    Loads one hot encoded bitpacked human genome reference.
    Expects a directory containing one bitpacked numpy array file per contig as well as a pickle containg the intervals with ambiguous bases.
    Returns a dict containing the mapping between contigs and their content as one hot encoded numpy arrays of shape (x,4), where x is the length of the contig.

    :param str in_path: Path to directory containing the contigs
    :return: reference genome
    :rtype: dict of str:np.ndarray
    """
    logger.info("Loading bitpacked reference from {}".format(in_path))
    reference = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            packed = np.load(in_path + "/" + f)
            reference[f[:-4]] = np.unpackbits(packed).reshape(packed.shape[1]*2, 4)

    with open(in_path + "/ambiguous_bases.pickle", "r") as file:
        ambiguous_bases = pickle.load(file)

    for contig in reference:
        if contig not in ambiguous_bases:
            logger.warn("No ambiguous bases indices found for contig {}.".format(contig))

    logger.info("Reference loaded.")
    return reference, ambiguous_bases

def load_coverage(in_path):
    """
    Loads coverage of the genome from numpy arrays.
    Expects a directory containing one numpy array file per contig.


    :param str in_path: Path to directory containing coverage files.
    :return: genome coverage
    :rtype: dict of str:np.ndarray
    """
    logger.info("Loading coverage files from {}".format(in_path))
    coverage = {}
    for f in listdir(in_path):
        if f.endswith(".npy"):
            coverage_values = np.load(in_path + "/" + f)
            coverage[f[:-4]] = coverage_values.reshape(len(coverage_values),1)

    logger.info("Coverage files loaded.")
    return coverage

