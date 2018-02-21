import logging
import numpy as np
from os import listdir
import pandas as pd
import pickle

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)


def load_recombination(in_path):
    """
    Loads recombination map of the genome from text file ***for a single contig***.
    Expects one text file per contig.

    :param str in_path: Path to recombination map file.
    :return: recombination map for contig
    :rtype: np.ndarray
    """
    logger.info("Loading recombination map from {}".format(in_path))
    r = np.genfromtxt(in_path, dtype = None, usecols = (0,1), skip_header = 1)
    indices = r['f0']  # positions (sparse)
    rates = r['f1']   # recombination rates
    # indices are 1 to maxKnownIndex r[-1][0], +1 for empty 0 index since we assume 1-based indexing
    recombination_map = np.zeros(indices[-1] + 1)
    recombination_map[indices] = rates
    mask = np.ones(len(recombination_map), dtype=bool)
    mask[indices] = False   # now the mask is True only for the indices that are missing from the data
    missing_indices = np.arange(len(recombination_map))[mask]
    recombination_map[missing_indices] = np.interp(missing_indices, indices, rates)
    recombination_map = recombination_map.reshape(len(recombination_map),1)
    logger.info("Recombination map loaded.")
    return recombination_map
