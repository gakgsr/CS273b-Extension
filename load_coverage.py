import logging
import numpy as np
from os import listdir
import pandas as pd
import pickle

logging.basicConfig(format="%(levelname)s (%(name)s %(lineno)s): %(message)s")
logger = logging.getLogger("indel")
logger.setLevel(logging.INFO)


def load_coverage(in_path):
    """
    Loads coverage of the genome from numpy arrays ***for a single contig***.
    Expects one numpy array file per contig.

    :param str in_path: Path to coverage file.
    :return: genome coverage for contig
    :rtype: np.ndarray
    """
    logger.info("Loading coverage files from {}".format(in_path))
    coverage_values = np.load(in_path)
    coverage_values = coverage_values.reshape(len(coverage_values),1)
    logger.info("Coverage files loaded.")
    return coverage_values
