""" Distance measures. """
import numpy as np


def l2(x1, x2, axis=0):
    """
    Returns the 2d euclidian distance (L2) between x1 and x2.
    """

    x1 = np.array(x1)
    x2 = np.array(x2)
    distances = np.sqrt(x1 ** 2 + x2 ** 2)

    return distances
