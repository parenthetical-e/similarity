""" A set of vectorized functions for measuring similarity. """
import numpy as np
from similarity.distance import l2


def prototype2d(x1, x2, u1, u2, p):
    """ Return the 2d similarity between a lists of examplars 
    (<x1> and <x2>) and their means (<u1> and <u2>)
    
    Similarity (s) is measured by:
        d = sqrt((x1 - u1)^2) + sqrt((x2 - u2)^2)
        s = exp(-d^p).
            if p = 1, -> exponential metric
            if p = 2, -> gaussian metric

    For discussion of both types, see 

    Nofosky (2000), Relations between Exemplar-Similarity and Likelihood 
    Models of Classification, Journal of mathematical psychology 34, 39.

    Jakel te al (2008) Generalization and similarity in exemplar models of 
    categorization: Insights from machine learning, Psychonomic Bulletin & 
    Review 15 (2), 256-271. """

    x1 = np.array(x1)
    x2 = np.array(x2)
    u1 = np.array(u1)
    u2 = np.array(u2)
    
    if (x1.shape != u1.shape) or (u1.ndim != 0):
        raise ValueError(
            "x1 and u1 must be the same shape or u1 must be a scalar")

    if (x2.shape != u2.shape) or (u2.ndim != 0):
        raise ValueError(
            "x2 and u2 must be the same shape or u2 must be a scalar")
    
    distances = l2(x1 - u1, x2 - u2)
    similarity = np.exp(-distances ** p)

    return similarity, distances


