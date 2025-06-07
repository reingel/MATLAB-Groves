import numpy as np

def Skew_symmetric(a):
    """
    Skew_symmetric - Calculates skew-symmetric matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        a (numpy.ndarray): 3-element vector, shape (3,)

    Returns:
        numpy.ndarray: 3x3 skew-symmetric matrix, shape (3,3)
    """
    A = np.array([
        [0, -a[2], a[1]],
        [a[2], 0, -a[0]],
        [-a[1], a[0], 0]
    ])
    return A