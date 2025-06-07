import numpy as np

def Euler_to_CTM(eul):
    """
    Euler_to_CTM - Converts a set of Euler angles to the corresponding
    coordinate transformation matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        eul (numpy.ndarray): Euler angles describing rotation from beta to alpha
                            in the order roll, pitch, yaw (rad), shape (3,)

    Returns:
        numpy.ndarray: coordinate transformation matrix describing transformation
                       from beta to alpha, shape (3,3)
    """
    # Precalculate sines and cosines of the Euler angles
    sin_phi = np.sin(eul[0])
    cos_phi = np.cos(eul[0])
    sin_theta = np.sin(eul[1])
    cos_theta = np.cos(eul[1])
    sin_psi = np.sin(eul[2])
    cos_psi = np.cos(eul[2])

    # Calculate coordinate transformation matrix using (2.22)
    C = np.zeros((3, 3))
    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta

    return C