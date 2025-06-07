import numpy as np
from Euler_to_CTM import Euler_to_CTM

def Initialize_NED_attitude(C_b_n, initialization_errors):
    """
    Initializes the attitude solution by adding errors to the truth.
    
    Args:
        C_b_n (np.ndarray): True body-to-NED coordinate transformation matrix
        initialization_errors (dict): Dictionary containing:
            - delta_eul_nb_n (np.ndarray): Attitude error as NED Euler angles (rad)
    
    Returns:
        np.ndarray: Body-to-NED coordinate transformation matrix solution
    """
    # Attitude initialization
    delta_C_b_n = Euler_to_CTM(-initialization_errors['delta_eul_nb_n'])
    est_C_b_n = delta_C_b_n @ C_b_n
    
    return est_C_b_n