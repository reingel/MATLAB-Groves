import numpy as np
from Radii_of_curvature import Radii_of_curvature
from Euler_to_CTM import Euler_to_CTM

def Initialize_NED(L_b, lambda_b, h_b, v_eb_n, C_b_n, initialization_errors):
    """
    Initializes the curvilinear position, velocity, and attitude solution by adding errors to the truth.
    
    Args:
        L_b (float): True latitude (rad)
        lambda_b (float): True longitude (rad)
        h_b (float): True height (m)
        v_eb_n (np.ndarray): True velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s)
        C_b_n (np.ndarray): True body-to-NED coordinate transformation matrix
        initialization_errors (dict): Dictionary containing:
            - delta_r_eb_n (np.ndarray): Position error resolved along NED (m)
            - delta_v_eb_n (np.ndarray): Velocity error resolved along NED (m/s)
            - delta_eul_nb_n (np.ndarray): Attitude error as NED Euler angles (rad)
    
    Returns:
        tuple: (est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n)
            - est_L_b (float): Latitude solution (rad)
            - est_lambda_b (float): Longitude solution (rad)
            - est_h_b (float): Height solution (m)
            - est_v_eb_n (np.ndarray): Velocity solution (m/s)
            - est_C_b_n (np.ndarray): Body-to-NED coordinate transformation matrix solution
    """
    # Position initialization
    R_N, R_E = Radii_of_curvature(L_b)
    est_L_b = L_b + initialization_errors['delta_r_eb_n'][0] / (R_N + h_b)
    est_lambda_b = lambda_b + initialization_errors['delta_r_eb_n'][1] / ((R_E + h_b) * np.cos(L_b))
    est_h_b = h_b - initialization_errors['delta_r_eb_n'][2]
    
    # Velocity initialization
    est_v_eb_n = v_eb_n + initialization_errors['delta_v_eb_n']
    
    # Attitude initialization
    delta_C_b_n = Euler_to_CTM(-initialization_errors['delta_eul_nb_n'])
    est_C_b_n = delta_C_b_n @ C_b_n
    
    return est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n