import numpy as np

def Initialize_LC_P_matrix(LC_KF_config):
    """
    Initializes the loosely coupled INS/GNSS KF error covariance matrix
    
    Args:
        LC_KF_config (dict): Configuration dictionary containing:
            - init_att_unc (float): Initial attitude uncertainty per axis (rad)
            - init_vel_unc (float): Initial velocity uncertainty per axis (m/s)
            - init_pos_unc (float): Initial position uncertainty per axis (m)
            - init_b_a_unc (float): Initial accelerometer bias uncertainty (m/s^2)
            - init_b_g_unc (float): Initial gyro bias uncertainty (rad/s)
    
    Returns:
        np.ndarray: State estimation error covariance matrix (15x15)
    """
    # Initialize error covariance matrix
    P_matrix = np.zeros((15, 15))
    P_matrix[0:3, 0:3] = np.eye(3) * LC_KF_config['init_att_unc']**2
    P_matrix[3:6, 3:6] = np.eye(3) * LC_KF_config['init_vel_unc']**2
    P_matrix[6:9, 6:9] = np.eye(3) * LC_KF_config['init_pos_unc']**2
    P_matrix[9:12, 9:12] = np.eye(3) * LC_KF_config['init_b_a_unc']**2
    P_matrix[12:15, 12:15] = np.eye(3) * LC_KF_config['init_b_g_unc']**2
    
    return P_matrix