import numpy as np

def Initialize_TC_P_matrix(TC_KF_config):
    """
    Initializes the tightly coupled INS/GNSS EKF error covariance matrix
    
    Args:
        TC_KF_config (dict): Configuration dictionary containing:
            - init_att_unc (float): Initial attitude uncertainty per axis (rad)
            - init_vel_unc (float): Initial velocity uncertainty per axis (m/s)
            - init_pos_unc (float): Initial position uncertainty per axis (m)
            - init_b_a_unc (float): Initial accelerometer bias uncertainty (m/s^2)
            - init_b_g_unc (float): Initial gyro bias uncertainty (rad/s)
            - init_clock_offset_unc (float): Initial clock offset uncertainty (m)
            - init_clock_drift_unc (float): Initial clock drift uncertainty (m/s)
    
    Returns:
        np.ndarray: State estimation error covariance matrix (17x17)
    """
    # Initialize error covariance matrix
    P_matrix = np.zeros((17, 17))
    P_matrix[0:3, 0:3] = np.eye(3) * TC_KF_config['init_att_unc']**2
    P_matrix[3:6, 3:6] = np.eye(3) * TC_KF_config['init_vel_unc']**2
    P_matrix[6:9, 6:9] = np.eye(3) * TC_KF_config['init_pos_unc']**2
    P_matrix[9:12, 9:12] = np.eye(3) * TC_KF_config['init_b_a_unc']**2
    P_matrix[12:15, 12:15] = np.eye(3) * TC_KF_config['init_b_g_unc']**2
    P_matrix[15, 15] = TC_KF_config['init_clock_offset_unc']**2
    P_matrix[16, 16] = TC_KF_config['init_clock_drift_unc']**2
    
    return P_matrix