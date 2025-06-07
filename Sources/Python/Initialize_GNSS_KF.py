import numpy as np

def Initialize_GNSS_KF(est_r_ea_e, est_v_ea_e, est_clock, GNSS_KF_config):
    """
    Initialize_GNSS_KF - Initializes the GNSS EKF state estimates and error
    covariance matrix

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        est_r_ea_e (numpy.ndarray): Estimated ECEF user position (m), shape (3,)
        est_v_ea_e (numpy.ndarray): Estimated ECEF user velocity (m/s), shape (3,)
        est_clock (numpy.ndarray): Estimated receiver clock offset (m) and drift (m/s), shape (2,)
        GNSS_KF_config (dict): Configuration dictionary with keys:
            - init_pos_unc (float): Initial position uncertainty per axis (m)
            - init_vel_unc (float): Initial velocity uncertainty per axis (m/s)
            - init_clock_offset_unc (float): Initial clock offset uncertainty per axis (m)
            - init_clock_drift_unc (float): Initial clock drift uncertainty per axis (m/s)

    Returns:
        tuple: (x_est, P_matrix)
            - x_est (numpy.ndarray): Kalman filter estimates, shape (8,)
                - Rows 1-3: estimated ECEF user position (m)
                - Rows 4-6: estimated ECEF user velocity (m/s)
                - Row 7: estimated receiver clock offset (m)
                - Row 8: estimated receiver clock drift (m/s)
            - P_matrix (numpy.ndarray): State estimation error covariance matrix, shape (8, 8)
    """
    # Initialize state estimates
    x_est = np.zeros(8)
    x_est[:3] = est_r_ea_e
    x_est[3:6] = est_v_ea_e
    x_est[6:8] = est_clock

    # Initialize error covariance matrix
    P_matrix = np.zeros((8, 8))
    P_matrix[0, 0] = GNSS_KF_config['init_pos_unc']**2
    P_matrix[1, 1] = GNSS_KF_config['init_pos_unc']**2
    P_matrix[2, 2] = GNSS_KF_config['init_pos_unc']**2
    P_matrix[3, 3] = GNSS_KF_config['init_vel_unc']**2
    P_matrix[4, 4] = GNSS_KF_config['init_vel_unc']**2
    P_matrix[5, 5] = GNSS_KF_config['init_vel_unc']**2
    P_matrix[6, 6] = GNSS_KF_config['init_clock_offset_unc']**2
    P_matrix[7, 7] = GNSS_KF_config['init_clock_drift_unc']**2

    return x_est, P_matrix