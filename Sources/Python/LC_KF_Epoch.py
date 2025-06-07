import numpy as np
from Gravity_ECEF import Gravity_ECEF
from Skew_symmetric import Skew_symmetric

def LC_KF_Epoch(GNSS_r_eb_e, GNSS_v_eb_e, tor_s, est_C_b_e_old, est_v_eb_e_old,
                est_r_eb_e_old, est_IMU_bias_old, P_matrix_old, meas_f_ib_b,
                est_L_b_old, LC_KF_config):
    """
    Implements one cycle of the loosely coupled INS/GNSS Kalman filter plus
    closed-loop correction of all inertial states

    Args:
        GNSS_r_eb_e (np.ndarray): GNSS estimated ECEF user position (m)
        GNSS_v_eb_e (np.ndarray): GNSS estimated ECEF user velocity (m/s)
        tor_s (float): Propagation interval (s)
        est_C_b_e_old (np.ndarray): Prior estimated body to ECEF coordinate transformation matrix
        est_v_eb_e_old (np.ndarray): Prior estimated ECEF user velocity (m/s)
        est_r_eb_e_old (np.ndarray): Prior estimated ECEF user position (m)
        est_IMU_bias_old (np.ndarray): Prior estimated IMU biases (body axes, [accel; gyro])
        P_matrix_old (np.ndarray): Previous Kalman filter error covariance matrix
        meas_f_ib_b (np.ndarray): Measured specific force (m/s^2)
        est_L_b_old (float): Previous latitude solution (rad)
        LC_KF_config (dict): Configuration dictionary containing:
            - gyro_noise_PSD (float): Gyro noise PSD (rad^2/s)
            - accel_noise_PSD (float): Accelerometer noise PSD (m^2 s^-3)
            - accel_bias_PSD (float): Accelerometer bias random walk PSD (m^2 s^-5)
            - gyro_bias_PSD (float): Gyro bias random walk PSD (rad^2 s^-3)
            - pos_meas_SD (float): Position measurement noise SD per axis (m)
            - vel_meas_SD (float): Velocity measurement noise SD per axis (m/s)

    Returns:
        tuple: (est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, P_matrix_new)
            - est_C_b_e_new (np.ndarray): Updated estimated body to ECEF coordinate transformation matrix
            - est_v_eb_e_new (np.ndarray): Updated estimated ECEF user velocity (m/s)
            - est_r_eb_e_new (np.ndarray): Updated estimated ECEF user position (m)
            - est_IMU_bias_new (np.ndarray): Updated estimated IMU biases ([accel; gyro])
            - P_matrix_new (np.ndarray): Updated Kalman filter error covariance matrix
    """
    # Constants
    C = 299792458  # Speed of light in m/s
    OMEGA_IE = 7.292115E-5  # Earth rotation rate in rad/s
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    E = 0.0818191908425  # WGS84 eccentricity

    # Skew symmetric matrix of Earth rate
    Omega_ie = Skew_symmetric(np.array([0, 0, OMEGA_IE]))

    # SYSTEM PROPAGATION PHASE

    # 1. Determine transition matrix using (14.50) (first-order approx)
    Phi_matrix = np.eye(15)
    Phi_matrix[0:3, 0:3] = Phi_matrix[0:3, 0:3] - Omega_ie * tor_s
    Phi_matrix[0:3, 12:15] = est_C_b_e_old * tor_s
    Phi_matrix[3:6, 0:3] = -tor_s * Skew_symmetric(est_C_b_e_old @ meas_f_ib_b)
    Phi_matrix[3:6, 3:6] = Phi_matrix[3:6, 3:6] - 2 * Omega_ie * tor_s
    geocentric_radius = R_0 / np.sqrt(1 - (E * np.sin(est_L_b_old))**2) * \
                        np.sqrt(np.cos(est_L_b_old)**2 + (1 - E**2)**2 * np.sin(est_L_b_old)**2)  # from (2.137)
    Phi_matrix[3:6, 6:9] = -tor_s * 2 * Gravity_ECEF(est_r_eb_e_old) / \
                           geocentric_radius * est_r_eb_e_old.T / np.sqrt(est_r_eb_e_old.T @ est_r_eb_e_old)
    Phi_matrix[3:6, 9:12] = est_C_b_e_old * tor_s
    Phi_matrix[6:9, 3:6] = np.eye(3) * tor_s

    # 2. Determine approximate system noise covariance matrix using (14.82)
    Q_prime_matrix = np.zeros((15, 15))
    Q_prime_matrix[0:3, 0:3] = np.eye(3) * LC_KF_config['gyro_noise_PSD'] * tor_s
    Q_prime_matrix[3:6, 3:6] = np.eye(3) * LC_KF_config['accel_noise_PSD'] * tor_s
    Q_prime_matrix[9:12, 9:12] = np.eye(3) * LC_KF_config['accel_bias_PSD'] * tor_s
    Q_prime_matrix[12:15, 12:15] = np.eye(3) * LC_KF_config['gyro_bias_PSD'] * tor_s

    # 3. Propagate state estimates using (3.14) noting that all states are zero due to closed-loop correction
    x_est_propagated = np.zeros((15,))

    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_matrix_propagated = Phi_matrix @ (P_matrix_old + 0.5 * Q_prime_matrix) @ Phi_matrix.T + 0.5 * Q_prime_matrix

    # MEASUREMENT UPDATE PHASE

    # 5. Set-up measurement matrix using (14.115)
    H_matrix = np.zeros((6, 15))
    H_matrix[0:3, 6:9] = -np.eye(3)
    H_matrix[3:6, 3:6] = -np.eye(3)

    # 6. Set-up measurement noise covariance matrix
    R_matrix = np.zeros((6, 6))
    R_matrix[0:3, 0:3] = np.eye(3) * LC_KF_config['pos_meas_SD']**2
    R_matrix[3:6, 3:6] = np.eye(3) * LC_KF_config['vel_meas_SD']**2

    # 7. Calculate Kalman gain using (3.21)
    K_matrix = P_matrix_propagated @ H_matrix.T @ np.linalg.inv(H_matrix @ P_matrix_propagated @ H_matrix.T + R_matrix)

    # 8. Formulate measurement innovations using (14.102), noting that zero lever arm is assumed
    delta_z = np.zeros((6,))
    delta_z[0:3] = GNSS_r_eb_e.flatten() - est_r_eb_e_old.flatten()
    delta_z[3:6] = GNSS_v_eb_e.flatten() - est_v_eb_e_old.flatten()

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + K_matrix @ delta_z
    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = (np.eye(15) - K_matrix @ H_matrix) @ P_matrix_propagated

    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3])) @ est_C_b_e_old
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[6:9]

    # Update IMU bias estimates
    est_IMU_bias_new = est_IMU_bias_old + x_est_new[9:15]

    return est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, P_matrix_new