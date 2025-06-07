import numpy as np
from Skew_symmetric import Skew_symmetric

def GNSS_KF_Epoch(GNSS_measurements, no_meas, tor_s, x_est_old, P_matrix_old, GNSS_KF_config):
    """
    GNSS_KF_Epoch - Implements one cycle of the GNSS extended Kalman filter

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 12/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        GNSS_measurements (numpy.ndarray): GNSS measurement data, shape (no_meas, 8):
            - Column 1: Pseudo-range measurements (m)
            - Column 2: Pseudo-range rate measurements (m/s)
            - Columns 3-5: Satellite ECEF position (m)
            - Columns 6-8: Satellite ECEF velocity (m/s)
        no_meas (int): Number of satellites for which measurements are supplied
        tor_s (float): Propagation interval (s)
        x_est_old (numpy.ndarray): Previous Kalman filter state estimates, shape (8,)
            - Rows 1-3: estimated ECEF user position (m)
            - Rows 4-6: estimated ECEF user velocity (m/s)
            - Row 7: estimated receiver clock offset (m)
            - Row 8: estimated receiver clock drift (m/s)
        P_matrix_old (numpy.ndarray): Previous Kalman filter error covariance matrix, shape (8, 8)
        GNSS_KF_config (dict): Configuration dictionary with keys:
            - accel_PSD (float): Acceleration PSD per axis (m^2/s^3)
            - clock_freq_PSD (float): Receiver clock frequency-drift PSD (m^2/s^3)
            - clock_phase_PSD (float): Receiver clock phase-drift PSD (m^2/s)
            - pseudo_range_SD (float): Pseudo-range measurement noise SD (m)
            - range_rate_SD (float): Pseudo-range rate measurement noise SD (m/s)

    Returns:
        tuple: (x_est_new, P_matrix_new)
            - x_est_new (numpy.ndarray): Updated Kalman filter state estimates, shape (8,)
            - P_matrix_new (numpy.ndarray): Updated Kalman filter error covariance matrix, shape (8, 8)
    """
    # Constants
    c = 299792458  # Speed of light in m/s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad/s

    # SYSTEM PROPAGATION PHASE

    # 1. Determine transition matrix using (9.147) and (9.150)
    Phi_matrix = np.eye(8)
    Phi_matrix[0, 3] = tor_s
    Phi_matrix[1, 4] = tor_s
    Phi_matrix[2, 5] = tor_s
    Phi_matrix[6, 7] = tor_s

    # 2. Determine system noise covariance matrix using (9.152)
    Q_matrix = np.zeros((8, 8))
    Q_matrix[0:3, 0:3] = np.eye(3) * GNSS_KF_config['accel_PSD'] * tor_s**3 / 3
    Q_matrix[0:3, 3:6] = np.eye(3) * GNSS_KF_config['accel_PSD'] * tor_s**2 / 2
    Q_matrix[3:6, 0:3] = np.eye(3) * GNSS_KF_config['accel_PSD'] * tor_s**2 / 2
    Q_matrix[3:6, 3:6] = np.eye(3) * GNSS_KF_config['accel_PSD'] * tor_s
    Q_matrix[6, 6] = (GNSS_KF_config['clock_freq_PSD'] * tor_s**3 / 3) + GNSS_KF_config['clock_phase_PSD'] * tor_s
    Q_matrix[6, 7] = GNSS_KF_config['clock_freq_PSD'] * tor_s**2 / 2
    Q_matrix[7, 6] = GNSS_KF_config['clock_freq_PSD'] * tor_s**2 / 2
    Q_matrix[7, 7] = GNSS_KF_config['clock_freq_PSD'] * tor_s

    # 3. Propagate state estimates using (3.14)
    x_est_propagated = Phi_matrix @ x_est_old

    # 4. Propagate state estimation error covariance matrix using (3.15)
    P_matrix_propagated = Phi_matrix @ P_matrix_old @ Phi_matrix.T + Q_matrix

    # MEASUREMENT UPDATE PHASE

    # Skew symmetric matrix of Earth rate
    Omega_ie = Skew_symmetric(np.array([0, 0, omega_ie]))

    u_as_e_T = np.zeros((no_meas, 3))
    pred_meas = np.zeros((no_meas, 2))

    # Loop measurements
    for j in range(no_meas):
        # Predict approx range
        delta_r = GNSS_measurements[j, 2:5] - x_est_propagated[:3]
        approx_range = np.sqrt(delta_r @ delta_r)

        # Calculate frame rotation during signal transit time using (8.36)
        C_e_I = np.array([
            [1, omega_ie * approx_range / c, 0],
            [-omega_ie * approx_range / c, 1, 0],
            [0, 0, 1]
        ])

        # Predict pseudo-range using (9.165)
        delta_r = C_e_I @ GNSS_measurements[j, 2:5] - x_est_propagated[:3]
        range_ = np.sqrt(delta_r @ delta_r)
        pred_meas[j, 0] = range_ + x_est_propagated[6]

        # Predict line of sight
        u_as_e_T[j, :] = delta_r / range_

        # Predict pseudo-range rate using (9.165)
        range_rate = u_as_e_T[j, :] @ (C_e_I @ (GNSS_measurements[j, 5:8] +
            Omega_ie @ GNSS_measurements[j, 2:5]) - (x_est_propagated[3:6] +
            Omega_ie @ x_est_propagated[:3]))
        pred_meas[j, 1] = range_rate + x_est_propagated[7]

    # 5. Set-up measurement matrix using (9.163)
    H_matrix = np.zeros((2 * no_meas, 8))
    H_matrix[:no_meas, :3] = -u_as_e_T
    H_matrix[:no_meas, 6] = 1
    H_matrix[no_meas:, 3:6] = -u_as_e_T
    H_matrix[no_meas:, 7] = 1

    # 6. Set-up measurement noise covariance matrix
    R_matrix = np.zeros((2 * no_meas, 2 * no_meas))
    R_matrix[:no_meas, :no_meas] = np.eye(no_meas) * GNSS_KF_config['pseudo_range_SD']**2
    R_matrix[no_meas:, no_meas:] = np.eye(no_meas) * GNSS_KF_config['range_rate_SD']**2

    # 7. Calculate Kalman gain using (3.21)
    K_matrix = P_matrix_propagated @ H_matrix.T @ np.linalg.inv(
        H_matrix @ P_matrix_propagated @ H_matrix.T + R_matrix)

    # 8. Formulate measurement innovations using (3.88)
    delta_z = np.zeros(2 * no_meas)
    delta_z[:no_meas] = GNSS_measurements[:, 0] - pred_meas[:, 0]
    delta_z[no_meas:] = GNSS_measurements[:, 1] - pred_meas[:, 1]

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + K_matrix @ delta_z

    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = (np.eye(8) - K_matrix @ H_matrix) @ P_matrix_propagated

    return x_est_new, P_matrix_new