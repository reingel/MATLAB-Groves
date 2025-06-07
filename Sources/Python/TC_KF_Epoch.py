import numpy as np
from Gravity_ECEF import Gravity_ECEF
from Skew_symmetric import Skew_symmetric

def TC_KF_Epoch(GNSS_measurements, no_meas, tor_s, est_C_b_e_old, est_v_eb_e_old,
                est_r_eb_e_old, est_IMU_bias_old, est_clock_old, P_matrix_old,
                meas_f_ib_b, est_L_b_old, TC_KF_config):
    """
    Implements one cycle of the tightly coupled INS/GNSS extended Kalman filter
    plus closed-loop correction of all inertial states

    Args:
        GNSS_measurements (np.ndarray): GNSS measurement data:
            - Column 0: Pseudo-range measurements (m)
            - Column 1: Pseudo-range rate measurements (m/s)
            - Columns 2-4: Satellite ECEF position (m)
            - Columns 5-7: Satellite ECEF velocity (m/s)
        no_meas (int): Number of satellites for which measurements are supplied
        tor_s (float): Propagation interval (s)
        est_C_b_e_old (np.ndarray): Prior estimated body to ECEF coordinate transformation matrix
        est_v_eb_e_old (np.ndarray): Prior estimated ECEF user velocity (m/s)
        est_r_eb_e_old (np.ndarray): Prior estimated ECEF user position (m)
        est_IMU_bias_old (np.ndarray): Prior estimated IMU biases (body axes, [accel; gyro])
        est_clock_old (np.ndarray): Prior Kalman filter state estimates ([offset; drift])
        P_matrix_old (np.ndarray): Previous Kalman filter error covariance matrix
        meas_f_ib_b (np.ndarray): Measured specific force (m/s^2)
        est_L_b_old (float): Previous latitude solution (rad)
        TC_KF_config (dict): Configuration dictionary containing:
            - gyro_noise_PSD (float): Gyro noise PSD (rad^2/s)
            - accel_noise_PSD (float): Accelerometer noise PSD (m^2 s^-3)
            - accel_bias_PSD (float): Accelerometer bias random walk PSD (m^2 s^-5)
            - gyro_bias_PSD (float): Gyro bias random walk PSD (rad^2 s^-3)
            - clock_freq_PSD (float): Receiver clock frequency-drift PSD (m^2/s^3)
            - clock_phase_PSD (float): Receiver clock phase-drift PSD (m^2/s)
            - pseudo_range_SD (float): Pseudo-range measurement noise SD (m)
            - range_rate_SD (float): Pseudo-range rate measurement noise SD (m/s)

    Returns:
        tuple: (est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, est_clock_new, P_matrix_new)
            - est_C_b_e_new (np.ndarray): Updated estimated body to ECEF coordinate transformation matrix
            - est_v_eb_e_new (np.ndarray): Updated estimated ECEF user velocity (m/s)
            - est_r_eb_e_new (np.ndarray): Updated estimated ECEF user position (m)
            - est_IMU_bias_new (np.ndarray): Updated estimated IMU biases ([accel; gyro])
            - est_clock_new (np.ndarray): Updated Kalman filter state estimates ([offset; drift])
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

    # 1. Determine transition matrix using (14.50) (first-order approximation)
    Phi_matrix = np.eye(17)
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
    Phi_matrix[15, 16] = tor_s

    # 2. Determine approximate system noise covariance matrix using (14.82)
    Q_prime_matrix = np.zeros((17, 17))
    Q_prime_matrix[0:3, 0:3] = np.eye(3) * TC_KF_config['gyro_noise_PSD'] * tor_s
    Q_prime_matrix[3:6, 3:6] = np.eye(3) * TC_KF_config['accel_noise_PSD'] * tor_s
    Q_prime_matrix[9:12, 9:12] = np.eye(3) * TC_KF_config['accel_bias_PSD'] * tor_s
    Q_prime_matrix[12:15, 12:15] = np.eye(3) * TC_KF_config['gyro_bias_PSD'] * tor_s
    Q_prime_matrix[15, 15] = TC_KF_config['clock_phase_PSD'] * tor_s
    Q_prime_matrix[16, 16] = TC_KF_config['clock_freq_PSD'] * tor_s

    # 3. Propagate state estimates using (3.14)
    x_est_propagated = np.zeros((17,))
    x_est_propagated[15] = est_clock_old[0] + est_clock_old[1] * tor_s
    x_est_propagated[16] = est_clock_old[1]

    # 4. Propagate state estimation error covariance matrix using (3.46)
    P_matrix_propagated = Phi_matrix @ (P_matrix_old + 0.5 * Q_prime_matrix) @ Phi_matrix.T + 0.5 * Q_prime_matrix

    # MEASUREMENT UPDATE PHASE

    u_as_e_T = np.zeros((no_meas, 3))
    pred_meas = np.zeros((no_meas, 2))

    # Loop measurements
    for j in range(no_meas):
        # Predict approximate range
        delta_r = GNSS_measurements[j, 2:5] - est_r_eb_e_old
        approx_range = np.sqrt(delta_r.T @ delta_r).item()

        # Calculate frame rotation during signal transit time using (8.36)
        C_e_I = np.array([
            [1, OMEGA_IE * approx_range / C, 0],
            [-OMEGA_IE * approx_range / C, 1, 0],
            [0, 0, 1]
        ])

        # Predict pseudo-range using (9.165)
        delta_r = C_e_I @ GNSS_measurements[j, 2:5] - est_r_eb_e_old
        range_val = np.sqrt(delta_r.T @ delta_r).item()
        pred_meas[j, 0] = range_val + x_est_propagated[15]

        # Predict line of sight
        u_as_e_T[j, :] = (delta_r.T / range_val).flatten()

        # Predict pseudo-range rate using (9.165)
        range_rate = u_as_e_T[j, :] @ (C_e_I @ (GNSS_measurements[j, 5:8] + \
                                        Omega_ie @ GNSS_measurements[j, 2:5]) - \
                                       (est_v_eb_e_old + Omega_ie @ est_r_eb_e_old)).flatten()
        pred_meas[j, 1] = range_rate + x_est_propagated[16]

    # 5. Set-up measurement matrix using (14.126)
    H_matrix = np.zeros((2 * no_meas, 17))
    H_matrix[0:no_meas, 6:9] = u_as_e_T
    H_matrix[0:no_meas, 15] = np.ones(no_meas)
    H_matrix[no_meas:2*no_meas, 3:6] = u_as_e_T
    H_matrix[no_meas:2*no_meas, 16] = np.ones(no_meas)

    # 6. Set-up measurement noise covariance matrix
    R_matrix = np.zeros((2 * no_meas, 2 * no_meas))
    R_matrix[0:no_meas, 0:no_meas] = np.eye(no_meas) * TC_KF_config['pseudo_range_SD']**2
    R_matrix[no_meas:2*no_meas, no_meas:2*no_meas] = np.eye(no_meas) * TC_KF_config['range_rate_SD']**2

    # 7. Calculate Kalman gain using (3.21)
    K_matrix = P_matrix_propagated @ H_matrix.T @ np.linalg.inv(H_matrix @ P_matrix_propagated @ H_matrix.T + R_matrix)

    # 8. Formulate measurement innovations using (14.119)
    delta_z = np.zeros((2 * no_meas,))
    delta_z[0:no_meas] = GNSS_measurements[:, 0] - pred_meas[:, 0]
    delta_z[no_meas:2*no_meas] = GNSS_measurements[:, 1] - pred_meas[:, 1]

    # 9. Update state estimates using (3.24)
    x_est_new = x_est_propagated + K_matrix @ delta_z

    # 10. Update state estimation error covariance matrix using (3.25)
    P_matrix_new = (np.eye(17) - K_matrix @ H_matrix) @ P_matrix_propagated

    # CLOSED-LOOP CORRECTION

    # Correct attitude, velocity, and position using (14.7-9)
    est_C_b_e_new = (np.eye(3) - Skew_symmetric(x_est_new[0:3])) @ est_C_b_e_old
    est_v_eb_e_new = est_v_eb_e_old - x_est_new[3:6]
    est_r_eb_e_new = est_r_eb_e_old - x_est_new[6:9]

    # Update IMU bias and GNSS receiver clock estimates
    est_IMU_bias_new = est_IMU_bias_old + x_est_new[9:15]
    est_clock_new = x_est_new[15:17]

    return est_C_b_e_new, est_v_eb_e_new, est_r_eb_e_new, est_IMU_bias_new, est_clock_new, P_matrix_new