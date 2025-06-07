import numpy as np
from Skew_symmetric import Skew_symmetric

def GNSS_LS_position_velocity(GNSS_measurements, no_GNSS_meas, predicted_r_ea_e, predicted_v_ea_e):
    """
    GNSS_LS_position_velocity - Calculates position, velocity, clock offset,
    and clock drift using unweighted iterated least squares. Separate
    calculations are implemented for position and clock offset and for
    velocity and clock drift

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        GNSS_measurements (numpy.ndarray): GNSS measurement data, shape (no_GNSS_meas, 8):
            - Column 1: Pseudo-range measurements (m)
            - Column 2: Pseudo-range rate measurements (m/s)
            - Columns 3-5: Satellite ECEF position (m)
            - Columns 6-8: Satellite ECEF velocity (m/s)
        no_GNSS_meas (int): Number of satellites for which measurements are supplied
        predicted_r_ea_e (numpy.ndarray): prior predicted ECEF user position (m), shape (3,)
        predicted_v_ea_e (numpy.ndarray): prior predicted ECEF user velocity (m/s), shape (3,)

    Returns:
        tuple: (est_r_ea_e, est_v_ea_e, est_clock)
            - est_r_ea_e (numpy.ndarray): estimated ECEF user position (m), shape (3,)
            - est_v_ea_e (numpy.ndarray): estimated ECEF user velocity (m/s), shape (3,)
            - est_clock (numpy.ndarray): estimated receiver clock offset (m) and drift (m/s), shape (2,)
    """
    # Constants
    c = 299792458  # Speed of light in m/s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad/s

    # POSITION AND CLOCK OFFSET
    # Setup predicted state
    x_pred = np.zeros(4)
    x_pred[:3] = predicted_r_ea_e
    x_pred[3] = 0
    test_convergence = 1
    pred_meas = np.zeros(no_GNSS_meas)
    H_matrix = np.zeros((no_GNSS_meas, 4))

    # Repeat until convergence
    while test_convergence > 0.0001:
        # Loop measurements
        for j in np.arange(no_GNSS_meas):
            # Predict approx range
            delta_r = GNSS_measurements[j, 2:5] - x_pred[:3]
            approx_range = np.sqrt(delta_r @ delta_r)

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.array([
                [1, omega_ie * approx_range / c, 0],
                [-omega_ie * approx_range / c, 1, 0],
                [0, 0, 1]
            ])

            # Predict pseudo-range using (9.143)
            delta_r = C_e_I @ GNSS_measurements[j, 2:5] - x_pred[:3]
            range = np.sqrt(delta_r @ delta_r)
            pred_meas[j] = range + x_pred[3]

            # Predict line of sight and deploy in measurement matrix, (9.144)
            H_matrix[j, :3] = -delta_r / range
            H_matrix[j, 3] = 1

        # Unweighted least-squares solution, (9.35)/(9.141)
        x_est = x_pred + np.linalg.inv(H_matrix[:no_GNSS_meas, :].T @ H_matrix[:no_GNSS_meas, :]) @ \
                H_matrix[:no_GNSS_meas, :].T @ (GNSS_measurements[:no_GNSS_meas, 0] - pred_meas[:no_GNSS_meas])

        # Test convergence
        test_convergence = np.sqrt((x_est - x_pred) @ (x_est - x_pred))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # Set outputs to estimates
    est_r_ea_e = x_est[:3]
    est_clock = np.zeros(2)
    est_clock[0] = x_est[3]

    # VELOCITY AND CLOCK DRIFT
    # Skew symmetric matrix of Earth rate
    Omega_ie = Skew_symmetric(np.array([0, 0, omega_ie]))

    # Setup predicted state
    x_pred = np.zeros(4)
    x_pred[:3] = predicted_v_ea_e
    x_pred[3] = 0
    test_convergence = 1
    pred_meas = np.zeros(no_GNSS_meas)
    H_matrix = np.zeros((no_GNSS_meas, 4))

    # Repeat until convergence
    while test_convergence > 0.0001:
        # Loop measurements
        for j in np.arange(no_GNSS_meas):
            # Predict approx range
            delta_r = GNSS_measurements[j, 2:5] - est_r_ea_e
            approx_range = np.sqrt(delta_r @ delta_r)

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.array([
                [1, omega_ie * approx_range / c, 0],
                [-omega_ie * approx_range / c, 1, 0],
                [0, 0, 1]
            ])

            # Calculate range using (8.35)
            delta_r = C_e_I @ GNSS_measurements[j, 2:5] - est_r_ea_e
            range = np.sqrt(delta_r @ delta_r)

            # Calculate line of sight using (8.41)
            u_as_e = delta_r / range

            # Predict pseudo-range rate using (9.143)
            range_rate = u_as_e @ (C_e_I @ (GNSS_measurements[j, 5:8] + Omega_ie @ GNSS_measurements[j, 2:5]) - \
                                  (x_pred[:3] + Omega_ie @ est_r_ea_e))
            pred_meas[j] = range_rate + x_pred[3]

            # Predict line of sight and deploy in measurement matrix, (9.144)
            H_matrix[j, :3] = -u_as_e
            H_matrix[j, 3] = 1

        # Unweighted least-squares solution, (9.35)/(9.141)
        x_est = x_pred + np.linalg.inv(H_matrix[:no_GNSS_meas, :].T @ H_matrix[:no_GNSS_meas, :]) @ \
                H_matrix[:no_GNSS_meas, :].T @ (GNSS_measurements[:no_GNSS_meas, 1] - pred_meas[:no_GNSS_meas])

        # Test convergence
        test_convergence = np.sqrt((x_est - x_pred) @ (x_est - x_pred))

        # Set predictions to estimates for next iteration
        x_pred = x_est

    # Set outputs to estimates
    est_v_ea_e = x_est[:3]
    est_clock[1] = x_est[3]

    return est_r_ea_e, est_v_ea_e, est_clock