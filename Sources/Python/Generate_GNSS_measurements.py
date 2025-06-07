import numpy as np
from Skew_symmetric import Skew_symmetric

def Generate_GNSS_measurements(time, sat_r_es_e, sat_v_es_e, r_ea_e, L_a, lambda_a, v_ea_e, GNSS_biases, GNSS_config):
    """
    Generate_GNSS_measurements - Generates a set of pseudo-range and pseudo-
    range rate measurements for all satellites above the elevation mask angle
    and adds satellite positions and velocities to the datasets.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        time (float): Current simulation time
        sat_r_es_e (numpy.ndarray): ECEF satellite positions (m), shape (no_sat, 3)
        sat_v_es_e (numpy.ndarray): ECEF satellite velocities (m/s), shape (no_sat, 3)
        r_ea_e (numpy.ndarray): ECEF user position (m), shape (3,)
        L_a (float): user latitude (rad)
        lambda_a (float): user longitude (rad)
        v_ea_e (numpy.ndarray): ECEF user velocity (m/s), shape (3,)
        GNSS_biases (numpy.ndarray): Bias-like GNSS range errors (m), shape (no_sat,)
        GNSS_config (dict): Configuration dictionary with keys:
            - no_sat (int): Number of satellites in constellation
            - mask_angle (float): Mask angle (deg)
            - code_track_err_SD (float): Code tracking error SD (m)
            - rate_track_err_SD (float): Range rate tracking error SD (m/s)
            - rx_clock_offset (float): Receiver clock offset at time=0 (m)
            - rx_clock_drift (float): Receiver clock drift at time=0 (m/s)

    Returns:
        tuple: (GNSS_measurements, no_GNSS_meas)
            - GNSS_measurements (numpy.ndarray): GNSS measurement data, shape (no_GNSS_meas, 8):
                - Column 1: Pseudo-range measurements (m)
                - Column 2: Pseudo-range rate measurements (m/s)
                - Columns 3-5: Satellite ECEF position (m)
                - Columns 6-8: Satellite ECEF velocity (m/s)
            - no_GNSS_meas (int): Number of satellites for which measurements are supplied
    """
    # Constants
    c = 299792458  # Speed of light in m/s
    omega_ie = 7.292115E-5  # Earth rotation rate in rad/s

    # Initialize output
    no_GNSS_meas = 0
    GNSS_measurements = np.zeros((GNSS_config['no_sat'], 8))

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = np.cos(L_a)
    sin_lat = np.sin(L_a)
    cos_long = np.cos(lambda_a)
    sin_long = np.sin(lambda_a)
    C_e_n = np.array([
        [-sin_lat * cos_long, -sin_lat * sin_long, cos_lat],
        [-sin_long,           cos_long,            0],
        [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
    ])

    # Skew symmetric matrix of Earth rate
    Omega_ie = Skew_symmetric(np.array([0, 0, omega_ie]))

    # Loop satellites
    for j in np.arange(GNSS_config['no_sat']):
        # Determine ECEF line-of-sight vector using (8.41)
        delta_r = sat_r_es_e[j, :] - r_ea_e
        approx_range = np.sqrt(delta_r @ delta_r)
        u_as_e = delta_r / approx_range

        # Convert line-of-sight vector to NED using (8.39) and determine
        # elevation using (8.57)
        elevation = -np.arcsin(C_e_n[2, :] @ u_as_e)

        # Determine if satellite is above the masking angle
        if elevation >= np.deg2rad(GNSS_config['mask_angle']):
            # Increment number of measurements
            no_GNSS_meas += 1

            # Calculate frame rotation during signal transit time using (8.36)
            C_e_I = np.array([
                [1, omega_ie * approx_range / c, 0],
                [-omega_ie * approx_range / c, 1, 0],
                [0, 0, 1]
            ])

            # Calculate range using (8.35)
            delta_r = C_e_I @ sat_r_es_e[j, :] - r_ea_e
            range = np.sqrt(delta_r @ delta_r)

            # Calculate range rate using (8.44)
            range_rate = u_as_e @ (C_e_I @ (sat_v_es_e[j, :] + Omega_ie @ sat_r_es_e[j, :]) - (v_ea_e + Omega_ie @ r_ea_e))

            # Calculate pseudo-range measurement
            GNSS_measurements[no_GNSS_meas - 1, 0] = range + GNSS_biases[j] + \
                GNSS_config['rx_clock_offset'] + GNSS_config['rx_clock_drift'] * time + \
                GNSS_config['code_track_err_SD'] * np.random.randn()

            # Calculate pseudo-range rate measurement
            GNSS_measurements[no_GNSS_meas - 1, 1] = range_rate + \
                GNSS_config['rx_clock_drift'] + GNSS_config['rate_track_err_SD'] * np.random.randn()

            # Append satellite position and velocity to output data
            GNSS_measurements[no_GNSS_meas - 1, 2:5] = sat_r_es_e[j, :]
            GNSS_measurements[no_GNSS_meas - 1, 5:8] = sat_v_es_e[j, :]

    # Trim the output array to the actual number of measurements
    GNSS_measurements = GNSS_measurements[:no_GNSS_meas, :]

    return GNSS_measurements, no_GNSS_meas