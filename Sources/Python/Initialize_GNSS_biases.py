import numpy as np

def Initialize_GNSS_biases(sat_r_es_e, r_ea_e, L_a, lambda_a, GNSS_config):
    """
    Initialize_GNSS_biases - Initializes the GNSS range errors due to signal
    in space, ionosphere, and troposphere errors based on the elevation angles.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        sat_r_es_e (numpy.ndarray): ECEF satellite positions (m), shape (no_sat, 3)
        r_ea_e (numpy.ndarray): ECEF user position (m), shape (3,)
        L_a (float): user latitude (rad)
        lambda_a (float): user longitude (rad)
        GNSS_config (dict): Configuration dictionary with keys:
            - no_sat (int): Number of satellites in constellation
            - mask_angle (float): Mask angle (deg)
            - SIS_err_SD (float): Signal in space error SD (m)
            - zenith_iono_err_SD (float): Zenith ionosphere error SD (m)
            - zenith_trop_err_SD (float): Zenith troposphere error SD (m)

    Returns:
        numpy.ndarray: Bias-like GNSS range errors, shape (no_sat,)
    """
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

    # Initialize biases
    GNSS_biases = np.zeros(GNSS_config['no_sat'])

    # Loop satellites
    for j in range(GNSS_config['no_sat']):
        # Determine ECEF line-of-sight vector using (8.41)
        delta_r = sat_r_es_e[j, :] - r_ea_e
        u_as_e = delta_r / np.sqrt(delta_r @ delta_r)

        # Convert line-of-sight vector to NED using (8.39) and determine
        # elevation using (8.57)
        elevation = -np.arcsin(C_e_n[2, :] @ u_as_e)

        # Limit the minimum elevation angle to the masking angle
        elevation = max(elevation, np.deg2rad(GNSS_config['mask_angle']))

        # Calculate ionosphere and troposphere error SDs using (9.79) and (9.80)
        iono_SD = GNSS_config['zenith_iono_err_SD'] / np.sqrt(1 - 0.899 * np.cos(elevation)**2)
        trop_SD = GNSS_config['zenith_trop_err_SD'] / np.sqrt(1 - 0.998 * np.cos(elevation)**2)

        # Determine range bias
        GNSS_biases[j] = GNSS_config['SIS_err_SD'] * np.random.randn() + \
                         iono_SD * np.random.randn() + trop_SD * np.random.randn()

    return GNSS_biases