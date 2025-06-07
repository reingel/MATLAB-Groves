import numpy as np

def pv_NED_to_ECEF(L_b, lambda_b, h_b, v_eb_n):
    """
    pv_NED_to_ECEF - Converts curvilinear to Cartesian position and velocity
    resolving axes from NED to ECEF

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        L_b (float): latitude (rad)
        lambda_b (float): longitude (rad)
        h_b (float): height (m)
        v_eb_n (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                along north, east, and down (m/s), shape (3,)

    Returns:
        tuple: (r_eb_e, v_eb_e)
            - r_eb_e (numpy.ndarray): Cartesian position of body frame w.r.t. ECEF frame,
                                      resolved along ECEF-frame axes (m), shape (3,)
            - v_eb_e (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                      along ECEF-frame axes (m/s), shape (3,)
    """
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # Calculate transverse radius of curvature using (2.105)
    R_E = R_0 / np.sqrt(1 - (e * np.sin(L_b))**2)

    # Convert position using (2.112)
    cos_lat = np.cos(L_b)
    sin_lat = np.sin(L_b)
    cos_long = np.cos(lambda_b)
    sin_long = np.sin(lambda_b)
    r_eb_e = np.array([
        (R_E + h_b) * cos_lat * cos_long,
        (R_E + h_b) * cos_lat * sin_long,
        ((1 - e**2) * R_E + h_b) * sin_lat
    ])

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    C_e_n = np.array([
        [-sin_lat * cos_long, -sin_lat * sin_long, cos_lat],
        [-sin_long,           cos_long,            0],
        [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
    ])

    # Transform velocity using (2.73)
    v_eb_e = C_e_n.T @ v_eb_n

    return r_eb_e, v_eb_e