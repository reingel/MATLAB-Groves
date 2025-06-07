import numpy as np

def ECEF_to_NED(r_eb_e, v_eb_e, C_b_e):
    """
    ECEF_to_NED - Converts Cartesian to curvilinear position, velocity
    resolving axes from ECEF to NED and attitude from ECEF- to NED-referenced

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        r_eb_e (numpy.ndarray): Cartesian position of body frame w.r.t. ECEF frame,
                                resolved along ECEF-frame axes (m), shape (3,)
        v_eb_e (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                along ECEF-frame axes (m/s), shape (3,)
        C_b_e (numpy.ndarray): body-to-ECEF-frame coordinate transformation matrix,
                               shape (3,3)

    Returns:
        tuple: (L_b, lambda_b, h_b, v_eb_n, C_b_n)
            - L_b (float): latitude (rad)
            - lambda_b (float): longitude (rad)
            - h_b (float): height (m)
            - v_eb_n (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                      along north, east, and down (m/s), shape (3,)
            - C_b_n (numpy.ndarray): body-to-NED coordinate transformation matrix,
                                     shape (3,3)
    """
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity

    # Convert position using Borkowski closed-form exact solution
    # From (2.113)
    lambda_b = np.arctan2(r_eb_e[1], r_eb_e[0])

    # From (C.29) and (C.30)
    k1 = np.sqrt(1 - e**2) * abs(r_eb_e[2])
    k2 = e**2 * R_0
    beta = np.sqrt(r_eb_e[0]**2 + r_eb_e[1]**2)
    E = (k1 - k2) / beta
    F = (k1 + k2) / beta

    # From (C.31)
    P = 4/3 * (E * F + 1)

    # From (C.32)
    Q = 2 * (E**2 - F**2)

    # From (C.33)
    D = P**3 + Q**2

    # From (C.34)
    V = (np.sqrt(D) - Q)**(1/3) - (np.sqrt(D) + Q)**(1/3)

    # From (C.35)
    G = 0.5 * (np.sqrt(E**2 + V) + E)

    # From (C.36)
    T = np.sqrt(G**2 + (F - V * G) / (2 * G - E)) - G

    # From (C.37)
    L_b = np.sign(r_eb_e[2]) * np.arctan((1 - T**2) / (2 * T * np.sqrt(1 - e**2)))

    # From (C.38)
    h_b = (beta - R_0 * T) * np.cos(L_b) + \
          (r_eb_e[2] - np.sign(r_eb_e[2]) * R_0 * np.sqrt(1 - e**2)) * np.sin(L_b)

    # Calculate ECEF to NED coordinate transformation matrix using (2.150)
    cos_lat = np.cos(L_b)
    sin_lat = np.sin(L_b)
    cos_long = np.cos(lambda_b)
    sin_long = np.sin(lambda_b)
    C_e_n = np.array([
        [-sin_lat * cos_long, -sin_lat * sin_long, cos_lat],
        [-sin_long,           cos_long,            0],
        [-cos_lat * cos_long, -cos_lat * sin_long, -sin_lat]
    ])

    # Transform velocity using (2.73)
    v_eb_n = C_e_n @ v_eb_e

    # Transform attitude using (2.15)
    C_b_n = C_e_n @ C_b_e

    return L_b, lambda_b, h_b, v_eb_n, C_b_n