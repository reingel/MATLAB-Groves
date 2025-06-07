import numpy as np

def ECEF_to_ECI(t, r_eb_e, v_eb_e, C_b_e):
    """
    ECEF_to_ECI - Converts position, velocity, and attitude from ECEF- to
    ECI-frame referenced and resolved

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        t (float): time (s)
        r_eb_e (numpy.ndarray): Cartesian position of body frame w.r.t. ECEF frame,
                                resolved along ECEF-frame axes (m), shape (3,)
        v_eb_e (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                along ECEF-frame axes (m/s), shape (3,)
        C_b_e (numpy.ndarray): body-to-ECEF-frame coordinate transformation matrix,
                               shape (3,3)

    Returns:
        tuple: (r_ib_i, v_ib_i, C_b_i)
            - r_ib_i (numpy.ndarray): Cartesian position of body frame w.r.t. ECI frame,
                                      resolved along ECI-frame axes (m), shape (3,)
            - v_ib_i (numpy.ndarray): velocity of body frame w.r.t. ECI frame, resolved
                                      along ECI-frame axes (m/s), shape (3,)
            - C_b_i (numpy.ndarray): body-to-ECI-frame coordinate transformation matrix,
                                     shape (3,3)
    """
    # Parameters
    omega_ie = 7.292115E-5  # Earth rotation rate (rad/s)

    # Calculate ECEF to ECI coordinate transformation matrix using (2.145)
    C_e_i = np.array([
        [np.cos(omega_ie * t), -np.sin(omega_ie * t), 0],
        [np.sin(omega_ie * t),  np.cos(omega_ie * t), 0],
        [0,                     0,                    1]
    ])

    # Transform position using (2.146)
    r_ib_i = C_e_i @ r_eb_e

    # Transform velocity using (2.145)
    v_ib_i = C_e_i @ (v_eb_e + omega_ie * np.array([-r_eb_e[1], r_eb_e[0], 0]))

    # Transform attitude using (2.15)
    C_b_i = C_e_i @ C_b_e

    return r_ib_i, v_ib_i, C_b_i