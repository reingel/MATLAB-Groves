import numpy as np

def ECI_to_ECEF(t, r_ib_i, v_ib_i, C_b_i):
    """
    ECI_to_ECEF - Converts position, velocity, and attitude from ECI- to
    ECEF-frame referenced and resolved

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        t (float): time (s)
        r_ib_i (numpy.ndarray): Cartesian position of body frame w.r.t. ECI frame,
                                resolved along ECI-frame axes (m), shape (3,)
        v_ib_i (numpy.ndarray): velocity of body frame w.r.t. ECI frame, resolved
                                along ECI-frame axes (m/s), shape (3,)
        C_b_i (numpy.ndarray): body-to-ECI-frame coordinate transformation matrix,
                               shape (3,3)

    Returns:
        tuple: (r_eb_e, v_eb_e, C_b_e)
            - r_eb_e (numpy.ndarray): Cartesian position of body frame w.r.t. ECEF frame,
                                      resolved along ECEF-frame axes (m), shape (3,)
            - v_eb_e (numpy.ndarray): velocity of body frame w.r.t. ECEF frame, resolved
                                      along ECEF-frame axes (m/s), shape (3,)
            - C_b_e (numpy.ndarray): body-to-ECEF-frame coordinate transformation matrix,
                                     shape (3,3)
    """
    # Parameters
    omega_ie = 7.292115E-5  # Earth rotation rate (rad/s)

    # Calculate ECI to ECEF coordinate transformation matrix using (2.145)
    C_i_e = np.array([
        [np.cos(omega_ie * t), np.sin(omega_ie * t), 0],
        [-np.sin(omega_ie * t), np.cos(omega_ie * t), 0],
        [0,                     0,                    1]
    ])

    # Transform position using (2.146)
    r_eb_e = C_i_e @ r_ib_i

    # Transform velocity using (2.145)
    v_eb_e = C_i_e @ (v_ib_i - omega_ie * np.array([-r_ib_i[1], r_ib_i[0], 0]))

    # Transform attitude using (2.15)
    C_b_e = C_i_e @ C_b_i

    return r_eb_e, v_eb_e, C_b_e