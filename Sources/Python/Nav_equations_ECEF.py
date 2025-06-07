import numpy as np
from Gravity_ECEF import Gravity_ECEF
from Skew_symmetric import Skew_symmetric

def Nav_equations_ECEF(tor_i, old_r_eb_e, old_v_eb_e, old_C_b_e, f_ib_b, omega_ib_b):
    """
    Runs precision ECEF-frame inertial navigation equations

    Args:
        tor_i (float): Time interval between epochs (s)
        old_r_eb_e (np.ndarray): Previous Cartesian position of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m)
        old_v_eb_e (np.ndarray): Previous velocity of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m/s)
        old_C_b_e (np.ndarray): Previous body-to-ECEF-frame coordinate transformation matrix
        f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECEF frame, resolved along body-frame axes, averaged over time interval (m/s^2)
        omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECEF frame, resolved about body-frame axes, averaged over time interval (rad/s)

    Returns:
        tuple: (r_eb_e, v_eb_e, C_b_e)
            - r_eb_e (np.ndarray): Cartesian position of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m)
            - v_eb_e (np.ndarray): Velocity of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m/s)
            - C_b_e (np.ndarray): Body-to-ECEF-frame coordinate transformation matrix
    """
    # Parameters
    OMEGA_IE = 7.292115E-5  # Earth rotation rate (rad/s)

    # ATTITUDE UPDATE
    # From (2.145) determine the Earth rotation over the update interval
    alpha_ie = OMEGA_IE * tor_i
    C_Earth = np.array([
        [np.cos(alpha_ie), np.sin(alpha_ie), 0],
        [-np.sin(alpha_ie), np.cos(alpha_ie), 0],
        [0, 0, 1]
    ])

    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b).item()
    Alpha_ib_b = Skew_symmetric(alpha_ib_b)

    # Obtain coordinate transformation matrix from new to old attitude using Rodrigues' formula (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.eye(3) + np.sin(mag_alpha) / mag_alpha * Alpha_ib_b + \
                    (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b
    else:
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude using (5.75)
    C_b_e = C_Earth @ old_C_b_e @ C_new_old

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate average body-to-ECEF-frame coordinate transformation matrix using (5.84) and (5.85)
    if mag_alpha > 1e-8:
        ave_C_b_e = old_C_b_e @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + \
                                 (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b) - \
                    0.5 * Skew_symmetric(np.array([0, 0, alpha_ie])) @ old_C_b_e
    else:
        ave_C_b_e = old_C_b_e - 0.5 * Skew_symmetric(np.array([0, 0, alpha_ie])) @ old_C_b_e

    # Transform specific force to ECEF-frame resolving axes using (5.85)
    f_ib_e = ave_C_b_e @ f_ib_b

    # UPDATE VELOCITY
    # From (5.36)
    v_eb_e = old_v_eb_e + tor_i * (f_ib_e + Gravity_ECEF(old_r_eb_e) - \
                                   2 * Skew_symmetric(np.array([0, 0, OMEGA_IE])) @ old_v_eb_e)

    # UPDATE CARTESIAN POSITION
    # From (5.38)
    r_eb_e = old_r_eb_e + (v_eb_e + old_v_eb_e) * 0.5 * tor_i

    return r_eb_e, v_eb_e, C_b_e