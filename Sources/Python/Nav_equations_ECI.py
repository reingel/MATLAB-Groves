import numpy as np
from Gravitation_ECI import Gravitation_ECI
from Skew_symmetric import Skew_symmetric

def Nav_equations_ECI(tor_i, old_r_ib_i, old_v_ib_i, old_C_b_i, f_ib_b, omega_ib_b):
    """
    Runs precision ECI-frame inertial navigation equations

    Args:
        tor_i (float): Time interval between epochs (s)
        old_r_ib_i (np.ndarray): Previous Cartesian position of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m)
        old_v_ib_i (np.ndarray): Previous velocity of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m/s)
        old_C_b_i (np.ndarray): Previous body-to-ECI-frame coordinate transformation matrix
        f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECI frame, resolved along body-frame axes, averaged over time interval (m/s^2)
        omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECI frame, resolved about body-frame axes, averaged over time interval (rad/s)

    Returns:
        tuple: (r_ib_i, v_ib_i, C_b_i)
            - r_ib_i (np.ndarray): Cartesian position of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m)
            - v_ib_i (np.ndarray): Velocity of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m/s)
            - C_b_i (np.ndarray): Body-to-ECI-frame coordinate transformation matrix
    """
    # ATTITUDE UPDATE
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

    # Update attitude
    C_b_i = old_C_b_i @ C_new_old

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate average body-to-ECI-frame coordinate transformation matrix using (5.84)
    if mag_alpha > 1e-8:
        ave_C_b_i = old_C_b_i @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + \
                                 (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b)
    else:
        ave_C_b_i = old_C_b_i

    # Transform specific force to ECI-frame resolving axes using (5.81)
    f_ib_i = ave_C_b_i @ f_ib_b

    # UPDATE VELOCITY
    # From (5.18) and (5.20)
    v_ib_i = old_v_ib_i + tor_i * (f_ib_i + Gravitation_ECI(old_r_ib_i))

    # UPDATE CARTESIAN POSITION
    # From (5.23)
    r_ib_i = old_r_ib_i + (v_ib_i + old_v_ib_i) * 0.5 * tor_i

    return r_ib_i, v_ib_i, C_b_i