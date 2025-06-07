import numpy as np
from Radii_of_curvature import Radii_of_curvature
from Skew_symmetric import Skew_symmetric
from Gravity_NED import Gravity_NED

def Nav_equations_NED(tor_i, old_L_b, old_lambda_b, old_h_b, old_v_eb_n, old_C_b_n, f_ib_b, omega_ib_b):
    """
    Runs precision local-navigation-frame inertial navigation equations
    (Note: only the attitude update and specific force frame transformation phases are precise)

    Args:
        tor_i (float): Time interval between epochs (s)
        old_L_b (float): Previous latitude (rad)
        old_lambda_b (float): Previous longitude (rad)
        old_h_b (float): Previous height (m)
        old_v_eb_n (np.ndarray): Previous velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s)
        old_C_b_n (np.ndarray): Previous body-to-NED coordinate transformation matrix
        f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECEF frame, resolved along body-frame axes (m/s^2)
        omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECEF frame, resolved about body-frame axes (rad/s)

    Returns:
        tuple: (L_b, lambda_b, h_b, v_eb_n, C_b_n)
            - L_b (float): Latitude (rad)
            - lambda_b (float): Longitude (rad)
            - h_b (float): Height (m)
            - v_eb_n (np.ndarray): Velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s)
            - C_b_n (np.ndarray): Body-to-NED coordinate transformation matrix
    """
    # Parameters
    OMEGA_IE = 7.292115E-5  # Earth rotation rate (rad/s)

    # PRELIMINARIES
    # Calculate attitude increment, magnitude, and skew-symmetric matrix
    alpha_ib_b = omega_ib_b * tor_i
    mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b).item()
    Alpha_ib_b = Skew_symmetric(alpha_ib_b)

    # From (2.123), determine the angular rate of the ECEF frame w.r.t the ECI frame, resolved about NED
    omega_ie_n = OMEGA_IE * np.array([np.cos(old_L_b), 0, -np.sin(old_L_b)])

    # From (5.44), determine the angular rate of the NED frame w.r.t the ECEF frame, resolved about NED
    old_R_N, old_R_E = Radii_of_curvature(old_L_b)
    old_omega_en_n = np.array([
        old_v_eb_n[1] / (old_R_E + old_h_b),
        -old_v_eb_n[0] / (old_R_N + old_h_b),
        -old_v_eb_n[1] * np.tan(old_L_b) / (old_R_E + old_h_b)
    ])

    # SPECIFIC FORCE FRAME TRANSFORMATION
    # Calculate average body-to-NED coordinate transformation matrix using (5.84) and (5.86)
    if mag_alpha > 1e-8:
        ave_C_b_n = old_C_b_n @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + \
                                 (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b) - \
                    0.5 * Skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n
    else:
        ave_C_b_n = old_C_b_n - 0.5 * Skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n

    # Transform specific force to NED-frame resolving axes using (5.86)
    f_ib_n = ave_C_b_n @ f_ib_b

    # UPDATE VELOCITY
    # From (5.54)
    v_eb_n = old_v_eb_n + tor_i * (f_ib_n + Gravity_NED(old_L_b, old_h_b) - \
                                   Skew_symmetric(old_omega_en_n + 2 * omega_ie_n) @ old_v_eb_n)

    # UPDATE CURVILINEAR POSITION
    # Update height using (5.56)
    h_b = old_h_b - 0.5 * tor_i * (old_v_eb_n[2] + v_eb_n[2])

    # Update latitude using (5.56)
    L_b = old_L_b + 0.5 * tor_i * (old_v_eb_n[0] / (old_R_N + old_h_b) + v_eb_n[0] / (old_R_N + h_b))

    # Calculate meridian and transverse radii of curvature
    R_N, R_E = Radii_of_curvature(L_b)

    # Update longitude using (5.56)
    lambda_b = old_lambda_b + 0.5 * tor_i * (old_v_eb_n[1] / ((old_R_E + old_h_b) * np.cos(old_L_b)) + \
                                             v_eb_n[1] / ((R_E + h_b) * np.cos(L_b)))

    # ATTITUDE UPDATE
    # From (5.44), determine the angular rate of the NED frame w.r.t the ECEF frame
    omega_en_n = np.array([
        v_eb_n[1] / (R_E + h_b),
        -v_eb_n[0] / (R_N + h_b),
        -v_eb_n[1] * np.tan(L_b) / (R_E + h_b)
    ])

    # Obtain coordinate transformation matrix from new to old attitude using Rodrigues' formula (5.73)
    if mag_alpha > 1e-8:
        C_new_old = np.eye(3) + np.sin(mag_alpha) / mag_alpha * Alpha_ib_b + \
                    (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b
    else:
        C_new_old = np.eye(3) + Alpha_ib_b

    # Update attitude using (5.77)
    C_b_n = (np.eye(3) - Skew_symmetric(omega_ie_n + 0.5 * omega_en_n + 0.5 * old_omega_en_n) * tor_i) @ old_C_b_n @ C_new_old

    return L_b, lambda_b, h_b, v_eb_n, C_b_n