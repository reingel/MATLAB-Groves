import numpy as np
from Radii_of_curvature import Radii_of_curvature
from Skew_symmetric import Skew_symmetric
from Gravity_NED import Gravity_NED

def Kinematics_NED(tor_i, C_b_n, old_C_b_n, v_eb_n, old_v_eb_n, L_b, h_b, old_L_b, old_h_b):
    """
    Calculates specific force and angular rate from input w.r.t and resolved along north, east, and down
    
    Args:
        tor_i (float): Time interval between epochs (s)
        C_b_n (np.ndarray): Body-to-NED coordinate transformation matrix
        old_C_b_n (np.ndarray): Previous body-to-NED coordinate transformation matrix
        v_eb_n (np.ndarray): Velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s)
        old_v_eb_n (np.ndarray): Previous velocity of body frame w.r.t. ECEF frame, resolved along NED (m/s)
        L_b (float): Latitude (rad)
        h_b (float): Height (m)
        old_L_b (float): Previous latitude (rad)
        old_h_b (float): Previous height (m)
    
    Returns:
        tuple: (f_ib_b, omega_ib_b)
            - f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECEF frame, resolved along body-frame axes (m/s^2)
            - omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECEF frame, resolved about body-frame axes (rad/s)
    """
    # Parameters
    OMEGA_IE = 7.292115E-5  # Earth rotation rate (rad/s)
    
    if tor_i > 0:
        # From (2.123), determine the angular rate of the ECEF frame w.r.t the ECI frame, resolved about NED
        omega_ie_n = OMEGA_IE * np.array([np.cos(old_L_b), 0, -np.sin(old_L_b)])
        
        # From (5.44), determine the angular rate of the NED frame w.r.t the ECEF frame, resolved about NED
        old_R_N, old_R_E = Radii_of_curvature(old_L_b)
        R_N, R_E = Radii_of_curvature(L_b)
        old_omega_en_n = np.array([
            old_v_eb_n[1] / (old_R_E + old_h_b),
            -old_v_eb_n[0] / (old_R_N + old_h_b),
            -old_v_eb_n[1] * np.tan(old_L_b) / (old_R_E + old_h_b)
        ])
        omega_en_n = np.array([
            v_eb_n[1] / (R_E + h_b),
            -v_eb_n[0] / (R_N + h_b),
            -v_eb_n[1] * np.tan(L_b) / (R_E + h_b)
        ])
        
        # Obtain coordinate transformation matrix from the old attitude to the new using (5.77)
        C_old_new = C_b_n.T @ (np.eye(3) - Skew_symmetric(omega_ie_n + 0.5 * omega_en_n + 0.5 * old_omega_en_n) * tor_i) @ old_C_b_n
        
        # Calculate the approximate angular rate w.r.t. an inertial frame
        alpha_ib_b = np.zeros((3,))
        alpha_ib_b[0] = 0.5 * (C_old_new[1, 2] - C_old_new[2, 1])
        alpha_ib_b[1] = 0.5 * (C_old_new[2, 0] - C_old_new[0, 2])
        alpha_ib_b[2] = 0.5 * (C_old_new[0, 1] - C_old_new[1, 0])
        
        # Calculate and apply the scaling factor
        temp = np.arccos(0.5 * (C_old_new[0, 0] + C_old_new[1, 1] + C_old_new[2, 2] - 1.0))
        if temp > 2e-5:  # Scaling is 1 if temp is less than this
            alpha_ib_b = alpha_ib_b * temp / np.sin(temp)
        
        # Calculate the angular rate
        omega_ib_b = alpha_ib_b / tor_i
        
        # Calculate the specific force resolved about ECEF-frame axes using (5.54)
        f_ib_n = ((v_eb_n - old_v_eb_n) / tor_i) - Gravity_NED(old_L_b, old_h_b) + \
                 Skew_symmetric(old_omega_en_n + 2 * omega_ie_n) @ old_v_eb_n
        
        # Calculate the average body-to-NED coordinate transformation matrix over the update interval using (5.84) and (5.86)
        mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b).item()
        Alpha_ib_b = Skew_symmetric(alpha_ib_b)
        if mag_alpha > 1e-8:
            ave_C_b_n = old_C_b_n @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + 
                                     (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b) - \
                        0.5 * Skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n
        else:
            ave_C_b_n = old_C_b_n - 0.5 * Skew_symmetric(old_omega_en_n + omega_ie_n) @ old_C_b_n
        
        # Transform specific force to body-frame resolving axes using (5.81)
        f_ib_b = np.linalg.inv(ave_C_b_n) @ f_ib_n
    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))
    
    return f_ib_b, omega_ib_b