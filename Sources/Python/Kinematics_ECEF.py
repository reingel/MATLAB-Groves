import numpy as np
from Gravity_ECEF import Gravity_ECEF
from Skew_symmetric import Skew_symmetric

def Kinematics_ECEF(tor_i, C_b_e, old_C_b_e, v_eb_e, old_v_eb_e, r_eb_e):
    """
    Calculates specific force and angular rate from input w.r.t and resolved along ECEF-frame axes
    
    Args:
        tor_i (float): Time interval between epochs (s)
        C_b_e (np.ndarray): Body-to-ECEF-frame coordinate transformation matrix
        old_C_b_e (np.ndarray): Previous body-to-ECEF-frame coordinate transformation matrix
        v_eb_e (np.ndarray): Velocity of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m/s)
        old_v_eb_e (np.ndarray): Previous velocity of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m/s)
        r_eb_e (np.ndarray): Cartesian position of body frame w.r.t. ECEF frame, resolved along ECEF-frame axes (m)
    
    Returns:
        tuple: (f_ib_b, omega_ib_b)
            - f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECEF frame, resolved along body-frame axes (m/s^2)
            - omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECEF frame, resolved about body-frame axes (rad/s)
    """
    # Parameters
    OMEGA_IE = 7.292115E-5  # Earth rotation rate (rad/s)
    
    if tor_i > 0:
        # From (2.145) determine the Earth rotation over the update interval
        alpha_ie = OMEGA_IE * tor_i
        C_Earth = np.array([
            [np.cos(alpha_ie), np.sin(alpha_ie), 0],
            [-np.sin(alpha_ie), np.cos(alpha_ie), 0],
            [0, 0, 1]
        ])
        
        # Obtain coordinate transformation matrix from the old attitude to the new
        C_old_new = C_b_e.T @ C_Earth @ old_C_b_e
        
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
        
        # Calculate the specific force resolved about ECEF-frame axes using (5.36)
        f_ib_e = ((v_eb_e - old_v_eb_e) / tor_i) - Gravity_ECEF(r_eb_e) + \
                 2 * Skew_symmetric(np.array([0, 0, OMEGA_IE])) @ old_v_eb_e
        
        # Calculate the average body-to-ECEF-frame coordinate transformation matrix using (5.84) and (5.85)
        mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b).item()
        Alpha_ib_b = Skew_symmetric(alpha_ib_b)
        if mag_alpha > 1e-8:
            ave_C_b_e = old_C_b_e @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + 
                                     (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b) - \
                        0.5 * Skew_symmetric(np.array([0, 0, alpha_ie])) @ old_C_b_e
        else:
            ave_C_b_e = old_C_b_e - 0.5 * Skew_symmetric(np.array([0, 0, alpha_ie])) @ old_C_b_e
        
        # Transform specific force to body-frame resolving axes using (5.81)
        f_ib_b = np.linalg.inv(ave_C_b_e) @ f_ib_e
    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))
    
    return f_ib_b, omega_ib_b