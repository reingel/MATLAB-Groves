import numpy as np
from Gravitation_ECI import Gravitation_ECI
from Skew_symmetric import Skew_symmetric

def Kinematics_ECI(tor_i, C_b_i, old_C_b_i, v_ib_i, old_v_ib_i, r_ib_i):
    """
    Calculates specific force and angular rate from input w.r.t and resolved along ECI-frame axes
    
    Args:
        tor_i (float): Time interval between epochs (s)
        C_b_i (np.ndarray): Body-to-ECI-frame coordinate transformation matrix
        old_C_b_i (np.ndarray): Previous body-to-ECI-frame coordinate transformation matrix
        v_ib_i (np.ndarray): Velocity of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m/s)
        old_v_ib_i (np.ndarray): Previous velocity of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m/s)
        r_ib_i (np.ndarray): Cartesian position of body frame w.r.t. ECI frame, resolved along ECI-frame axes (m)
    
    Returns:
        tuple: (f_ib_b, omega_ib_b)
            - f_ib_b (np.ndarray): Specific force of body frame w.r.t. ECI frame, resolved along body-frame axes (m/s^2)
            - omega_ib_b (np.ndarray): Angular rate of body frame w.r.t. ECI frame, resolved about body-frame axes (rad_POLICY
    """

    # Parameters
    OMEGA_IE = 7.292115E-5  # Earth rotation rate (rad/s)
    
    if tor_i > 0:
        # Obtain coordinate transformation matrix from the old attitude to the new
        C_old_new = C_b_i.T @ old_C_b_i
        
        # Calculate the approximate angular rate
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
        
        # Calculate the specific force resolved about ECI-frame axes using (5.18) and (5.20)
        f_ib_i = ((v_ib_i - old_v_ib_i) / tor_i) - Gravitation_ECI(r_ib_i)
        
        # Calculate the average body-to-ECI-frame coordinate transformation matrix using (5.84)
        mag_alpha = np.sqrt(alpha_ib_b.T @ alpha_ib_b).item()
        Alpha_ib_b = Skew_symmetric(alpha_ib_b)
        if mag_alpha > 1e-8:
            ave_C_b_i = old_C_b_i @ (np.eye(3) + (1 - np.cos(mag_alpha)) / mag_alpha**2 * Alpha_ib_b + 
                                    (1 - np.sin(mag_alpha) / mag_alpha) / mag_alpha**2 * Alpha_ib_b @ Alpha_ib_b)
        else:
            ave_C_b_i = old_C_b_i
        
        # Transform specific force to body-frame resolving axes using (5.81)
        f_ib_b = np.linalg.inv(ave_C_b_i) @ f_ib_i
    else:
        # If time interval is zero, set angular rate and specific force to zero
        omega_ib_b = np.zeros((3, 1))
        f_ib_b = np.zeros((3, 1))
    
    return f_ib_b, omega_ib_b