import numpy as np
from tqdm import tqdm
from Initialize_NED import Initialize_NED
from Euler_to_CTM import Euler_to_CTM
from Kinematics_ECEF import Kinematics_ECEF
from IMU_model import IMU_model
from Nav_equations_ECEF import Nav_equations_ECEF
from NED_to_ECEF import NED_to_ECEF
from ECEF_to_NED import ECEF_to_NED
from Calculate_errors_NED import Calculate_errors_NED
from CTM_to_Euler import CTM_to_Euler

def Inertial_navigation_ECEF(in_profile, no_epochs, initialization_errors, IMU_errors):
    """
    Simulates inertial navigation using ECEF navigation equations and kinematic model
    
    Args:
        in_profile (np.ndarray): True motion profile array (no_epochs x 10)
        no_epochs (int): Number of epochs of profile data
        initialization_errors (dict): Dictionary containing:
            - delta_r_eb_n (np.ndarray): Position error resolved along NED (m)
            - delta_v_eb_n (np.ndarray): Velocity error resolved along NED (m/s)
            - delta_eul_nb_n (np.ndarray): Attitude error as NED Euler angles (rad)
        IMU_errors (dict): Dictionary containing IMU error parameters
            - delta_r_eb_n (np.ndarray): Position error resolved along NED (m)
            - b_a (np.ndarray): Accelerometer biases (m/s^2)
            - b_g (np.ndarray): Gyro biases (rad/s)
            - M_a (np.ndarray): Accelerometer scale factor and cross coupling errors
            - M_g (np.ndarray): Gyro scale factor and cross coupling errors
            - G_g (np.ndarray): Gyro g-dependent biases (rad-sec/m)
            - accel_noise_root_PSD (float): Accelerometer noise root PSD (m s^-1.5)
            - gyro_noise_root_PSD (float): Gyro noise root PSD (rad s^-0.5)
            - accel_quant_level (float): Accelerometer quantization level (m/s^2)
            - gyro_quant_level (float): Gyro quantization level (rad/s)
    
    Returns:
        tuple: (out_profile, out_errors)
            - out_profile (np.ndarray): Navigation solution as a motion profile array (no_epochs x 10)
            - out_errors (np.ndarray): Navigation solution error array (no_epochs x 10)
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]
    true_L_b = in_profile[0, 1]
    true_lambda_b = in_profile[0, 2]
    true_h_b = in_profile[0, 3]
    true_v_eb_n = in_profile[0, 4:7]
    true_eul_nb = in_profile[0, 7:10]
    true_C_b_n = Euler_to_CTM(true_eul_nb).T
    old_true_r_eb_e, old_true_v_eb_e, old_true_C_b_e = NED_to_ECEF(
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

    # Initialize estimated navigation solution
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n = Initialize_NED(
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n, initialization_errors)

    # Initialize output profile and error arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))

    # Generate initial output profile and error records
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[0, 4:7] = old_est_v_eb_n.T
    out_profile[0, 7:10] = CTM_to_Euler(old_est_C_b_n.T).T

    out_errors[0, 0] = old_time
    out_errors[0, 1:4] = initialization_errors['delta_r_eb_n'].T
    out_errors[0, 4:7] = initialization_errors['delta_v_eb_n'].T
    out_errors[0, 7:10] = initialization_errors['delta_eul_nb_n'].T

    old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e = NED_to_ECEF(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n)

    # Initialize IMU quantization residuals
    quant_residuals = np.zeros((6,))

    # Main loop with progress bar
    for epoch in tqdm(range(1, no_epochs), desc="Processing", ncols=80):
        # Input data from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = in_profile[epoch, 4:7]
        true_eul_nb = in_profile[epoch, 7:10]
        true_C_b_n = Euler_to_CTM(true_eul_nb).T
        true_r_eb_e, true_v_eb_e, true_C_b_e = NED_to_ECEF(
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = Kinematics_ECEF(
            tor_i, true_C_b_e, old_true_C_b_e, true_v_eb_e, old_true_v_eb_e, old_true_r_eb_e)

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = IMU_model(
            tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors, quant_residuals)

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
            tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e, meas_f_ib_b, meas_omega_ib_b)

        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = ECEF_to_NED(
            est_r_eb_e, est_v_eb_e, est_C_b_e)

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[epoch, 4:7] = est_v_eb_n.T
        out_profile[epoch, 7:10] = CTM_to_Euler(est_C_b_n.T).T

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = Calculate_errors_NED(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
        out_errors[epoch, 0] = time
        out_errors[epoch, 1:4] = delta_r_eb_n.T
        out_errors[epoch, 4:7] = delta_v_eb_n.T
        out_errors[epoch, 7:10] = delta_eul_nb_n.T

        # Reset old values
        old_time = time
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_C_b_e = true_C_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_C_b_e = est_C_b_e

    return out_profile, out_errors