import numpy as np
from pv_NED_to_ECEF import pv_NED_to_ECEF
from Satellite_positions_and_velocities import Satellite_positions_and_velocities
from Initialize_GNSS_biases import Initialize_GNSS_biases
from Generate_GNSS_measurements import Generate_GNSS_measurements
from GNSS_LS_position_velocity import GNSS_LS_position_velocity
from pv_ECEF_to_NED import pv_ECEF_to_NED
from Euler_to_CTM import Euler_to_CTM
from CTM_to_Euler import CTM_to_Euler
from Initialize_GNSS_KF import Initialize_GNSS_KF
from GNSS_KF_Epoch import GNSS_KF_Epoch
from Calculate_errors_NED import Calculate_errors_NED

def GNSS_Kalman_Filter(in_profile, no_epochs, GNSS_config, GNSS_KF_config):
    """
    GNSS_Kalman_Filter - Simulates stand-alone GNSS using an Extended Kalman
    positioning algorithm

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        in_profile (numpy.ndarray): True motion profile array, shape (no_epochs, 10)
            - Column 1: time (sec)
            - Column 2: latitude (rad)
            - Column 3: longitude (rad)
            - Column 4: height (m)
            - Column 5: north velocity (m/s)
            - Column 6: east velocity (m/s)
            - Column 7: down velocity (m/s)
            - Column 8: roll angle of body w.r.t NED (rad)
            - Column 9: pitch angle of body w.r.t NED (rad)
            - Column 10: yaw angle of body w.r.t NED (rad)
        no_epochs (int): Number of epochs of profile data
        GNSS_config (dict): Configuration dictionary with keys:
            - epoch_interval (float): Interval between GNSS epochs (s)
            - init_est_r_ea_e (numpy.ndarray): Initial estimated position (m; ECEF), shape (3,)
            - no_sat (int): Number of satellites in constellation
            - r_os (float): Orbital radius of satellites (m)
            - inclination (float): Inclination angle of satellites (deg)
            - const_delta_lambda (float): Longitude offset of constellation (deg)
            - const_delta_t (float): Timing offset of constellation (s)
            - mask_angle (float): Mask angle (deg)
            - SIS_err_SD (float): Signal in space error SD (m)
            - zenith_iono_err_SD (float): Zenith ionosphere error SD (m)
            - zenith_trop_err_SD (float): Zenith troposphere error SD (m)
            - code_track_err_SD (float): Code tracking error SD (m)
            - rate_track_err_SD (float): Range rate tracking error SD (m/s)
            - rx_clock_offset (float): Receiver clock offset at time=0 (m)
            - rx_clock_drift (float): Receiver clock drift at time=0 (m/s)
        GNSS_KF_config (dict): Kalman filter configuration dictionary with keys:
            - init_pos_unc (float): Initial position uncertainty per axis (m)
            - init_vel_unc (float): Initial velocity uncertainty per axis (m/s)
            - init_clock_offset_unc (float): Initial clock offset uncertainty per axis (m)
            - init_clock_drift_unc (float): Initial clock drift uncertainty per axis (m/s)
            - accel_PSD (float): Acceleration PSD per axis (m^2/s^3)
            - clock_freq_PSD (float): Receiver clock frequency-drift PSD (m^2/s^3)
            - clock_phase_PSD (float): Receiver clock phase-drift PSD (m^2/s)
            - pseudo_range_SD (float): Pseudo-range measurement noise SD (m)
            - range_rate_SD (float): Pseudo-range rate measurement noise SD (m/s)

    Returns:
        tuple: (out_profile, out_errors, out_clock, out_KF_SD)
            - out_profile (numpy.ndarray): Navigation solution as a motion profile array, shape (no_epochs, 10)
            - out_errors (numpy.ndarray): Navigation solution error array, shape (no_epochs, 10)
            - out_clock (numpy.ndarray): Receiver clock estimate array, shape (no_epochs, 3)
            - out_KF_SD (numpy.ndarray): Kalman filter state uncertainties, shape (no_epochs, 9)
    """
    # Initialize true navigation solution
    old_time = in_profile[0, 0]
    true_L_b = in_profile[0, 1]
    true_lambda_b = in_profile[0, 2]
    true_h_b = in_profile[0, 3]
    true_v_eb_n = in_profile[0, 4:7]
    true_eul_nb = in_profile[0, 7:10]
    true_C_b_n = Euler_to_CTM(true_eul_nb).T
    true_r_eb_e, true_v_eb_e = pv_NED_to_ECEF(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

    time_last_GNSS = old_time
    GNSS_epoch = 0

    # Initialize output arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_clock = np.zeros((no_epochs, 3))
    out_KF_SD = np.zeros((no_epochs, 9))

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = Satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize the GNSS biases
    GNSS_biases = Initialize_GNSS_biases(sat_r_es_e, true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = Generate_GNSS_measurements(
        old_time, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_L_b, true_lambda_b,
        true_v_eb_e, GNSS_biases, GNSS_config)

    # Determine Least-squares GNSS position solution
    est_r_eb_e, est_v_eb_e, est_clock = GNSS_LS_position_velocity(
        GNSS_measurements, no_GNSS_meas, GNSS_config['init_est_r_ea_e'], np.zeros(3))

    # Initialize Kalman filter
    x_est, P_matrix = Initialize_GNSS_KF(est_r_eb_e, est_v_eb_e, est_clock, GNSS_KF_config)

    est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

    est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ECEF_to_NED(x_est[:3], x_est[3:6])

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = est_L_b
    out_profile[0, 2] = est_lambda_b
    out_profile[0, 3] = est_h_b
    out_profile[0, 4:7] = est_v_eb_n
    out_profile[0, 7:10] = CTM_to_Euler(est_C_b_n.T)

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = Calculate_errors_NED(
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
    out_errors[0, 0] = old_time
    out_errors[0, 1:4] = delta_r_eb_n
    out_errors[0, 4:7] = delta_v_eb_n
    out_errors[0, 7:10] = [0, 0, 0]

    # Generate clock output record
    out_clock[0, 0] = old_time
    out_clock[0, 1:3] = x_est[6:8]

    # Generate KF uncertainty record
    out_KF_SD[0, 0] = old_time
    for i in range(8):
        out_KF_SD[0, i + 1] = np.sqrt(P_matrix[i, i])

    # Progress bar
    dots = '....................'
    bars = '||||||||||||||||||||'
    rewind = '\b' * 20
    print(f"Processing: {dots}", end='', flush=True)
    progress_mark = 0
    progress_epoch = 0

    # Main loop
    for epoch in range(1, no_epochs):
        # Update progress bar
        if (epoch - progress_epoch) > (no_epochs / 20):
            progress_mark += 1
            progress_epoch = epoch
            print(f"{rewind}{bars[:progress_mark]}{dots[progress_mark:]}", end='', flush=True)

        # Input time from motion profile
        time = in_profile[epoch, 0]

        # Determine whether to update GNSS simulation
        if (time - time_last_GNSS) >= GNSS_config['epoch_interval']:
            GNSS_epoch += 1
            tor_s = time - time_last_GNSS  # KF time interval
            time_last_GNSS = time

            # Input data from motion profile
            true_L_b = in_profile[epoch, 1]
            true_lambda_b = in_profile[epoch, 2]
            true_h_b = in_profile[epoch, 3]
            true_v_eb_n = in_profile[epoch, 4:7]
            true_eul_nb = in_profile[epoch, 7:10]
            true_C_b_n = Euler_to_CTM(true_eul_nb).T
            true_r_eb_e, true_v_eb_e = pv_NED_to_ECEF(true_L_b, true_lambda_b, true_h_b, true_v_eb_n)

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = Satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = Generate_GNSS_measurements(
                time, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_L_b, true_lambda_b,
                true_v_eb_e, GNSS_biases, GNSS_config)

            # Update GNSS position solution
            x_est, P_matrix = GNSS_KF_Epoch(GNSS_measurements, no_GNSS_meas,
                                            tor_s, x_est, P_matrix, GNSS_KF_config)
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n = pv_ECEF_to_NED(x_est[:3], x_est[3:6])

            est_C_b_n = true_C_b_n  # This sets the attitude errors to zero

            # Generate output profile record
            out_profile[GNSS_epoch, 0] = time
            out_profile[GNSS_epoch, 1] = est_L_b
            out_profile[GNSS_epoch, 2] = est_lambda_b
            out_profile[GNSS_epoch, 3] = est_h_b
            out_profile[GNSS_epoch, 4:7] = est_v_eb_n
            out_profile[GNSS_epoch, 7:10] = CTM_to_Euler(est_C_b_n.T)

            # Determine errors and generate output record
            delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = Calculate_errors_NED(
                est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
                true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n)
            out_errors[GNSS_epoch, 0] = time
            out_errors[GNSS_epoch, 1:4] = delta_r_eb_n
            out_errors[GNSS_epoch, 4:7] = delta_v_eb_n
            out_errors[GNSS_epoch, 7:10] = [0, 0, 0]

            # Generate clock output record
            out_clock[GNSS_epoch, 0] = time
            out_clock[GNSS_epoch, 1:3] = x_est[6:8]

            # Generate KF uncertainty record
            out_KF_SD[GNSS_epoch, 0] = time
            for i in range(8):
                out_KF_SD[GNSS_epoch, i + 1] = np.sqrt(P_matrix[i, i])

            # Reset old values
            old_time = time

    # Complete progress bar
    print(f"{rewind}{bars}")

    # Trim output arrays to match GNSS_epoch
    out_profile = out_profile[:GNSS_epoch + 1, :]
    out_errors = out_errors[:GNSS_epoch + 1, :]
    out_clock = out_clock[:GNSS_epoch + 1, :]
    out_KF_SD = out_KF_SD[:GNSS_epoch + 1, :]

    return out_profile, out_errors, out_clock, out_KF_SD