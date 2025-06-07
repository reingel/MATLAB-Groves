import numpy as np
from Euler_to_CTM import Euler_to_CTM
from NED_to_ECEF import NED_to_ECEF
from Satellite_positions_and_velocities import Satellite_positions_and_velocities
from Initialize_GNSS_biases import Initialize_GNSS_biases
from Generate_GNSS_measurements import Generate_GNSS_measurements
from GNSS_LS_position_velocity import GNSS_LS_position_velocity
from pv_ECEF_to_NED import pv_ECEF_to_NED
from CTM_to_Euler import CTM_to_Euler
from Initialize_NED_attitude import Initialize_NED_attitude
from Calculate_errors_NED import Calculate_errors_NED
from Initialize_TC_P_matrix import Initialize_TC_P_matrix
from Kinematics_ECEF import Kinematics_ECEF
from IMU_model import IMU_model
from Nav_equations_ECEF import Nav_equations_ECEF
from ECEF_to_NED import ECEF_to_NED
from TC_KF_Epoch import TC_KF_Epoch

def Tightly_coupled_INS_GNSS(in_profile, no_epochs, initialization_errors, IMU_errors, GNSS_config, TC_KF_config):
    """
    Simulates inertial navigation using ECEF navigation equations and kinematic model,
    GNSS and tightly coupled INS/GNSS integration.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    Inputs:
        in_profile: True motion profile array (numpy array)
        no_epochs: Number of epochs of profile data (int)
        initialization_errors: Dict with keys:
            - delta_r_eb_n: position error resolved along NED (m)
            - delta_v_eb_n: velocity error resolved along NED (m/s)
            - delta_eul_nb_n: attitude error as NED Euler angles (rad)
        IMU_errors: Dict with keys:
            - delta_r_eb_n: position error resolved along NED (m)
            - b_a: Accelerometer biases (m/s^2)
            - b_g: Gyro biases (rad/s)
            - M_a: Accelerometer scale factor and cross coupling errors
            - M_g: Gyro scale factor and cross coupling errors
            - G_g: Gyro g-dependent biases (rad-sec/m)
            - accel_noise_root_PSD: Accelerometer noise root PSD (m s^-1.5)
            - gyro_noise_root_PSD: Gyro noise root PSD (rad s^-0.5)
            - accel_quant_level: Accelerometer quantization level (m/s^2)
            - gyro_quant_level: Gyro quantization level (rad/s)
        GNSS_config: Dict with keys:
            - epoch_interval: Interval between GNSS epochs (s)
            - init_est_r_ea_e: Initial estimated position (m; ECEF)
            - no_sat: Number of satellites in constellation
            - r_os: Orbital radius of satellites (m)
            - inclination: Inclination angle of satellites (deg)
            - const_delta_lambda: Longitude offset of constellation (deg)
            - const_delta_t: Timing offset of constellation (s)
            - mask_angle: Mask angle (deg)
            - SIS_err_SD: Signal in space error SD (m)
            - zenith_iono_err_SD: Zenith ionosphere error SD (m)
            - zenith_trop_err_SD: Zenith troposphere error SD (m)
            - code_track_err_SD: Code tracking error SD (m)
            - rate_track_err_SD: Range rate tracking error SD (m/s)
            - rx_clock_offset: Receiver clock offset at time=0 (m)
            - rx_clock_drift: Receiver clock drift at time=0 (m/s)
        TC_KF_config: Dict with keys:
            - init_att_unc: Initial attitude uncertainty per axis (rad)
            - init_vel_unc: Initial velocity uncertainty per axis (m/s)
            - init_pos_unc: Initial position uncertainty per axis (m)
            - init_b_a_unc: Initial accel. bias uncertainty (m/s^2)
            - init_b_g_unc: Initial gyro. bias uncertainty (rad/s)
            - init_clock_offset_unc: Initial clock offset uncertainty per axis (m)
            - init_clock_drift_unc: Initial clock drift uncertainty per axis (m/s)
            - gyro_noise_PSD: Gyro noise PSD (rad^2/s)
            - accel_noise_PSD: Accelerometer noise PSD (m^2 s^-3)
            - accel_bias_PSD: Accelerometer bias random walk PSD (m^2 s^-5)
            - gyro_bias_PSD: Gyro bias random walk PSD (rad^2 s^-3)
            - clock_freq_PSD: Receiver clock frequency-drift PSD (m^2/s^3)
            - clock_phase_PSD: Receiver clock phase-drift PSD (m^2/s)
            - pseudo_range_SD: Pseudo-range measurement noise SD (m)
            - range_rate_SD: Pseudo-range rate measurement noise SD (m/s)

    Outputs:
        out_profile: Navigation solution as a motion profile array
        out_errors: Navigation solution error array
        out_IMU_bias_est: Kalman filter IMU bias estimate array
        out_clock: GNSS Receiver clock estimate array
        out_KF_SD: Output Kalman filter state uncertainties
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
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
    )

    # Determine satellite positions and velocities
    sat_r_es_e, sat_v_es_e = Satellite_positions_and_velocities(old_time, GNSS_config)

    # Initialize GNSS biases
    GNSS_biases = Initialize_GNSS_biases(sat_r_es_e, old_true_r_eb_e, true_L_b, true_lambda_b, GNSS_config)

    # Generate GNSS measurements
    GNSS_measurements, no_GNSS_meas = Generate_GNSS_measurements(
        old_time, sat_r_es_e, sat_v_es_e, old_true_r_eb_e, true_L_b, true_lambda_b,
        old_true_v_eb_e, GNSS_biases, GNSS_config
    )

    # Determine Least-squares GNSS position solution
    old_est_r_eb_e, old_est_v_eb_e, est_clock = GNSS_LS_position_velocity(
        GNSS_measurements, no_GNSS_meas, GNSS_config['init_est_r_ea_e'], np.zeros(3)
    )
    old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n = pv_ECEF_to_NED(
        old_est_r_eb_e, old_est_v_eb_e
    )
    est_L_b = old_est_L_b

    # Initialize estimated attitude solution
    old_est_C_b_n = Initialize_NED_attitude(true_C_b_n, initialization_errors)
    _, _, old_est_C_b_e = NED_to_ECEF(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n
    )

    # Initialize output arrays
    out_profile = np.zeros((no_epochs, 10))
    out_errors = np.zeros((no_epochs, 10))
    out_IMU_bias_est = np.zeros((no_epochs, 7))
    out_clock = np.zeros((no_epochs, 3))
    out_KF_SD = np.zeros((no_epochs, 18))

    # Generate output profile record
    out_profile[0, 0] = old_time
    out_profile[0, 1] = old_est_L_b
    out_profile[0, 2] = old_est_lambda_b
    out_profile[0, 3] = old_est_h_b
    out_profile[0, 4:7] = old_est_v_eb_n
    out_profile[0, 7:10] = CTM_to_Euler(old_est_C_b_n.T)

    # Determine errors and generate output record
    delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = Calculate_errors_NED(
        old_est_L_b, old_est_lambda_b, old_est_h_b, old_est_v_eb_n, old_est_C_b_n,
        true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
    )
    out_errors[0, 0] = old_time
    out_errors[0, 1:4] = delta_r_eb_n
    out_errors[0, 4:7] = delta_v_eb_n
    out_errors[0, 7:10] = delta_eul_nb_n

    # Initialize Kalman filter P matrix and IMU bias states
    P_matrix = Initialize_TC_P_matrix(TC_KF_config)
    est_IMU_bias = np.zeros(6)
    quant_residuals = np.zeros(6)

    # Generate IMU bias and clock output records
    out_IMU_bias_est[0, 0] = old_time
    out_IMU_bias_est[0, 1:7] = est_IMU_bias
    out_clock[0, 0] = old_time
    out_clock[0, 1:3] = est_clock

    # Generate KF uncertainty record
    out_KF_SD[0, 0] = old_time
    for i in range(17):
        out_KF_SD[0, i + 1] = np.sqrt(P_matrix[i, i])

    # Initialize GNSS model timing
    time_last_GNSS = old_time
    GNSS_epoch = 1

    # Progress bar
    print('Processing: ' + '.' * 20, end='')
    progress_mark = 0
    progress_epoch = 0

    # Main loop
    for epoch in range(1, no_epochs):
        # Update progress bar
        if (epoch - progress_epoch) > (no_epochs / 20):
            progress_mark += 1
            progress_epoch = epoch
            print('\b' * 20 + '|' * progress_mark + '.' * (20 - progress_mark), end='')

        # Input data from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = in_profile[epoch, 4:7]
        true_eul_nb = in_profile[epoch, 7:10]
        true_C_b_n = Euler_to_CTM(true_eul_nb).T
        true_r_eb_e, true_v_eb_e, true_C_b_e = NED_to_ECEF(
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
        )

        # Time interval
        tor_i = time - old_time

        # Calculate specific force and angular rate
        true_f_ib_b, true_omega_ib_b = Kinematics_ECEF(
            tor_i, true_C_b_e, old_true_C_b_e, true_v_eb_e, old_true_v_eb_e, old_true_r_eb_e
        )

        # Simulate IMU errors
        meas_f_ib_b, meas_omega_ib_b, quant_residuals = IMU_model(
            tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors, quant_residuals
        )

        # Correct IMU errors
        meas_f_ib_b = meas_f_ib_b - est_IMU_bias[:3]
        meas_omega_ib_b = meas_omega_ib_b - est_IMU_bias[3:6]

        # Update estimated navigation solution
        est_r_eb_e, est_v_eb_e, est_C_b_e = Nav_equations_ECEF(
            tor_i, old_est_r_eb_e, old_est_v_eb_e, old_est_C_b_e, meas_f_ib_b, meas_omega_ib_b
        )

        # Determine whether to update GNSS simulation and run Kalman filter
        if (time - time_last_GNSS) >= GNSS_config['epoch_interval']:
            GNSS_epoch += 1
            tor_s = time - time_last_GNSS
            time_last_GNSS = time

            # Determine satellite positions and velocities
            sat_r_es_e, sat_v_es_e = Satellite_positions_and_velocities(time, GNSS_config)

            # Generate GNSS measurements
            GNSS_measurements, no_GNSS_meas = Generate_GNSS_measurements(
                time, sat_r_es_e, sat_v_es_e, true_r_eb_e, true_L_b, true_lambda_b,
                true_v_eb_e, GNSS_biases, GNSS_config
            )

            # Run Integration Kalman filter
            est_C_b_e, est_v_eb_e, est_r_eb_e, est_IMU_bias, est_clock, P_matrix = TC_KF_Epoch(
                GNSS_measurements, no_GNSS_meas, tor_s, est_C_b_e, est_v_eb_e, est_r_eb_e,
                est_IMU_bias, est_clock, P_matrix, meas_f_ib_b, est_L_b, TC_KF_config
            )

            # Generate IMU bias and clock output records
            out_IMU_bias_est[GNSS_epoch - 1, 0] = time
            out_IMU_bias_est[GNSS_epoch - 1, 1:7] = est_IMU_bias
            out_clock[GNSS_epoch - 1, 0] = time
            out_clock[GNSS_epoch - 1, 1:3] = est_clock

            # Generate KF uncertainty output record
            out_KF_SD[GNSS_epoch - 1, 0] = time
            for i in range(17):
                out_KF_SD[GNSS_epoch - 1, i + 1] = np.sqrt(P_matrix[i, i])

        # Convert navigation solution to NED
        est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n = ECEF_to_NED(
            est_r_eb_e, est_v_eb_e, est_C_b_e
        )

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = est_L_b
        out_profile[epoch, 2] = est_lambda_b
        out_profile[epoch, 3] = est_h_b
        out_profile[epoch, 4:7] = est_v_eb_n
        out_profile[epoch, 7:10] = CTM_to_Euler(est_C_b_n.T)

        # Determine errors and generate output record
        delta_r_eb_n, delta_v_eb_n, delta_eul_nb_n = Calculate_errors_NED(
            est_L_b, est_lambda_b, est_h_b, est_v_eb_n, est_C_b_n,
            true_L_b, true_lambda_b, true_h_b, true_v_eb_n, true_C_b_n
        )
        out_errors[epoch, 0] = time
        out_errors[epoch, 1:4] = delta_r_eb_n
        out_errors[epoch, 4:7] = delta_v_eb_n
        out_errors[epoch, 7:10] = delta_eul_nb_n

        # Reset old values
        old_time = time
        old_true_r_eb_e = true_r_eb_e
        old_true_v_eb_e = true_v_eb_e
        old_true_C_b_e = true_C_b_e
        old_est_r_eb_e = est_r_eb_e
        old_est_v_eb_e = est_v_eb_e
        old_est_C_b_e = est_C_b_e

    # Complete progress bar
    print('\b' * 20 + '|' * 20)

    return out_profile, out_errors, out_IMU_bias_est, out_clock, out_KF_SD