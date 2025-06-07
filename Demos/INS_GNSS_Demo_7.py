import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Sources/Python'))
import numpy as np
import matplotlib.pyplot as plt
from Read_profile import Read_profile
from Tightly_coupled_INS_GNSS import Tightly_coupled_INS_GNSS
from Write_profile import Write_profile
from Write_errors import Write_errors
from Plot_profile import Plot_profile
from Plot_errors import Plot_errors

# Constants
deg_to_rad = 0.01745329252
micro_g_to_meters_per_second_squared = 9.80665e-6

# CONFIGURATION
input_profile_folder = os.path.join(os.path.dirname(__file__), '..', 'Profiles/Input')
output_profile_folder = os.path.join(os.path.dirname(__file__), '..', 'Profiles/Output')
# Input truth motion profile filename
input_profile_name = os.path.join(input_profile_folder, 'Profile_1.csv')
# Output motion profile and error filenames
output_profile_name = os.path.join(output_profile_folder, 'INS_GNSS_Demo_7_Profile.csv')
output_errors_name = os.path.join(output_profile_folder, 'INS_GNSS_Demo_7_Errors.csv')

# Initialization errors
initialization_errors = {
    'delta_eul_nb_n': np.array([-0.5, 0.4, 2]) * deg_to_rad  # Attitude error (rad; N,E,D)
}

# IMU errors (Consumer-grade)
IMU_errors = {
    'b_a': np.array([9000, -13000, 8000]) * micro_g_to_meters_per_second_squared,  # Accelerometer biases (m/s^2)
    'b_g': np.array([-180, 260, -160]) * deg_to_rad / 3600,  # Gyro biases (rad/s)
    'M_a': np.array([[50000, -15000, 10000], [-7500, -60000, 12500], [-12500, 5000, 20000]]) * 1e-6,  # Accel scale factor/cross coupling
    'M_g': np.array([[40000, -14000, 12500], [0, -30000, -7500], [0, 0, -17500]]) * 1e-6,  # Gyro scale factor/cross coupling
    'G_g': np.array([[90, -110, -60], [-50, 190, -160], [30, 110, -130]]) * deg_to_rad / (3600 * 9.80665),  # Gyro g-dependent biases
    'accel_noise_root_PSD': 1000 * micro_g_to_meters_per_second_squared,  # Accel noise root PSD (m s^-1.5)
    'gyro_noise_root_PSD': 1 * deg_to_rad / 60,  # Gyro noise root PSD (rad s^-0.5)
    'accel_quant_level': 1e-1,  # Accel quantization level (m/s^2)
    'gyro_quant_level': 2e-3  # Gyro quantization level (rad/s)
}

# GNSS configuration
GNSS_config = {
    'epoch_interval': 0.5,  # Interval between GNSS epochs (s)
    'init_est_r_ea_e': np.array([0, 0, 0]),  # Initial estimated position (m; ECEF)
    'no_sat': 30,  # Number of satellites
    'r_os': 2.656175e7,  # Orbital radius of satellites (m)
    'inclination': 55,  # Inclination angle of satellites (deg)
    'const_delta_lambda': 0,  # Longitude offset of constellation (deg)
    'const_delta_t': 0,  # Timing offset of constellation (s)
    'mask_angle': 10,  # Mask angle (deg)
    'SIS_err_SD': 1,  # Signal in space error SD (m)
    'zenith_iono_err_SD': 2,  # Zenith ionosphere error SD (m)
    'zenith_trop_err_SD': 0.2,  # Zenith troposphere error SD (m)
    'code_track_err_SD': 1,  # Code tracking error SD (m)
    'rate_track_err_SD': 0.02,  # Range rate tracking error SD (m/s)
    'rx_clock_offset': 10000,  # Receiver clock offset at t=0 (m)
    'rx_clock_drift': 100  # Receiver clock drift at t=0 (m/s)
}

# Tightly coupled Kalman filter configuration
TC_KF_config = {
    'init_att_unc': np.deg2rad(2),  # Initial attitude uncertainty per axis (rad)
    'init_vel_unc': 0.1,  # Initial velocity uncertainty per axis (m/s)
    'init_pos_unc': 10,  # Initial position uncertainty per axis (m)
    'init_b_a_unc': 10000 * micro_g_to_meters_per_second_squared,  # Initial accel bias uncertainty (m/s^2)
    'init_b_g_unc': 200 * deg_to_rad / 3600,  # Initial gyro bias uncertainty (rad/s)
    'init_clock_offset_unc': 10,  # Initial clock offset uncertainty (m)
    'init_clock_drift_unc': 0.1,  # Initial clock drift uncertainty (m/s)
    'gyro_noise_PSD': 0.01**2,  # Gyro noise PSD (rad^2/s)
    'accel_noise_PSD': 0.2**2,  # Accel noise PSD (m^2 s^-3)
    'accel_bias_PSD': 1.0e-5,  # Accel bias random walk PSD (m^2 s^-5)
    'gyro_bias_PSD': 4.0e-11,  # Gyro bias random walk PSD (rad^2 s^-3)
    'clock_freq_PSD': 1,  # Receiver clock frequency-drift PSD (m^2/s^3)
    'clock_phase_PSD': 1,  # Receiver clock phase-drift PSD (m^2/s)
    'pseudo_range_SD': 2.5,  # Pseudo-range measurement noise SD (m)
    'range_rate_SD': 0.1  # Pseudo-range rate measurement noise SD (m/s)
}

# Seed random number generator for reproducibility
np.random.seed(1)

# Read input truth motion profile
in_profile, no_epochs, ok = Read_profile(input_profile_name)
if not ok:
    raise FileNotFoundError(f"Error reading {input_profile_name}")

# Tightly coupled INS/GNSS simulation
out_profile, out_errors, out_IMU_bias_est, out_clock, out_KF_SD = Tightly_coupled_INS_GNSS(
    in_profile, no_epochs, initialization_errors, IMU_errors, GNSS_config, TC_KF_config)

# Plot profiles and errors (placeholder)
Plot_profile(in_profile)
Plot_errors(out_errors)
plt.show()

# Write output profile and errors
Write_profile(output_profile_name, out_profile)
Write_errors(output_errors_name, out_errors)
