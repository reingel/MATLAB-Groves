import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Sources/Python'))
import numpy as np
import matplotlib.pyplot as plt
from Plot_profile import Plot_profile
from Plot_errors import Plot_errors
from GNSS_Kalman_Filter import GNSS_Kalman_Filter
from Write_errors import Write_errors
from Read_profile import Read_profile
from Write_profile import Write_profile

# GNSS_Demo_4
# SCRIPT Stand-alone GNSS demo with Kalman filter solution:
#   Profile_2 (175s car)
#
# Software for use with "Principles of GNSS, Inertial, and Multisensor
# Integrated Navigation Systems," Second Edition.
#
# Created 27/5/12 by Paul Groves, converted to Python 6/6/2025

# Constants
deg_to_rad = 0.01745329252
rad_to_deg = 1 / deg_to_rad
micro_g_to_meters_per_second_squared = 9.80665E-6

# CONFIGURATION
input_profile_folder = os.path.join(os.path.dirname(__file__), '..', 'Profiles/Input')
output_profile_folder = os.path.join(os.path.dirname(__file__), '..', 'Profiles/Output')
# Input truth motion profile filename
input_profile_name = os.path.join(input_profile_folder, 'Profile_2.csv')
# Output motion profile and error filenames
output_profile_name = os.path.join(output_profile_folder, 'GNSS_Demo_4_Profile.csv')
output_errors_name = os.path.join(output_profile_folder, 'GNSS_Demo_4_Errors.csv')

# Interval between GNSS epochs (s)
GNSS_config = {
    'epoch_interval': 1,
    # Initial estimated position (m; ECEF)
    'init_est_r_ea_e': np.array([0, 0, 0]),
    # Number of satellites in constellation
    'no_sat': 30,
    # Orbital radius of satellites (m)
    'r_os': 2.656175E7,
    # Inclination angle of satellites (deg)
    'inclination': 55,
    # Longitude offset of constellation (deg)
    'const_delta_lambda': 0,
    # Timing offset of constellation (s)
    'const_delta_t': 0,
    # Mask angle (deg)
    'mask_angle': 10,
    # Signal in space error SD (m)
    'SIS_err_SD': 1,
    # Zenith ionosphere error SD (m)
    'zenith_iono_err_SD': 2,
    # Zenith troposphere error SD (m)
    'zenith_trop_err_SD': 0.2,
    # Code tracking error SD (m)
    'code_track_err_SD': 1,
    # Range rate tracking error SD (m/s)
    'rate_track_err_SD': 0.02,
    # Receiver clock offset at time=0 (m)
    'rx_clock_offset': 10000,
    # Receiver clock drift at time=0 (m/s)
    'rx_clock_drift': 100
}

# Kalman filter configuration
GNSS_KF_config = {
    # Initial position uncertainty per axis (m)
    'init_pos_unc': 10,
    # Initial velocity uncertainty per axis (m/s)
    'init_vel_unc': 0.1,
    # Initial clock offset uncertainty per axis (m)
    'init_clock_offset_unc': 10,
    # Initial clock drift uncertainty per axis (m/s)
    'init_clock_drift_unc': 0.1,
    # Acceleration PSD per axis (m^2/s^3)
    'accel_PSD': 10,
    # Receiver clock frequency-drift PSD (m^2/s^3)
    'clock_freq_PSD': 1,
    # Receiver clock phase-drift PSD (m^2/s)
    'clock_phase_PSD': 1,
    # Pseudo-range measurement noise SD (m)
    'pseudo_range_SD': 2.5,
    # Pseudo-range rate measurement noise SD (m/s)
    'range_rate_SD': 0.05
}

# Seeding of the random number generator for reproducibility
np.random.seed(1)

# Input truth motion profile from .csv format file
in_profile, no_epochs, ok = Read_profile(input_profile_name)

# End script if there is a problem with the file
if not ok:
    exit()

# NED Inertial navigation simulation
out_profile, out_errors, out_clock, out_KF_SD = GNSS_Kalman_Filter(
    in_profile, no_epochs, GNSS_config, GNSS_KF_config)

# Plot the input motion profile and the errors
plt.close('all')
Plot_profile(in_profile)
Plot_errors(out_errors)
plt.show()

# Write output profile and errors file
Write_profile(output_profile_name, out_profile)
Write_errors(output_errors_name, out_errors)