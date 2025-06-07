import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Sources/Python'))
import numpy as np
import matplotlib.pyplot as plt
from Read_profile import Read_profile
from Inertial_navigation_NED import Inertial_navigation_NED
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
output_profile_name = os.path.join(output_profile_folder, 'Inertial_Demo_3_Profile.csv')
output_errors_name = os.path.join(output_profile_folder, 'Inertial_Demo_3_Errors.csv')

# Initialization errors
initialization_errors = {
    'delta_r_eb_n': np.array([4, 2, 3]),  # Position error (m; N,E,D)
    'delta_v_eb_n': np.array([0.05, -0.05, 0.1]),  # Velocity error (m/s; N,E,D)
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

# Seed random number generator for reproducibility
np.random.seed(1)

# Read input truth motion profile
in_profile, no_epochs, ok = Read_profile(input_profile_name)
if not ok:
    raise FileNotFoundError(f"Error reading {input_profile_name}")

# NED Inertial navigation simulation
out_profile, out_errors = Inertial_navigation_NED(in_profile, no_epochs, initialization_errors, IMU_errors)

# Plot profiles and errors (placeholder)
Plot_profile(in_profile)
Plot_errors(out_errors)
plt.show()

# Write output profile and errors
Write_profile(output_profile_name, out_profile)
Write_errors(output_errors_name, out_errors)
