import numpy as np
import pandas as pd
from Read_profile import Read_profile
from Write_profile import Write_profile

def Interpolate_profile(input_profile_name, output_profile_name):
    """
    Interpolates .csv motion profiles by adding intermediate points with jitter.
    
    Args:
        input_profile_name (str): Input motion profile filename
        output_profile_name (str): Output motion profile filename
    """
    # Configuration
    VELOCITY_JITTER = 0.001  # Jitter standard deviation on interpolated velocity (m/s)
    ATTITUDE_JITTER = 0.00002  # Jitter standard deviation on interpolated attitude (rad)
    
    # Seed random number generator for reproducibility
    np.random.seed(1)
    
    # Input truth motion profile from .csv format file
    try:
        in_profile = Read_profile(input_profile_name)  # Assumes Read_profile returns a numpy array
        no_epochs = in_profile.shape[0]
    except Exception as e:
        print(f"Error reading input profile: {e}")
        return
    
    # Initialize true navigation solution
    old_time = in_profile[0, 0]
    old_true_L_b = in_profile[0, 1]
    old_true_lambda_b = in_profile[0, 2]
    old_true_h_b = in_profile[0, 3]
    old_true_v_eb_n = in_profile[0, 4:7].reshape(-1, 1)
    old_true_eul_nb = in_profile[0, 7:10].reshape(-1, 1)
    
    # Initialize output profile (twice the size to accommodate interpolated points)
    out_profile = np.zeros((2 * no_epochs - 1, 10))
    out_profile[0, :] = in_profile[0, :]
    
    # Main loop
    for epoch in range(1, no_epochs):
        # Input data from motion profile
        time = in_profile[epoch, 0]
        true_L_b = in_profile[epoch, 1]
        true_lambda_b = in_profile[epoch, 2]
        true_h_b = in_profile[epoch, 3]
        true_v_eb_n = in_profile[epoch, 4:7].reshape(-1, 1)
        true_eul_nb = in_profile[epoch, 7:10].reshape(-1, 1)
        
        # Time interval
        tor_i = time - old_time
        
        # Interpolate
        inter_L_b = 0.5 * (true_L_b + old_true_L_b)
        inter_lambda_b = 0.5 * (true_lambda_b + old_true_lambda_b)
        inter_h_b = 0.5 * (true_h_b + old_true_h_b)
        inter_v_eb_n = 0.5 * (true_v_eb_n + old_true_v_eb_n) + np.random.randn(3, 1) * VELOCITY_JITTER
        inter_eul_nb = 0.5 * (true_eul_nb + old_true_eul_nb) + np.random.randn(3, 1) * ATTITUDE_JITTER
        
        # Generate output profile records
        epoch1 = 2 * epoch - 2
        epoch2 = 2 * epoch - 1
        out_profile[epoch1, 0] = 0.5 * (time + old_time)
        out_profile[epoch1, 1] = inter_L_b
        out_profile[epoch1, 2] = inter_lambda_b
        out_profile[epoch1, 3] = inter_h_b
        out_profile[epoch1, 4:7] = inter_v_eb_n.T
        out_profile[epoch1, 7:10] = inter_eul_nb.T
        out_profile[epoch2, 0] = time
        out_profile[epoch2, 1] = true_L_b
        out_profile[epoch2, 2] = true_lambda_b
        out_profile[epoch2, 3] = true_h_b
        out_profile[epoch2, 4:7] = true_v_eb_n.T
        out_profile[epoch2, 7:10] = true_eul_nb.T
        
        # Reset old values
        old_time = time
        old_true_L_b = true_L_b
        old_true_lambda_b = true_lambda_b
        old_true_h_b = true_h_b
        old_true_v_eb_n = true_v_eb_n
        old_true_eul_nb = true_eul_nb
    
    # Write output profile
    Write_profile(output_profile_name, out_profile)