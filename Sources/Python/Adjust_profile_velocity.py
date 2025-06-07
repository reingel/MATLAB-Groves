"""
Adjust_profile_velocity.py - Adjusts the velocity in a motion profile file to
make it consistent with the position

Software for use with "Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems," Second Edition.

This function created 2/4/2012 by Paul Groves
Converted to Python 2025

Copyright 2012, Paul Groves
License: BSD; see license.txt for details
"""

import numpy as np
from Read_profile import read_profile
from Write_profile import write_profile
from Velocity_from_curvilinear import velocity_from_curvilinear


def adjust_profile_velocity(in_filename: str, out_filename: str) -> None:
    """
    Adjusts the velocity in a motion profile file to make it consistent with the position.
    
    Args:
        in_filename (str): Name of input file, e.g. 'In_profile.csv'
        out_filename (str): Name of output file, e.g. 'Out_profile.csv'
    """
    
    # Parameters
    deg_to_rad = 0.01745329252
    rad_to_deg = 1 / deg_to_rad
    
    # Input truth motion profile from .csv format file
    in_profile, no_epochs, ok = read_profile(in_filename)
    
    # End script if there is a problem with the file
    if not ok:
        return
    
    # Initialize navigation solution
    old_time = in_profile[0, 0]
    old_L_b = in_profile[0, 1]
    old_lambda_b = in_profile[0, 2]
    old_h_b = in_profile[0, 3]
    old_v_eb_n = in_profile[0, 4:7]
    old_eul_nb = in_profile[0, 7:10]
    
    # Initialize output profile with first row
    out_profile = np.zeros_like(in_profile)
    out_profile[0, :] = in_profile[0, :]
    
    # Main loop
    for epoch in range(1, no_epochs):
        # Input data from profile
        time = in_profile[epoch, 0]
        L_b = in_profile[epoch, 1]
        lambda_b = in_profile[epoch, 2]
        h_b = in_profile[epoch, 3]
        eul_nb = in_profile[epoch, 7:10]
        
        # Time interval
        tor_i = time - old_time
        
        # Update velocity
        v_eb_n = velocity_from_curvilinear(tor_i, old_L_b, old_lambda_b, old_h_b,
                                          old_v_eb_n, L_b, lambda_b, h_b)
        
        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = L_b
        out_profile[epoch, 2] = lambda_b
        out_profile[epoch, 3] = h_b
        out_profile[epoch, 4:7] = v_eb_n
        out_profile[epoch, 7:10] = eul_nb
        
        # Reset old values
        old_time = time
        old_L_b = L_b
        old_lambda_b = lambda_b
        old_h_b = h_b
        old_v_eb_n = v_eb_n
        old_eul_nb = eul_nb
    
    # Write output profile
    write_profile(out_filename, out_profile)


if __name__ == "__main__":
    # Example usage
    input_file = "In_profile.csv"
    output_file = "Out_profile.csv"
    adjust_profile_velocity(input_file, output_file)