"""
Read_profile.py - inputs a motion profile in the following .csv format
Column 1: time (sec)
Column 2: latitude (deg)
Column 3: longitude (deg)
Column 4: height (m)
Column 5: north velocity (m/s)
Column 6: east velocity (m/s)
Column 7: down velocity (m/s)
Column 8: roll angle of body w.r.t NED (deg)
Column 9: pitch angle of body w.r.t NED (deg)
Column 10: yaw angle of body w.r.t NED (deg)

Software for use with "Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems," Second Edition.

This function created 31/3/2012 by Paul Groves
Converted to Python 2025

Copyright 2012, Paul Groves
License: BSD; see license.txt for details
"""

import numpy as np
from typing import Tuple


def Read_profile(filename: str) -> Tuple[np.ndarray, int, bool]:
    """
    Read motion profile from CSV file.
    
    Args:
        filename (str): Name of file to read
        
    Returns:
        tuple: (in_profile, no_epochs, ok)
            in_profile: Array of data from the file
            no_epochs: Number of epochs of data in the file
            ok: Indicates file has the expected number of columns
    """
    # Parameters
    deg_to_rad = 0.01745329252
    
    try:
        # Read in the profile in .csv format
        in_profile = np.loadtxt(filename, delimiter=',')
        
        # Handle single row case
        if in_profile.ndim == 1:
            in_profile = in_profile.reshape(1, -1)
        
        # Determine size of file
        no_epochs, no_columns = in_profile.shape
        
        # Check number of columns is correct (otherwise return)
        if no_columns != 10:
            print('Input file has the wrong number of columns')
            ok = False
        else:
            ok = True
            # Convert degrees to radians
            in_profile[:, 1:3] = deg_to_rad * in_profile[:, 1:3]  # lat, lon
            in_profile[:, 7:10] = deg_to_rad * in_profile[:, 7:10]  # roll, pitch, yaw
            
        return in_profile, no_epochs, ok
        
    except Exception as e:
        print(f"Error reading file {filename}: {e}")
        return np.array([]), 0, False

