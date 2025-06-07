"""
Write_profile.py - outputs a motion profile in the following .csv format
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


def Write_profile(filename: str, out_profile: np.ndarray) -> None:
    """
    Write motion profile to CSV file.
    
    Args:
        filename (str): Name of file to write
        out_profile (np.ndarray): Array of data to write
    """
    # Parameters
    deg_to_rad = 0.01745329252
    rad_to_deg = 1 / deg_to_rad
    
    # Make a copy to avoid modifying the original
    output_data = out_profile.copy()
    
    # Convert output profile from radians to degrees
    output_data[:, 1:3] = rad_to_deg * output_data[:, 1:3]  # lat, lon
    output_data[:, 7:10] = rad_to_deg * output_data[:, 7:10]  # roll, pitch, yaw
    
    try:
        # Write output profile with high precision (matching MATLAB's precision=12)
        np.savetxt(filename, output_data, delimiter=',', fmt='%.12f')
        print(f"Profile written to {filename}")
    except Exception as e:
        print(f"Error writing file {filename}: {e}")


if __name__ == "__main__":
    # Example usage and testing
    # Create test data
    test_data = np.array([
        [0.0, np.radians(37.4), np.radians(-122.1), 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [1.0, np.radians(37.4001), np.radians(-122.1001), 100.0, 1.0, 1.0, 0.0, np.radians(1.0), np.radians(1.0), np.radians(1.0)]
    ])
    
    write_profile("test_output.csv", test_data)
    print("Test data written to test_output.csv")