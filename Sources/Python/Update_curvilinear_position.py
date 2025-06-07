"""
Update_curvilinear_position.py - Updates latitude, longitude, and height by
integrating the velocity

Software for use with "Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems," Second Edition.

This function created 31/3/2012 by Paul Groves
Converted to Python 2025

Copyright 2012, Paul Groves
License: BSD; see license.txt for details
"""

import numpy as np
from typing import Tuple
from Radii_of_curvature import radii_of_curvature


def update_curvilinear_position(tor_i: float, old_L_b: float, old_lambda_b: float, 
                               old_h_b: float, old_v_eb_n: np.ndarray, 
                               v_eb_n: np.ndarray) -> Tuple[float, float, float]:
    """
    Updates latitude, longitude, and height by integrating the velocity.
    
    Args:
        tor_i (float): time interval between epochs (s)
        old_L_b (float): previous latitude (rad)
        old_lambda_b (float): previous longitude (rad)
        old_h_b (float): previous height (m)
        old_v_eb_n (np.ndarray): previous velocity of body w.r.t earth, resolved about 
                                north, east, and down (m/s)
        v_eb_n (np.ndarray): current velocity of body w.r.t earth, resolved about 
                            north, east, and down (m/s)
    
    Returns:
        tuple: (L_b, lambda_b, h_b) - current latitude (rad), longitude (rad), height (m)
    """
    
    # Calculate meridian and transverse radii of curvature
    old_R_N, old_R_E = radii_of_curvature(old_L_b)
    
    # Update height using (5.56)
    h_b = old_h_b - 0.5 * tor_i * (old_v_eb_n[2] + v_eb_n[2])
    
    # Update latitude using (5.56)
    L_b = old_L_b + 0.5 * tor_i * (old_v_eb_n[0] / (old_R_N + old_h_b) +
                                   v_eb_n[0] / (old_R_N + h_b))
    
    # Calculate meridian and transverse radii of curvature
    R_N, R_E = radii_of_curvature(L_b)
    
    # Update longitude using (5.56)
    lambda_b = old_lambda_b + 0.5 * tor_i * (old_v_eb_n[1] / ((old_R_E + old_h_b) * np.cos(old_L_b)) +
                                            v_eb_n[1] / ((R_E + h_b) * np.cos(L_b)))
    
    return L_b, lambda_b, h_b


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    # Test parameters
    tor_i = 1.0  # 1 second
    old_L_b = np.radians(37.4)  # San Francisco latitude
    old_lambda_b = np.radians(-122.1)  # San Francisco longitude
    old_h_b = 100.0  # 100m height
    old_v_eb_n = np.array([1.0, 1.0, 0.0])  # 1 m/s north, 1 m/s east, 0 m/s down
    v_eb_n = np.array([1.0, 1.0, 0.0])  # same velocity
    
    # Update position
    L_b, lambda_b, h_b = update_curvilinear_position(tor_i, old_L_b, old_lambda_b, old_h_b, old_v_eb_n, v_eb_n)
    
    print(f"Old position: Lat={np.degrees(old_L_b):.6f}°, Lon={np.degrees(old_lambda_b):.6f}°, Height={old_h_b:.1f}m")
    print(f"New position: Lat={np.degrees(L_b):.6f}°, Lon={np.degrees(lambda_b):.6f}°, Height={h_b:.1f}m")