"""
Velocity_from_curvilinear.py - updates velocity by differentiating
latitude, longitude, and height

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


def velocity_from_curvilinear(tor_i: float, old_L_b: float, old_lambda_b: float,
                             old_h_b: float, old_v_eb_n: np.ndarray,
                             L_b: float, lambda_b: float, h_b: float) -> np.ndarray:
    """
    Updates velocity by differentiating latitude, longitude, and height.
    
    Args:
        tor_i (float): time interval between epochs (s)
        old_L_b (float): previous latitude (rad)
        old_lambda_b (float): previous longitude (rad)
        old_h_b (float): previous height (m)
        old_v_eb_n (np.ndarray): previous velocity of body w.r.t earth, resolved about 
                                north, east, and down (m/s)
        L_b (float): current latitude (rad)
        lambda_b (float): current longitude (rad)
        h_b (float): current height (m)
        
    Returns:
        np.ndarray: current velocity of body w.r.t earth, resolved about 
                   north, east, and down (m/s)
    """
    
    # Calculate meridian and transverse radii of curvature
    old_R_N, old_R_E = radii_of_curvature(old_L_b)
    R_N, R_E = radii_of_curvature(L_b)
    
    # Differentiate latitude, longitude, and height
    lat_rate = (L_b - old_L_b) / tor_i
    long_rate = (lambda_b - old_lambda_b) / tor_i
    ht_rate = (h_b - old_h_b) / tor_i
    
    # Initialize velocity vector
    v_eb_n = np.zeros(3)
    
    # Derive the current velocity using (5.56)
    v_eb_n[0] = (old_R_N + h_b) * (2 * lat_rate - old_v_eb_n[0] / (old_R_N + old_h_b))
    
    v_eb_n[1] = ((R_E + h_b) * np.cos(L_b)) * (2 * long_rate - old_v_eb_n[1] / 
                                               ((old_R_E + old_h_b) * np.cos(old_L_b)))
    
    v_eb_n[2] = -2 * ht_rate - old_v_eb_n[2]
    
    return v_eb_n


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    # Test parameters
    tor_i = 1.0  # 1 second
    old_L_b = np.radians(37.4)  # San Francisco latitude
    old_lambda_b = np.radians(-122.1)  # San Francisco longitude
    old_h_b = 100.0  # 100m height
    old_v_eb_n = np.array([1.0, 1.0, 0.0])  # 1 m/s north, 1 m/s east, 0 m/s down
    
    # New position (slightly moved)
    L_b = np.radians(37.4001)  # Slightly north
    lambda_b = np.radians(-122.0999)  # Slightly east
    h_b = 101.0  # Slightly up
    
    # Calculate velocity from position change
    v_eb_n = velocity_from_curvilinear(tor_i, old_L_b, old_lambda_b, old_h_b, 
                                      old_v_eb_n, L_b, lambda_b, h_b)
    
    print(f"Old position: Lat={np.degrees(old_L_b):.6f}°, Lon={np.degrees(old_lambda_b):.6f}°, Height={old_h_b:.1f}m")
    print(f"New position: Lat={np.degrees(L_b):.6f}°, Lon={np.degrees(lambda_b):.6f}°, Height={h_b:.1f}m")
    print(f"Old velocity: North={old_v_eb_n[0]:.3f} m/s, East={old_v_eb_n[1]:.3f} m/s, Down={old_v_eb_n[2]:.3f} m/s")
    print(f"New velocity: North={v_eb_n[0]:.3f} m/s, East={v_eb_n[1]:.3f} m/s, Down={v_eb_n[2]:.3f} m/s")