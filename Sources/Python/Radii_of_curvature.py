"""
Radii_of_curvature.py - Calculates the meridian and transverse radii of curvature

Software for use with "Principles of GNSS, Inertial, and Multisensor
Integrated Navigation Systems," Second Edition.

This function created 31/3/2012 by Paul Groves
Converted to Python 2025

Copyright 2012, Paul Groves
License: BSD; see license.txt for details
"""

import numpy as np
from typing import Tuple


def Radii_of_curvature(L: float) -> Tuple[float, float]:
    """
    Calculates the meridian and transverse radii of curvature.
    
    Args:
        L (float): geodetic latitude (rad)
        
    Returns:
        tuple: (R_N, R_E) - meridian and transverse radii of curvature (m)
    """
    
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    e = 0.0818191908425  # WGS84 eccentricity
    
    # Calculate meridian radius of curvature using (2.105)
    temp = 1 - (e * np.sin(L))**2
    R_N = R_0 * (1 - e**2) / temp**1.5
    
    # Calculate transverse radius of curvature using (2.105)
    R_E = R_0 / np.sqrt(temp)
    
    return R_N, R_E


if __name__ == "__main__":
    # Example usage and testing
    import numpy as np
    
    # Test at various latitudes
    test_latitudes = [0, 30, 45, 60, 90]  # degrees
    
    print("Latitude (deg) | R_N (m)      | R_E (m)")
    print("---------------|--------------|------------")
    
    for lat_deg in test_latitudes:
        lat_rad = np.radians(lat_deg)
        R_N, R_E = radii_of_curvature(lat_rad)
        print(f"{lat_deg:13.0f} | {R_N:12.0f} | {R_E:10.0f}")
    
    # Test at San Francisco coordinates
    sf_lat = np.radians(37.4)
    R_N, R_E = radii_of_curvature(sf_lat)
    print(f"\nSan Francisco (37.4°N):")
    print(f"Meridian radius: {R_N:.0f} m")
    print(f"Transverse radius: {R_E:.0f} m")