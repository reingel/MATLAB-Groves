import numpy as np

def Gravity_NED(L_b, h_b):
    """
    Gravity_NED - Calculates acceleration due to gravity resolved about
    north, east, and down

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 2/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        L_b (float): latitude (rad)
        h_b (float): height (m)

    Returns:
        numpy.ndarray: Acceleration due to gravity (m/s^2), shape (3,)
    """
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    R_P = 6356752.31425  # WGS84 Polar radius in meters
    e = 0.0818191908425  # WGS84 eccentricity
    f = 1 / 298.257223563  # WGS84 flattening
    mu = 3.986004418E14  # WGS84 Earth gravitational constant (m^3 s^-2)
    omega_ie = 7.292115E-5  # Earth rotation rate (rad/s)

    # Calculate surface gravity using the Somigliana model, (2.134)
    sinsqL = np.sin(L_b)**2
    g_0 = 9.7803253359 * (1 + 0.001931853 * sinsqL) / np.sqrt(1 - e**2 * sinsqL)

    # Initialize output
    g = np.zeros(3)

    # Calculate north gravity using (2.140)
    g[0] = -8.08E-9 * h_b * np.sin(2 * L_b)

    # East gravity is zero
    g[1] = 0

    # Calculate down gravity using (2.139)
    g[2] = g_0 * (1 - (2 / R_0) * (1 + f * (1 - 2 * sinsqL) +
                  (omega_ie**2 * R_0**2 * R_P / mu)) * h_b + (3 * h_b**2 / R_0**2))

    return g