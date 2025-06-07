import numpy as np

def Gravity_ECEF(r_eb_e):
    """
    Gravity_ECEF - Calculates acceleration due to gravity resolved about
    ECEF-frame

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 1/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        r_eb_e (numpy.ndarray): Cartesian position of body frame w.r.t. ECEF frame,
                                resolved about ECEF-frame axes (m), shape (3,)

    Returns:
        numpy.ndarray: Acceleration due to gravity (m/s^2), shape (3,)
    """
    # Parameters
    R_0 = 6378137  # WGS84 Equatorial radius in meters
    mu = 3.986004418E14  # WGS84 Earth gravitational constant (m^3 s^-2)
    J_2 = 1.082627E-3  # WGS84 Earth's second gravitational constant
    omega_ie = 7.292115E-5  # Earth rotation rate (rad/s)

    # Calculate distance from center of the Earth
    mag_r = np.sqrt(r_eb_e @ r_eb_e)

    # If the input position is 0,0,0, produce a dummy output
    if mag_r == 0:
        g = np.zeros(3)
    else:
        # Calculate gravitational acceleration using (2.142)
        z_scale = 5 * (r_eb_e[2] / mag_r)**2
        gamma = -mu / mag_r**3 * (r_eb_e + 1.5 * J_2 * (R_0 / mag_r)**2 * np.array([
            (1 - z_scale) * r_eb_e[0],
            (1 - z_scale) * r_eb_e[1],
            (3 - z_scale) * r_eb_e[2]
        ]))

        # Add centripetal acceleration using (2.133)
        g = np.zeros(3)
        g[:2] = gamma[:2] + omega_ie**2 * r_eb_e[:2]
        g[2] = gamma[2]

    return g