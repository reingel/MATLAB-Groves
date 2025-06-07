import numpy as np

def Satellite_positions_and_velocities(time, GNSS_config):
    """
    Satellite_positions_and_velocities - Returns ECEF Cartesian positions and
    ECEF velocities of all satellites in the constellation. Simple circular
    orbits with regularly distributed satellites are modeled.

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 11/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        time (float): Current simulation time (s)
        GNSS_config (dict): Configuration dictionary with keys:
            - no_sat (int): Number of satellites in constellation
            - r_os (float): Orbital radius of satellites (m)
            - inclination (float): Inclination angle of satellites (deg)
            - const_delta_lambda (float): Longitude offset of constellation (deg)
            - const_delta_t (float): Timing offset of constellation (s)

    Returns:
        tuple: (sat_r_es_e, sat_v_es_e)
            - sat_r_es_e (numpy.ndarray): ECEF satellite position, shape (no_sat, 3)
            - sat_v_es_e (numpy.ndarray): ECEF satellite velocity, shape (no_sat, 3)
    """
    # Constants
    mu = 3.986004418E14  # WGS84 Earth gravitational constant (m^3 s^-2)
    omega_ie = 7.292115E-5  # Earth rotation rate in rad/s

    # Convert inclination angle to radians
    inclination = np.deg2rad(GNSS_config['inclination'])

    # Determine orbital angular rate using (8.8)
    omega_is = np.sqrt(mu / GNSS_config['r_os']**3)

    # Determine constellation time
    const_time = time + GNSS_config['const_delta_t']

    # Initialize output arrays
    sat_r_es_e = np.zeros((GNSS_config['no_sat'], 3))
    sat_v_es_e = np.zeros((GNSS_config['no_sat'], 3))

    # Loop satellites
    for j in range(GNSS_config['no_sat']):
        # (Corrected) argument of latitude
        u_os_o = 2 * np.pi * (j) / GNSS_config['no_sat'] + omega_is * const_time

        # Satellite position in the orbital frame from (8.14)
        r_os_o = GNSS_config['r_os'] * np.array([np.cos(u_os_o), np.sin(u_os_o), 0])

        # Longitude of the ascending node from (8.16)
        Omega = (np.pi * (j % 6) / 3 + np.deg2rad(GNSS_config['const_delta_lambda'])) - omega_ie * const_time

        # ECEF Satellite Position from (8.19)
        sat_r_es_e[j, :] = [
            r_os_o[0] * np.cos(Omega) - r_os_o[1] * np.cos(inclination) * np.sin(Omega),
            r_os_o[0] * np.sin(Omega) + r_os_o[1] * np.cos(inclination) * np.cos(Omega),
            r_os_o[1] * np.sin(inclination)
        ]

        # Satellite velocity in the orbital frame from (8.25)
        v_os_o = GNSS_config['r_os'] * omega_is * np.array([-np.sin(u_os_o), np.cos(u_os_o), 0])

        # ECEF Satellite velocity from (8.26)
        sat_v_es_e[j, :] = [
            v_os_o[0] * np.cos(Omega) - v_os_o[1] * np.cos(inclination) * np.sin(Omega) + omega_ie * sat_r_es_e[j, 1],
            v_os_o[0] * np.sin(Omega) + v_os_o[1] * np.cos(inclination) * np.cos(Omega) - omega_ie * sat_r_es_e[j, 0],
            v_os_o[1] * np.sin(inclination)
        ]

    return sat_r_es_e, sat_v_es_e