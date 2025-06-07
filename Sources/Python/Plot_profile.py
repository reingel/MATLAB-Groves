import numpy as np
import matplotlib.pyplot as plt
from Radii_of_curvature import Radii_of_curvature

def Plot_profile(profile):
    """
    Plot_profile - Plots a motion profile

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        profile (numpy.ndarray): Array of motion profile data to plot, shape (N, 10)
            - Column 1: time (sec)
            - Column 2: latitude (deg)
            - Column 3: longitude (deg)
            - Column 4: height (m)
            - Column 5: north velocity (m/s)
            - Column 6: east velocity (m/s)
            - Column 7: down velocity (m/s)
            - Column 8: roll angle of body w.r.t NED (deg)
            - Column 9: pitch angle of body w.r.t NED (deg)
            - Column 10: yaw angle of body w.r.t NED (deg)
    """
    rad_to_deg = 57.2957795131  # Conversion factor from radians to degrees

    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, wspace=0.08, hspace=0.16)

    R_N, R_E = Radii_of_curvature(profile[0, 1])

    # North displacement
    ax1 = fig.add_subplot(331)
    ax1.plot(profile[:, 0], (profile[:, 1] - profile[0, 1]) * (R_N + profile[0, 3]), linewidth=1.5, color=[0.9, 0.45, 0])
    ax1.set_title('North displacement, m')
    ax1.set_position([0.1, 0.68, 0.27, 0.25])

    # East displacement
    ax2 = fig.add_subplot(332)
    ax2.plot(profile[:, 0], (profile[:, 2] - profile[0, 2]) * (R_N + profile[0, 3]) * np.cos(np.deg2rad(profile[0, 1])), linewidth=1.5, color=[0, 0.9, 0.45])
    ax2.set_title('East displacement, m')
    ax2.set_position([0.38, 0.68, 0.27, 0.25])

    # Down displacement
    ax3 = fig.add_subplot(333)
    ax3.plot(profile[:, 0], (profile[0, 3] - profile[:, 3]), linewidth=1.5, color=[0.45, 0, 0.9])
    ax3.set_title('Down displacement, m')
    ax3.set_position([0.66, 0.68, 0.27, 0.25])

    # North velocity
    ax4 = fig.add_subplot(334)
    ax4.plot(profile[:, 0], profile[:, 4], linewidth=1.5, color=[0.9, 0, 0.45])
    ax4.set_title('North velocity, m/s')
    ax4.set_position([0.1, 0.38, 0.27, 0.25])

    # East velocity
    ax5 = fig.add_subplot(335)
    ax5.plot(profile[:, 0], profile[:, 5], linewidth=1.5, color=[0.45, 0.9, 0])
    ax5.set_title('East velocity, m/s')
    ax5.set_position([0.38, 0.38, 0.27, 0.25])

    # Down velocity
    ax6 = fig.add_subplot(336)
    ax6.plot(profile[:, 0], profile[:, 6], linewidth=1.5, color=[0, 0.45, 0.9])
    ax6.set_title('Down velocity, m/s')
    ax6.set_position([0.66, 0.38, 0.27, 0.25])

    # Bank (roll)
    ax7 = fig.add_subplot(337)
    ax7.plot(profile[:, 0], profile[:, 7] * rad_to_deg, linewidth=1.5, color=[0, 0.7, 0.7])
    ax7.set_xlabel('Time, s')
    ax7.set_title('Bank, deg')
    ax7.set_position([0.1, 0.08, 0.27, 0.25])

    # Elevation (pitch)
    ax8 = fig.add_subplot(338)
    ax8.plot(profile[:, 0], profile[:, 8] * rad_to_deg, linewidth=1.5, color=[0.7, 0, 0.7])
    ax8.set_xlabel('Time, s')
    ax8.set_title('Elevation, deg')
    ax8.set_position([0.38, 0.08, 0.27, 0.25])

    # Heading (yaw)
    ax9 = fig.add_subplot(339)
    ax9.plot(profile[:, 0], profile[:, 9] * rad_to_deg, linewidth=1.5, color=[0.7, 0.7, 0])
    ax9.set_xlabel('Time, s')
    ax9.set_title('Heading, deg')
    ax9.set_position([0.66, 0.08, 0.27, 0.25])

    plt.tight_layout()
