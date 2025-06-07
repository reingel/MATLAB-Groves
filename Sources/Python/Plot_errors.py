import numpy as np
import matplotlib.pyplot as plt

def Plot_errors(errors):
    """
    Plot_errors - Plots navigation solution errors

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 3/4/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        errors (numpy.ndarray): Array of error data to plot, shape (N, 10)
            - Column 1: time (sec)
            - Column 2: north position error (m)
            - Column 3: east position error (m)
            - Column 4: down position error (m)
            - Column 5: north velocity (m/s)
            - Column 6: east velocity (m/s)
            - Column 7: down velocity (m/s)
            - Column 8: roll component of NED attitude error (rad)
            - Column 9: pitch component of NED attitude error (rad)
            - Column 10: yaw component of NED attitude error (rad)
    """
    fig = plt.figure(figsize=(12, 8))
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.08, top=0.92, wspace=0.08, hspace=0.16)

    # North position error
    ax1 = fig.add_subplot(331)
    ax1.plot(errors[:, 0], errors[:, 1], linewidth=1.5, color=[0.9, 0.45, 0])
    ax1.set_title('North position error, m')
    ax1.set_position([0.1, 0.68, 0.27, 0.25])

    # East position error
    ax2 = fig.add_subplot(332)
    ax2.plot(errors[:, 0], errors[:, 2], linewidth=1.5, color=[0, 0.9, 0.45])
    ax2.set_title('East position error, m')
    ax2.set_position([0.38, 0.68, 0.27, 0.25])

    # Down position error
    ax3 = fig.add_subplot(333)
    ax3.plot(errors[:, 0], errors[:, 3], linewidth=1.5, color=[0.45, 0, 0.9])
    ax3.set_title('Down position error, m')
    ax3.set_position([0.66, 0.68, 0.27, 0.25])

    # North velocity error
    ax4 = fig.add_subplot(334)
    ax4.plot(errors[:, 0], errors[:, 4], linewidth=1.5, color=[0.9, 0, 0.45])
    ax4.set_title('North velocity error, m/s')
    ax4.set_position([0.1, 0.38, 0.27, 0.25])

    # East velocity error
    ax5 = fig.add_subplot(335)
    ax5.plot(errors[:, 0], errors[:, 5], linewidth=1.5, color=[0.45, 0.9, 0])
    ax5.set_title('East velocity error, m/s')
    ax5.set_position([0.38, 0.38, 0.27, 0.25])

    # Down velocity error
    ax6 = fig.add_subplot(336)
    ax6.plot(errors[:, 0], errors[:, 6], linewidth=1.5, color=[0, 0.45, 0.9])
    ax6.set_title('Down velocity error, m/s')
    ax6.set_position([0.66, 0.38, 0.27, 0.25])

    # Attitude error about North
    ax7 = fig.add_subplot(337)
    ax7.plot(errors[:, 0], np.rad2deg(errors[:, 7]), linewidth=1.5, color=[0, 0.7, 0.7])
    ax7.set_xlabel('Time, s')
    ax7.set_title('Attitude error about North, deg')
    ax7.set_position([0.1, 0.08, 0.27, 0.25])

    # Attitude error about East
    ax8 = fig.add_subplot(338)
    ax8.plot(errors[:, 0], np.rad2deg(errors[:, 8]), linewidth=1.5, color=[0.7, 0, 0.7])
    ax8.set_xlabel('Time, s')
    ax8.set_title('Attitude error about East, deg')
    ax8.set_position([0.38, 0.08, 0.27, 0.25])

    # Heading error
    ax9 = fig.add_subplot(339)
    ax9.plot(errors[:, 0], np.rad2deg(errors[:, 9]), linewidth=1.5, color=[0.7, 0.7, 0])
    ax9.set_xlabel('Time, s')
    ax9.set_title('Heading error, deg')
    ax9.set_position([0.66, 0.08, 0.27, 0.25])

    plt.tight_layout()
