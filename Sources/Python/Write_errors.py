import numpy as np
import csv

def Write_errors(filename, out_errors):
    """
    Write_errors - Outputs the errors in the following .csv format
        - Column 1: time (sec)
        - Column 2: north position error (m)
        - Column 3: east position error (m)
        - Column 4: down position error (m)
        - Column 5: north velocity (m/s)
        - Column 6: east velocity (m/s)
        - Column 7: down velocity (m/s)
        - Column 8: roll component of NED attitude error (deg)
        - Column 9: pitch component of NED attitude error (deg)
        - Column 10: yaw component of NED attitude error (deg)

    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.

    This function created 31/3/2012 by Paul Groves, converted to Python 6/6/2025

    Parameters:
        filename (str): Name of file to write
        out_errors (numpy.ndarray): Array of data to write, shape (N, 10)
    """
    # Parameters
    rad_to_deg = 180 / np.pi

    # Convert attitude errors from radians to degrees
    out_errors_converted = out_errors.copy()
    out_errors_converted[:, 7:10] = rad_to_deg * out_errors[:, 7:10]

    # Write output profile to CSV
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(np.round(out_errors_converted, 12))