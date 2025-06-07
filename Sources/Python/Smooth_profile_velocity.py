import numpy as np
from Read_profile import Read_profile
from Write_profile import Write_profile

def Smooth_profile_velocity(in_filename, out_filename):
    """
    Adjusts the velocity in a motion profile file to remove jitter resulting
    from numerical rounding in the position input.

    Args:
        in_filename (str): Name of input file, e.g., 'In_profile.csv'
        out_filename (str): Name of output file, e.g., 'Out_profile.csv'
    """
    # Parameters
    DEG_TO_RAD = 0.01745329252
    RAD_TO_DEG = 1 / DEG_TO_RAD

    # Input truth motion profile from .csv format file
    try:
        in_profile, no_epochs, ok = Read_profile(in_filename)  # Assumes Read_profile returns (array, int, bool)
    except Exception as e:
        print(f"Error reading input profile: {e}")
        return

    # End script if there is a problem with the file
    if not ok:
        return

    # Initialize output profile
    out_profile = np.zeros_like(in_profile)
    out_profile[0, :] = in_profile[0, :]

    # Main loop
    for epoch in range(1, no_epochs - 1):
        # Input data from profile
        time = in_profile[epoch, 0]
        L_b = in_profile[epoch, 1]
        lambda_b = in_profile[epoch, 2]
        h_b = in_profile[epoch, 3]
        v_eb_n_current = in_profile[epoch, 4:7].reshape(-1, 1)
        eul_nb = in_profile[epoch, 7:10].reshape(-1, 1)
        v_eb_n_prev = in_profile[epoch - 1, 4:7].reshape(-1, 1)
        v_eb_n_next = in_profile[epoch + 1, 4:7].reshape(-1, 1)

        # Smooth velocity
        v_eb_n = 0.5 * v_eb_n_current + 0.25 * v_eb_n_prev + 0.25 * v_eb_n_next

        # Generate output profile record
        out_profile[epoch, 0] = time
        out_profile[epoch, 1] = L_b
        out_profile[epoch, 2] = lambda_b
        out_profile[epoch, 3] = h_b
        out_profile[epoch, 4:7] = v_eb_n.T
        out_profile[epoch, 7:10] = eul_nb.T

    # Last epoch
    out_profile[no_epochs - 1, :] = in_profile[no_epochs - 1, :]

    # Write output profile
    Write_profile(out_filename, out_profile)