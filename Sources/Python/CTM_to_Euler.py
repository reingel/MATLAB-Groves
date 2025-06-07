import numpy as np


def CTM_to_Euler(C):
    """
    CTM_to_Euler - Converts a coordinate transformation matrix to the
    corresponding set of Euler angles
    
    Software for use with "Principles of GNSS, Inertial, and Multisensor
    Integrated Navigation Systems," Second Edition.
    
    This function created 1/4/2012 by Paul Groves
    Converted to Python 2025
    
    Parameters:
    -----------
    C : array_like
        coordinate transformation matrix describing transformation from
        beta to alpha
    
    Returns:
    --------
    eul : ndarray
        Euler angles describing rotation from beta to alpha in the 
        order roll, pitch, yaw (rad)
    
    Copyright 2012, Paul Groves
    License: BSD; see license.txt for details
    """
    
    # Convert input to numpy array
    C = np.array(C)
    
    # Initialize output array
    eul = np.zeros(3)
    
    # Calculate Euler angles using (2.23)
    eul[0] = np.arctan2(C[1, 2], C[2, 2])  # roll
    eul[1] = -np.arcsin(C[0, 2])           # pitch
    eul[2] = np.arctan2(C[0, 1], C[0, 0])  # yaw
    
    return eul


# Test cases
if __name__ == "__main__":
    print("Testing CTM_to_Euler function...")
    print("=" * 50)
    
    # Test Case 1: Identity matrix (no rotation)
    print("Test Case 1: Identity matrix (no rotation)")
    C_identity = np.eye(3)
    euler_identity = CTM_to_Euler(C_identity)
    print(f"Input matrix:\n{C_identity}")
    print(f"Euler angles (roll, pitch, yaw) in radians: {euler_identity}")
    print(f"Euler angles in degrees: {np.degrees(euler_identity)}")
    print()
    
    # Test Case 2: 90-degree yaw rotation
    print("Test Case 2: 90-degree yaw rotation")
    yaw_90 = np.radians(90)
    C_yaw90 = np.array([
        [np.cos(yaw_90), -np.sin(yaw_90), 0],
        [np.sin(yaw_90),  np.cos(yaw_90), 0],
        [0,               0,              1]
    ])
    euler_yaw90 = CTM_to_Euler(C_yaw90)
    print(f"Input matrix (90° yaw):\n{C_yaw90}")
    print(f"Euler angles (roll, pitch, yaw) in radians: {euler_yaw90}")
    print(f"Euler angles in degrees: {np.degrees(euler_yaw90)}")
    print()
    
    # Test Case 3: 45-degree pitch rotation
    print("Test Case 3: 45-degree pitch rotation")
    pitch_45 = np.radians(45)
    C_pitch45 = np.array([
        [np.cos(pitch_45),  0, np.sin(pitch_45)],
        [0,                 1, 0],
        [-np.sin(pitch_45), 0, np.cos(pitch_45)]
    ])
    euler_pitch45 = CTM_to_Euler(C_pitch45)
    print(f"Input matrix (45° pitch):\n{C_pitch45}")
    print(f"Euler angles (roll, pitch, yaw) in radians: {euler_pitch45}")
    print(f"Euler angles in degrees: {np.degrees(euler_pitch45)}")
    print()
    
    # Test Case 4: 30-degree roll rotation
    print("Test Case 4: 30-degree roll rotation")
    roll_30 = np.radians(30)
    C_roll30 = np.array([
        [1, 0,               0],
        [0, np.cos(roll_30), -np.sin(roll_30)],
        [0, np.sin(roll_30),  np.cos(roll_30)]
    ])
    euler_roll30 = CTM_to_Euler(C_roll30)
    print(f"Input matrix (30° roll):\n{C_roll30}")
    print(f"Euler angles (roll, pitch, yaw) in radians: {euler_roll30}")
    print(f"Euler angles in degrees: {np.degrees(euler_roll30)}")
    print()
    
    # Test Case 5: Combined rotation (roll=15°, pitch=25°, yaw=35°)
    print("Test Case 5: Combined rotation (roll=15°, pitch=25°, yaw=35°)")
    roll = np.radians(15)
    pitch = np.radians(25)
    yaw = np.radians(35)
    
    # Create individual rotation matrices
    R_roll = np.array([
        [1, 0,           0],
        [0, np.cos(roll), -np.sin(roll)],
        [0, np.sin(roll),  np.cos(roll)]
    ])
    
    R_pitch = np.array([
        [np.cos(pitch),  0, np.sin(pitch)],
        [0,              1, 0],
        [-np.sin(pitch), 0, np.cos(pitch)]
    ])
    
    R_yaw = np.array([
        [np.cos(yaw), -np.sin(yaw), 0],
        [np.sin(yaw),  np.cos(yaw), 0],
        [0,            0,           1]
    ])
    
    # Combined rotation matrix (yaw * pitch * roll)
    C_combined = np.dot(R_yaw, np.dot(R_pitch, R_roll))
    euler_combined = CTM_to_Euler(C_combined)
    
    print(f"Expected angles (roll, pitch, yaw) in degrees: [15.0, 25.0, 35.0]")
    print(f"Input matrix (combined rotation):\n{C_combined}")
    print(f"Computed Euler angles (roll, pitch, yaw) in radians: {euler_combined}")
    print(f"Computed Euler angles in degrees: {np.degrees(euler_combined)}")
    print(f"Error in degrees: {np.degrees(euler_combined) - [15.0, 25.0, 35.0]}")
    print()
    
    print("All test cases completed!")