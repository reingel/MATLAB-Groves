import numpy as np

def IMU_model(tor_i, true_f_ib_b, true_omega_ib_b, IMU_errors, old_quant_residuals):
    """
    Simulates an inertial measurement unit (IMU body axes used throughout this function)
    
    Args:
        tor_i (float): time interval between epochs (s)
        true_f_ib_b (np.ndarray): true specific force of body frame w.r.t. ECEF frame,
                                 resolved along body-frame axes, averaged over time interval (m/s^2)
        true_omega_ib_b (np.ndarray): true angular rate of body frame w.r.t. ECEF frame,
                                     resolved about body-frame axes, averaged over time interval (rad/s)
        IMU_errors (dict): dictionary containing IMU error parameters
            - b_a (np.ndarray): Accelerometer biases (m/s^2)
            - b_g (np.ndarray): Gyro biases (rad/s)
            - M_a (np.ndarray): Accelerometer scale factor and cross coupling errors
            - M_g (np.ndarray): Gyro scale factor and cross coupling errors
            - G_g (np.ndarray): Gyro g-dependent biases (rad-sec/m)
            - accel_noise_root_PSD (float): Accelerometer noise root PSD (m s^-1.5)
            - gyro_noise_root_PSD (float): Gyro noise root PSD (rad s^-0.5)
            - accel_quant_level (float): Accelerometer quantization level (m/s^2)
            - gyro_quant_level (float): Gyro quantization level (rad/s)
        old_quant_residuals (np.ndarray): residuals of previous output quantization process
    
    Returns:
        tuple: (meas_f_ib_b, meas_omega_ib_b, quant_residuals)
            - meas_f_ib_b (np.ndarray): output specific force of body frame w.r.t. ECEF frame
            - meas_omega_ib_b (np.ndarray): output angular rate of body frame w.r.t. ECEF frame
            - quant_residuals (np.ndarray): residuals of output quantization process
    """
    
    # Generate noise
    if tor_i > 0:
        accel_noise = np.random.randn(3,) * IMU_errors['accel_noise_root_PSD'] / np.sqrt(tor_i)
        gyro_noise = np.random.randn(3,) * IMU_errors['gyro_noise_root_PSD'] / np.sqrt(tor_i)
    else:
        accel_noise = np.zeros((3,))
        gyro_noise = np.zeros((3,))
    
    # Calculate accelerometer and gyro outputs
    uq_f_ib_b = (IMU_errors['b_a'] + 
                 (np.eye(3) + IMU_errors['M_a']) @ true_f_ib_b + 
                 accel_noise)
    
    uq_omega_ib_b = (IMU_errors['b_g'] + 
                    (np.eye(3) + IMU_errors['M_g']) @ true_omega_ib_b + 
                    IMU_errors['G_g'] @ true_f_ib_b + 
                    gyro_noise)
    
    # Quantize accelerometer outputs
    quant_residuals = np.zeros((6,))
    if IMU_errors['accel_quant_level'] > 0:
        meas_f_ib_b = IMU_errors['accel_quant_level'] * np.round(
            (uq_f_ib_b + old_quant_residuals[:3]) / IMU_errors['accel_quant_level'])
        quant_residuals[:3] = uq_f_ib_b + old_quant_residuals[:3] - meas_f_ib_b
    else:
        meas_f_ib_b = uq_f_ib_b
        quant_residuals[:3] = np.zeros((3,))
    
    # Quantize gyro outputs
    if IMU_errors['gyro_quant_level'] > 0:
        meas_omega_ib_b = IMU_errors['gyro_quant_level'] * np.round(
            (uq_omega_ib_b + old_quant_residuals[3:6]) / IMU_errors['gyro_quant_level'])
        quant_residuals[3:6] = uq_omega_ib_b + old_quant_residuals[3:6] - meas_omega_ib_b
    else:
        meas_omega_ib_b = uq_omega_ib_b
        quant_residuals[3:6] = np.zeros((3,))
    
    return meas_f_ib_b, meas_omega_ib_b, quant_residuals