### Key Differences Across Demos
1. **Motion Profiles**:
   - **Demo 1NED, 1ECI, 1ECEF, 2, 3**: Use `Profile_1.csv` (60s artificial car motion with two 90° turns).
   - **Demo 4**: Uses `Profile_0.csv` (60s stationary and level).
   - **Demo 5**: Uses `Profile_2.csv` (175s car motion).
   - **Demo 6**: Uses `Profile_3.csv` (418s aircraft motion).
   - **Demo 7**: Uses `Profile_4.csv` (300s boat motion).

2. **Reference Frames**:
   - **Demo 1NED, 2, 3, 4, 5, 6, 7**: Use NED (North-East-Down) frame with `Inertial_navigation_NED`.
   - **Demo 1ECI**: Uses ECI (Earth-Centered Inertial) frame with `Inertial_navigation_ECI`.
   - **Demo 1ECEF**: Uses ECEF (Earth-Centered Earth-Fixed) frame with `Inertial_navigation_ECEF`.

3. **IMU Error Models**:
   - **Tactical-grade** (Demo 1NED, 1ECI, 1ECEF, 4, 5, 7):
     - Accelerometer biases: [900, -1300, 800] µg
     - Gyro biases: [-9, 13, -8] deg/hour
     - Accel noise PSD: 100 µg/√Hz
     - Gyro noise PSD: 0.01 deg/√hour
     - Accel quantization: 0.01 m/s²
     - Gyro quantization: 0.0002 rad/s
   - **Aviation-grade** (Demo 2, 6):
     - Accelerometer biases: [30, -45, 26] µg
     - Gyro biases: [-0.0009, 0.0013, -0.0008] deg/hour
     - Accel noise PSD: 20 µg/√Hz
     - Gyro noise PSD: 0.002 deg/√hour
     - Accel quantization: 5e-5 m/s²
     - Gyro quantization: 1e-6 rad/s
     - Zero g-dependent gyro biases
   - **Consumer-grade** (Demo 3):
     - Accelerometer biases: [9000, -13000, 8000] µg
     - Gyro biases: [-180, 260, -160] deg/hour
     - Accel noise PSD: 1000 µg/√Hz
     - Gyro noise PSD: 1 deg/√hour
     - Accel quantization: 0.1 m/s²
     - Gyro quantization: 0.002 rad/s

4. **Initialization Errors**:
   - **Demo 2, 6**: Smaller attitude errors ([-0.01, 0.008, 0.01] deg).
   - **Demo 3**: Larger attitude errors ([-0.5, 0.4, 2] deg).
   - **Others**: Attitude errors ([-0.05, 0.04, 1] deg).
