# ====================================================================================
# Student implementation scaffold for the ENN583 2026 coding project assessment.
# ====================================================================================
# 
# You may add helper functions, classes, and additional modules inside ``src/``.
# IMPORTANT: Do not change the names or parameters of the three provided functions. 
# Gradescope and the local assessment checker call them directly. If you make changes to these
# functions or their parameters, your code will not run correctly on Gradescope and you will lose marks.
# 
# ``match_features(img_i, img_j)``
#     Match visual features between two provided images and write
#     ``results_matches.csv``.
# 
# ``estimate_relative_pose(dataset, frame_i, frame_j)``
#     Estimate the relative motion between two dataset frames and write
#     ``results_relative_pose.csv``.
# 
# ``visual_odometry(dataset)``
#     Estimate the trajectory for a dataset sequence and write
#     ``results_visual_odometry.csv``.


import numpy as np
import spatialmath as sm
import symforce
symforce.set_epsilon_to_symbol()
import symforce.symbolic as sf
from symforce.values import Values
from symforce.opt.factor import Factor
from symforce.opt.optimizer import Optimizer
from spatialmath import SE3
import cv2 as cv
import pandas as pd
import csv
import symforce
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix



# ====================================================================================
# Your code goes here! 
# ====================================================================================
# Add your helper functions and classes below this line. 
# You can also create additional files in ``src/`` and import them here. 
#
# You may use packages provided by environment.yml. Ask the teaching team before
# adding another dependency because it may not be installed in Gradescope.
#
# Complete the three provided functions below. 
# Do not change their names or parameters because Gradescope calls them directly.
# Start with the match_features function, then implement estimate_relative_pose, and finally implement visual_odometry.
# You can and should reuse the functions, i.e. call match_features from estimate_relative_pose, and call estimate_relative_pose from visual_odometry.



#  ====================================================================================
#  ====================================================================================
def match_features(img_i: np.ndarray, img_j: np.ndarray):
    """Detect and match visual features between two provided images.

    This is the first, most local part of the visual odometry pipeline. Your
    implementation should detect keypoints in ``img_i`` and ``img_j``, compute
    descriptors, match the descriptors, and apply any filtering strategy you
    think is appropriate.

    Parameters
    ----------
    img_i : np.ndarray
        First image. It may be grayscale or colour. Pixel coordinates u_i,v_i written to
        the output file must refer to this image's coordinate system.

    img_j : np.ndarray
        Second image. It may be grayscale or colour. Pixel coordinates u_j,v_j written
        to the output file must refer to this image's coordinate system.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_matches.csv`` in the current working
    directory. Each row should describe one matched point pair between the two
    input images. The required columns are:

    ``match_id,u_i,v_i,u_j,v_j``
    
    Here ``u_i,v_i`` are the pixel coordinates of the feature in ``img_i`` and
    ``u_j,v_j`` are the pixel coordinates of the corresponding feature in
    ``img_j``. The ``match_id`` column should contain a unique integer for each
    match, starting from 0. The order of the features does not matter.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    img_i = dataset.stereo(frame_i)[0]  # left image from frame_i
    img_j = dataset.stereo(frame_j)[0]  # left image from frame_j

    match_features(img_i, img_j)
    ```
    
    """ 

    # ====================================================================================
    # Implement feature matching here.
    #
    # Suggested steps:
    #   1. Detect keypoints in img_i and img_j.
    #   2. Compute descriptors for the keypoints.
    #   3. Match descriptors between the two images.
    #   4. Optionally filter unreliable matches.
    #   5. Write results_matches.csv with columns:
    #      match_id,u_i,v_i,u_j,v_j
    # ====================================================================================
    MAX_POINTS = 3000
    sift = cv.SIFT_create(nfeatures=MAX_POINTS, contrastThreshold=0.02, edgeThreshold=15)
    sift_i_keypoints, sift_i_descriptors = sift.detectAndCompute(img_i, None)
    sift_j_keypoints, sift_j_descriptors = sift.detectAndCompute(img_j, None)

    matcher = cv.BFMatcher(cv.NORM_L2)
    nearest_matches = matcher.knnMatch(sift_i_descriptors, sift_j_descriptors, k=2) # return 2 matches per point instead of 1 for ratio test
    # Conduct ratio test to refine matches
    ratio_matches = []
    for (best, second_best) in nearest_matches:         
        if best.distance < 0.55 * second_best.distance:
            ratio_matches.append(best)
    
        ratio_matches = sorted(
            ratio_matches, key=lambda match: match.distance
        )
    # Initialize df to be turned into a CSV later
    headers = ["match_id", "u_i", "v_i", "u_j", "v_j"]
    with open("results_matches.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for match_id, m in enumerate(ratio_matches):
            u_i, v_i = sift_i_keypoints[m.queryIdx].pt
            u_j, v_j = sift_j_keypoints[m.trainIdx].pt
            writer.writerow([match_id, u_i, v_i, u_j, v_j])
    
    return ratio_matches, sift_i_keypoints, sift_j_keypoints

def _relative_pose(dataset, frame_i: int, frame_j: int):
    """Estimate relative pose. Returns SE3 (frame_i <- frame_j), or None on failure."""

    def ransac_iterations(inlier_ratio, sample_size, confidence=0.99):
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            raise ValueError("inlier_ratio must be between 0 and 1 (exclusive)")
        return int(np.ceil(np.log(1 - confidence) / np.log(1 - inlier_ratio ** sample_size)))

    left_current = cv.cvtColor(dataset.stereo(frame_i)[0], cv.COLOR_RGB2GRAY)
    right_current = cv.cvtColor(dataset.stereo(frame_i)[1], cv.COLOR_RGB2GRAY)
    left_next = cv.cvtColor(dataset.stereo(frame_j)[0], cv.COLOR_RGB2GRAY)

    stereo = cv.StereoSGBM_create(
        minDisparity=0, numDisparities=16 * 10, blockSize=5,
        P1=8 * 1 * 5 ** 2, P2=32 * 1 * 5 ** 2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32,
        preFilterCap=63, mode=cv.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disparity = stereo.compute(left_current, right_current).astype(np.float32) / 16.0

    matches, kp_current, kp_next = match_features(left_current, left_next)

    left_calibration = dataset.camera_calibration(camera=2)
    right_calibration = dataset.camera_calibration(camera=3)
    K = left_calibration['K']
    T_right_left = right_calibration['T_cam_imu'] @ np.linalg.inv(left_calibration['T_cam_imu'])
    baseline = np.linalg.norm(T_right_left[:3, 3])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points_3d_current, points_2d_next = [], []
    h, w = disparity.shape
    for match in matches:
        pt_current = kp_current[match.queryIdx].pt
        pt_next = kp_next[match.trainIdx].pt
        u, v = int(round(pt_current[0])), int(round(pt_current[1]))
        if not (0 <= u < w and 0 <= v < h):
            continue
        disparity_value = disparity[v, u]
        if disparity_value > 0:
            Z = (fx * baseline) / disparity_value
            X = (pt_current[0] - cx) * Z / fx
            Y = (pt_current[1] - cy) * Z / fy
            points_3d_current.append([X, Y, Z])
            points_2d_next.append([pt_next[0], pt_next[1]])

    if len(points_2d_next) < 4:
        return None

    points_3d = np.array(points_3d_current)
    points_2d = np.array(points_2d_next)
    N = ransac_iterations(inlier_ratio=0.5, sample_size=3, confidence=0.99)
    ok, rvec, tvec, inliers = cv.solvePnPRansac(
        points_3d, points_2d, K, distCoeffs=None,
        flags=cv.SOLVEPNP_AP3P, reprojectionError=1.25, iterationsCount=N,
    )
    if not ok or inliers is None or len(inliers) < 4:
        return None

    idx = inliers.flatten()
    rvec, tvec = cv.solvePnPRefineVVS(points_3d[idx], points_2d[idx], K, None, rvec, tvec)
    return SE3.Rt(cv.Rodrigues(rvec)[0], tvec).inv()

def estimate_relative_pose(dataset, frame_i, frame_j):
    T = _relative_pose(dataset, frame_i, frame_j) or SE3()
    x, y, z = T.t
    roll, pitch, yaw = T.rpy(order='zyx')
    with open("results_relative_pose.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(['frame_i','frame_j','x','y','z','roll','pitch','yaw'])
        w.writerow([frame_i, frame_j, x, y, z, roll, pitch, yaw])
#  ====================================================================================
#  ====================================================================================
# def estimate_relative_pose(dataset, frame_i: int, frame_j: int):
#     """Estimate the relative camera pose between two dataset frames.

#     This is the frame-to-frame motion-estimation stage of the visual odometry
#     pipeline. Your implementation should load the two requested frames from the
#     dataset, establish visual correspondences, and estimate the rigid
#     transformation from ``frame_i`` to ``frame_j``.
    
#     Parameters
#     ----------
#     dataset:
#         Dataset supplied by the runner. It provides ``stereo(i)``,
#         ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
#         While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
#         However, ground-truth poses are unavailable during marking.

#     frame_i : int
#         Index of the first frame.

#     frame_j : int
#         Index of the second frame.

#     Returns
#     -------
#     None
#         The assessment does not require this function to return anything.

#     Side effects
#     ------------
#     Write a CSV file named ``results_relative_pose.csv`` in the current working
#     directory. The file should contain your estimated relative pose. The
#     required format is one row with these columns:

#     ``frame_i,frame_j,x,y,z,roll,pitch,yaw``

#     Here ``x,y,z`` are the translation components and ``roll,pitch,yaw`` are
#     Euler angles, all describing the left-camera pose of ``frame_j`` relative
#     to the left-camera pose of ``frame_i``. Angles must be in radians. The
#     checker interprets them using SpatialMath's
#     ``SE3.RPY(roll, pitch, yaw, order="zyx")`` convention.

#     Example
#     -------
#     The local checker and Gradescope call this function like this:

#     ```python
#     frame_i = 0
#     frame_j = 1

#     estimate_relative_pose(dataset, frame_i, frame_j)
#     ```
       
#     """

#     # ====================================================================================
#     # Implement frame-to-frame motion estimation here.
#     #
#     # Suggested steps:
#     #   1. Load the left images for frame_i and frame_j using dataset.stereo(...).
#     #   2. Find feature matches between the two frames.
#     #   3. Use calibration/depth/geometry to estimate the relative pose.
#     #   4. Write results_relative_pose.csv with columns:
#     #      frame_i,frame_j,x,y,z,roll,pitch,yaw
#     # ====================================================================================
#     def ransac_iterations(inlier_ratio, sample_size, confidence=0.99):
#         """
#         Minimum number of RANSAC iterations needed to have `confidence`
#         probability of drawing at least one all-inlier minimal sample.
    
#         inlier_ratio : estimated fraction of correspondences that are inliers (0-1)
#         sample_size  : points needed per hypothesis (e.g. 3 for P3P, 5 for five-point, 8 for eight-point)
#         confidence   : desired probability of success (default 0.99)
#         """
#         if inlier_ratio <= 0 or inlier_ratio >= 1:
#             raise ValueError("inlier_ratio must be between 0 and 1 (exclusive)")
    
#         numerator = np.log(1 - confidence)
#         denominator = np.log(1 - inlier_ratio ** sample_size)
#         return int(np.ceil(numerator / denominator))

#     left_current = cv.cvtColor(dataset.stereo(frame_i)[0], cv.COLOR_RGB2GRAY)
#     right_current = cv.cvtColor(dataset.stereo(frame_i)[1], cv.COLOR_RGB2GRAY)
#     left_next = cv.cvtColor(dataset.stereo(frame_j)[0], cv.COLOR_RGB2GRAY)

#     stereo = cv.StereoSGBM_create(
#     minDisparity=0,
#     numDisparities=16 * 10,  # must be divisible by 16
#     blockSize=5,
#     P1=8 * 1 * 5 ** 2,   # P1: penalty for changing disparity by exactly one pixel. It discourages small fluctuations and surface noise.
#     P2=32 * 1 * 5 ** 2,  # P2: penalty for changing disparity by more than one pixel. It strongly discourages sudden depth jumps.
#     disp12MaxDiff=1,     # 1 enables the left-right consistency check.
#     uniquenessRatio=10,  # like the ratio check: Requires the best disparity candidate to be sufficiently better than competing candidates. A value of 10 means that a match must pass a roughly 10% uniqueness margin.
#     speckleWindowSize=100,  # Removes small connected regions of approximately similar disparity.
#     speckleRange=32,     # Controls how much disparity variation is permitted while determining whether pixels belong to the same speckle component.
#     preFilterCap=63,     
#     mode=cv.STEREO_SGBM_MODE_SGBM_3WAY        
#     )
    
    
#     # now calculate the disparity map for the current frame
#     disparity = stereo.compute(left_current, right_current).astype(np.float32) / 16.0

#     matches, kp_current, kp_next  = match_features(left_current, left_next)

#     left_calibration = dataset.camera_calibration(camera=2)
#     right_calibration = dataset.camera_calibration(camera=3)
#     K = left_calibration['K']

#     T_right_left = right_calibration['T_cam_imu'] @ np.linalg.inv(left_calibration['T_cam_imu'])
#     baseline = np.linalg.norm(T_right_left[:3, 3])
#     fx, fy = K[0, 0], K[1, 1]
#     cx, cy = K[0, 2], K[1, 2]

#     points_3d_current = []
#     points_2d_next = [] 

#     # Assign disparity values to matched keypoints
#     for match in matches:
#         pt_current = kp_current[match.queryIdx].pt
#         pt_next = kp_next[match.trainIdx].pt
#         u, v = int(round(pt_current[0])), int(round(pt_current[1]))
#         disparity_value = disparity[v, u]
#         if disparity_value > 0:
#             Z = (fx * baseline) / disparity_value
#             X = (pt_current[0] - cx) * Z / fx
#             Y = (pt_current[1] - cy) * Z / fy
#             points_3d_current.append([X, Y, Z])
#             points_2d_next.append([pt_next[0], pt_next[1]])
    
#     # Use the 3d points to calculate estimated camera pose
#     T = None
#     if len(points_2d_next) >= 4:
#         points_3d = np.array(points_3d_current)
#         points_2d = np.array(points_2d_next)
#         K = dataset.camera_calibration(2)["K"]
#         N = ransac_iterations(inlier_ratio=0.5, sample_size=3, confidence=0.99)
#         ok, rvec, tvec, inliers = cv.solvePnPRansac(points_3d, points_2d, K, flags=cv.SOLVEPNP_AP3P, reprojectionError=1.25, distCoeffs=None, iterationsCount=N)
#         if ok and inliers is not None and len(inliers) >= 4:
#             idx = inliers.flatten()
#             rvec, tvec = cv.solvePnPRefineVVS(points_3d[idx], points_2d[idx], K, None, rvec, tvec)
#             T = SE3.Rt(cv.Rodrigues(rvec)[0], tvec).inv()
#     if T is None:
#         T = SE3()

#     # Write CSV
#     headers = ['frame_i' ,'frame_j' , 'x', 'y', 'z', 'roll' ,'pitch', 'yaw']
#     x, y, z = T.t
#     roll, pitch, yaw = T.rpy(order='zyx')
#     with open("results_relative_pose.csv", "w", newline="") as f:
#         writer = csv.writer(f)
#         writer.writerow(headers)  
#         writer.writerow([frame_i, frame_j, x, y, z, roll, pitch, yaw])

#     return T

def spatialmath_to_symforce(spatial_pose):
    """Convert a SpatialMath SE3 into a SymForce Pose3."""
    T = spatial_pose.A
    rotation = sf.Rot3.from_rotation_matrix(sf.Matrix33(T[:3, :3]))
    translation = sf.V3(*T[:3, 3])
    return sf.Pose3(rotation, translation)


# def symforce_to_spatialmath(symforce_pose):
#     """Convert a SymForce Pose3 into a SpatialMath SE3."""
#     rotation = np.array(
#         symforce_pose.R.to_rotation_matrix(), dtype=float
#     ).reshape(3, 3)
#     translation = np.array(symforce_pose.t, dtype=float).reshape(3)
#     return SE3.Rt(rotation, translation)
def symforce_to_spatialmath(pose):
    R = np.array(pose.R.to_rotation_matrix()).astype(np.float64)
    t = np.array(pose.t).astype(np.float64).ravel()
    return SE3.Rt(R, t)

def plot_trajectory(poses, label, style):
    x = [float(pose.t[0]) for pose in poses]
    z = [float(pose.t[2]) for pose in poses]
    plt.plot(x, z, style, label=label)


def translation_rmse(estimated, reference):
    squared_errors = []
    for estimate, truth in zip(estimated, reference):
        difference = np.array(estimate.t, dtype=float) - np.array(truth.t, dtype=float)
        squared_errors.append(difference @ difference)
    return np.sqrt(np.mean(squared_errors))


def final_position_error(estimated, reference):
    difference = (
        np.array(estimated[-1].t, dtype=float)
        - np.array(reference[-1].t, dtype=float)
    )
    return np.linalg.norm(difference)
    
def between_residual(
    pose_a: sf.Pose3,
    pose_b: sf.Pose3,
    measured_a_T_b: sf.Pose3,
    sqrt_information: sf.V6,
    epsilon: sf.Scalar,
) -> sf.V6:
    predicted_a_T_b = pose_a.inverse() * pose_b
    # local_coordinates() expresses the difference from the measured pose to
    # the predicted pose as a 6D tangent-space vector. It is zero when the
    # poses agree and avoids incorrectly subtracting rotation matrices.
    pose_error = measured_a_T_b.local_coordinates(predicted_a_T_b, epsilon)
    return sf.V6(*pose_error).multiply_elementwise(sqrt_information)

#  ====================================================================================
#  ====================================================================================
def visual_odometry(dataset):
    """Estimate the camera trajectory for a full dataset sequence.

    This is the complete visual odometry stage. Your implementation should
    estimate the camera pose for each frame in the sequence, usually by chaining
    together frame-to-frame relative poses.

    Parameters
    ----------
    dataset:
        Dataset supplied by the runner. It provides ``stereo(i)``,
        ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
        While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
        However, ground-truth poses are unavailable during marking.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_visual_odometry.csv`` in the current
    working directory. The file should contain your estimated trajectory. The
    required format is one row per frame with these columns:

    ``frame,x,y,z,roll,pitch,yaw``

    Each pose should describe the camera pose for that frame relative to frame
    0. The pose for frame 0 should normally be the identity pose.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    visual_odometry(dataset)
    ```

    """

    # ====================================================================================
    # Implement full visual odometry here.
    #
    # Suggested steps:
    #   1. Start with the identity pose for frame 0.
    #   2. Estimate relative poses between successive frames.
    #   3. Chain the relative poses to build the full trajectory.
    #   4. Write results_visual_odometry.csv with columns:
    #      frame,x,y,z,roll,pitch,yaw
    # ====================================================================================
    data = dataset
    T_cam_imu_matrix = data.camera_calibration(camera=2)['T_cam_imu']
    T_cam_imu = SE3.Rt(T_cam_imu_matrix[:3, :3], T_cam_imu_matrix[:3, 3], check=False) 

    relative_motion = []

    left_calibration = data.camera_calibration(camera=2)
    right_calibration = data.camera_calibration(camera=3)
    K = left_calibration['K']
    
    T_right_left = right_calibration['T_cam_imu'] @ np.linalg.inv(left_calibration['T_cam_imu'])
    baseline = np.linalg.norm(T_right_left[:3, 3])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Estimate motion across consecutive frames.
    # relative_motion = [
    #     estimate_relative_pose(data, i, i + 1)
    #     for i in range(data.frame_count - 1)
    # ]
    relative_motion = []
    for i in range(data.frame_count - 1):
        m = _relative_pose(data, i, i + 1)
        if m is None:
            print(f"WARNING: odometry {i}->{i+1} failed, using identity")
            m = SE3()
        relative_motion.append(m)

    
    # Estimate motion across non-consecutive frames to supplement the consecutive estimation.

    additional_edges, additional_measurements_raw = [], []
    for a in range(data.frame_count - 3):
        m = _relative_pose(data, a, a + 3)
        if m is not None:
            additional_edges.append((a, a + 3))
            additional_measurements_raw.append(m)
            
    # additional_edges = [(i, i + 3) for i in range(data.frame_count - 3)]
    # additional_measurements_raw = [
    #     estimate_relative_pose(data, a, b) for a, b in additional_edges
    # ]
    
    estimated_trajectory = [SE3()]

    # Pose graph optimization

    frame_motions_in_imu = [
    T_cam_imu.inv() @ motion @ T_cam_imu
    for motion in relative_motion
    ]
    
    # sigma = np.maximum(errors.std(axis=0), 1e-4)
    
    odometry_measurements = [
        spatialmath_to_symforce(motion)
        for motion in relative_motion
    ]
    
    initial_poses = [sf.Pose3.identity()]
    for relative_pose in odometry_measurements:
        initial_poses.append(initial_poses[-1] * relative_pose)
    

    additional_in_imu = [
    T_cam_imu.inv() @ motion @ T_cam_imu
    for motion in additional_measurements_raw
    ]


    
    # sigma_sf            = np.concatenate([sigma[3:], sigma[:3]])

    
    additional_measurements = [spatialmath_to_symforce(m) for m in additional_measurements_raw]
    

    values = Values(
        poses=initial_poses,
        odometry=odometry_measurements,
        additional_motion=additional_measurements,
        # odometry_sqrt_information = sf.V6(*[float(x) for x in (1.0 / sigma_sf)]),
        # additional_sqrt_information=sf.V6(*(1.0 / sigma_additional_sf)),
        odometry_sqrt_information=sf.V6(1/0.008, 1/0.008, 1/0.01, 1/0.04, 1/0.01, 1/0.04),
        additional_sqrt_information=sf.V6(1/0.005, 1/0.005, 1/0.007, 1/0.02, 1/0.008, 1/0.02),
        epsilon=sf.numeric_epsilon,
    )
    
    # print("sigmas: ", sigma)
    # print("sigmas_additional : ", sigma_additional)
    
    factors = []
    initial_poses = list(initial_poses)
    
    for i in range(len(odometry_measurements)):
        factors.append(
            Factor(
                residual=between_residual,
                keys=[
                    f"poses[{i}]",
                    f"poses[{i + 1}]",
                    f"odometry[{i}]",
                    "odometry_sqrt_information",
                    "epsilon",
                ],
            )
        )
    
    for measurement_index, (i, j) in enumerate(additional_edges):
        factors.append(
            Factor(
                residual=between_residual,
                keys=[
                    f"poses[{i}]",
                    f"poses[{j}]",
                    f"additional_motion[{measurement_index}]",
                    "additional_sqrt_information",
                    "epsilon",
                ],
            )
        )
    
    print(f"{len(initial_poses)} pose variables")
    print(f"{len(factors)} relative-pose factors")

    keys_to_optimise = [f"poses[{i}]" for i in range(1, len(initial_poses))]

    optimizer = Optimizer(
        factors=factors,
        optimized_keys=keys_to_optimise,
        params=Optimizer.Params(verbose=False),
    )
    result = optimizer.optimize(values)
    optimised_poses = result.optimized_values["poses"]
    
    
    
    print("Status:", result.status)
    print(f"Final squared error: {result.error():.3f}")
    estimated_trajectory = [symforce_to_spatialmath(p) for p in optimised_poses]

    # End of PGO implementation
    headers = ['frame', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']
    with open("results_visual_odometry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for frame, pose in enumerate(estimated_trajectory):
            x, y, z = pose.t
            roll, pitch, yaw = pose.rpy(order='zyx')
            writer.writerow([frame, x, y, z, roll, pitch, yaw])
    
    
    # Convert to imu frame to be comparible with ground truth motion
    # for T in optimised_poses:
    #     # relative_motion_in_imu_frame = T_cam_imu.inv() @ T @ T_cam_imu
    #     # estimated_trajectory.append(estimated_trajectory[-1] @ relative_motion_in_imu_frame)
    #     estimated_trajectory.append(estimated_trajectory[-1] @ T)
    
    #     # convert to numpy array for plotting
    #     # estimated_trajectory = np.array([pose.t for pose in estimated_trajectory])
    
    #     # ground_truth_trajectory = [data.ground_truth_pose(frame) for frame in range(data.frame_count)]
    #     # ground_truth_positions = np.array([pose.t for pose in ground_truth_trajectory])
    
    #     # Write CSV
    #     headers = ['frame', 'x', 'y', 'z', 'roll' ,'pitch', 'yaw']
    #     with open("results_visual_odometry.csv", "w", newline="") as f:
    #         writer = csv.writer(f)
    #         writer.writerow(headers)
    #         for frame, pose in enumerate(estimated_trajectory):
    #             x, y, z = pose.t
    #             roll, pitch, yaw = pose.rpy(order='zyx')
    #             writer.writerow([frame, x, y, z, roll, pitch, yaw])
    
    return None
