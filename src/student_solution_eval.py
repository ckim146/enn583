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
from spatialmath import SE3
import cv2 as cv
import pandas as pd
import csv
import symforce
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix
symforce.set_epsilon_to_symbol()


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

FAIL_THRESH = dict(
    min_matches=100,       # feature matches between frame i and j
    min_points_3d=50,      # matches with valid stereo depth
    min_inliers=30,        # PnP RANSAC inliers
    min_inlier_ratio=0.3,  # inliers / 3D points
    max_rot_deg=10.0,      # a car at 10 Hz rarely turns more than this per frame
    max_trans_m=4.0,       # ~144 km/h at 10 Hz; KITTI is well below this
)

def check_estimate(T, info, th=FAIL_THRESH):
    """Sanity-check one PnP estimate. Returns (ok, reasons, metrics)."""
    n_pts = int(info.get("n_points_3d", 0))
    n_inl = int(np.count_nonzero(info.get("inlier_mask", [])))
    if T is not None:
        rot = float(np.degrees(np.arccos(np.clip((np.trace(T.R) - 1) / 2, -1, 1))))
        trans = float(np.linalg.norm(T.t))
    else:
        rot = trans = float("nan")
    m = dict(n_matches=int(info.get("n_matches", 0)), n_points_3d=n_pts,
             n_inliers=n_inl, inlier_ratio=n_inl / n_pts if n_pts else 0.0,
             rot_deg=rot, trans_m=trans)
 
    r = []
    if T is None:
        r.append("PnP failed to return a pose")
    if m["n_matches"] < th["min_matches"]:
        r.append(f"Too few matches: {m['n_matches']} < {th['min_matches']}")
    if n_pts < th["min_points_3d"]:
        r.append(f"Too few points with valid depth: {n_pts} < {th['min_points_3d']}")
    if n_inl < th["min_inliers"]:
        r.append(f"Too few PnP inliers: {n_inl} < {th['min_inliers']}")
    if m["inlier_ratio"] < th["min_inlier_ratio"]:
        r.append(f"Low PnP inlier ratio: {m['inlier_ratio']:.2f} < {th['min_inlier_ratio']}")
    if rot > th["max_rot_deg"]:
        r.append(f"Implausible rotation: {rot:.1f} deg > {th['max_rot_deg']} deg")
    if trans > th["max_trans_m"]:
        r.append(f"Implausible translation: {trans:.2f} m > {th['max_trans_m']} m")
    return len(r) == 0, r, m

def build_consecutive_motions(dataset):
    relative_motion = []
    diag_rows = []
    last_good = SE3()          # constant-velocity fallback; identity at start
    first_failure_saved = False
 
    for i in range(dataset.frame_count - 1):
        T, info = estimate_relative_pose(dataset, i, i + 1, return_info=True)
        ok, reasons, m = check_estimate(T, info)
 
        if ok:
            last_good = T
        else:
            print(f"WARNING: {i}->{i + 1} rejected: {'; '.join(reasons)}")
            T = last_good
            if not first_failure_saved:
                np.savez("results_first_failure.npz", frame_i=i, frame_j=i + 1,
                         pts_i=np.asarray(info.get("pts_i", np.empty((0, 2)))),
                         pts_j=np.asarray(info.get("pts_j", np.empty((0, 2)))),
                         inlier_mask=np.asarray(info.get("inlier_mask", []), bool),
                         reasons=np.array(reasons))
                first_failure_saved = True
 
        relative_motion.append(T)
        diag_rows.append([i, i + 1, int(ok), *m.values(), "; ".join(reasons)])
 
    with open("results_vo_diagnostics.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame_i", "frame_j", "accepted", "n_matches", "n_points_3d",
                    "n_inliers", "inlier_ratio", "rot_deg", "trans_m", "reasons"])
        w.writerows(diag_rows)
    return relative_motion

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

    ratio_matches = sorted(ratio_matches, key=lambda match: match.distance)
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


#  ====================================================================================
#  ====================================================================================
def estimate_relative_pose(dataset, frame_i: int, frame_j: int, return_info=False):
    """Estimate the relative camera pose between two dataset frames.

    This is the frame-to-frame motion-estimation stage of the visual odometry
    pipeline. Your implementation should load the two requested frames from the
    dataset, establish visual correspondences, and estimate the rigid
    transformation from ``frame_i`` to ``frame_j``.
    
    Parameters
    ----------
    dataset:
        Dataset supplied by the runner. It provides ``stereo(i)``,
        ``camera_calibration(camera)``, ``frame_count``, and ``len(dataset)``.
        
        While you are developing your code, you can also access ``ground_truth_pose(i)`` to get the true pose of frame ``i``. 
        However, ground-truth poses are unavailable during marking.

    frame_i : int
        Index of the first frame.

    frame_j : int
        Index of the second frame.

    Returns
    -------
    None
        The assessment does not require this function to return anything.

    Side effects
    ------------
    Write a CSV file named ``results_relative_pose.csv`` in the current working
    directory. The file should contain your estimated relative pose. The
    required format is one row with these columns:

    ``frame_i,frame_j,x,y,z,roll,pitch,yaw``

    Here ``x,y,z`` are the translation components and ``roll,pitch,yaw`` are
    Euler angles, all describing the left-camera pose of ``frame_j`` relative
    to the left-camera pose of ``frame_i``. Angles must be in radians. The
    checker interprets them using SpatialMath's
    ``SE3.RPY(roll, pitch, yaw, order="zyx")`` convention.

    Example
    -------
    The local checker and Gradescope call this function like this:

    ```python
    frame_i = 0
    frame_j = 1

    estimate_relative_pose(dataset, frame_i, frame_j)
    ```
       
    """

    # ====================================================================================
    # Implement frame-to-frame motion estimation here.
    #
    # Suggested steps:
    #   1. Load the left images for frame_i and frame_j using dataset.stereo(...).
    #   2. Find feature matches between the two frames.
    #   3. Use calibration/depth/geometry to estimate the relative pose.
    #   4. Write results_relative_pose.csv with columns:
    #      frame_i,frame_j,x,y,z,roll,pitch,yaw
    # ====================================================================================
    def ransac_iterations(inlier_ratio, sample_size, confidence=0.99):
        """
        Minimum number of RANSAC iterations needed to have `confidence`
        probability of drawing at least one all-inlier minimal sample.
    
        inlier_ratio : estimated fraction of correspondences that are inliers (0-1)
        sample_size  : points needed per hypothesis (e.g. 3 for P3P, 5 for five-point, 8 for eight-point)
        confidence   : desired probability of success (default 0.99)
        """
        if inlier_ratio <= 0 or inlier_ratio >= 1:
            raise ValueError("inlier_ratio must be between 0 and 1 (exclusive)")
    
        numerator = np.log(1 - confidence)
        denominator = np.log(1 - inlier_ratio ** sample_size)
        return int(np.ceil(numerator / denominator))

    left_current = cv.cvtColor(dataset.stereo(frame_i)[0], cv.COLOR_RGB2GRAY)
    right_current = cv.cvtColor(dataset.stereo(frame_i)[1], cv.COLOR_RGB2GRAY)
    left_next = cv.cvtColor(dataset.stereo(frame_j)[0], cv.COLOR_RGB2GRAY)

    stereo = cv.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16 * 10,  # must be divisible by 16
    blockSize=5,
    P1=8 * 1 * 5 ** 2,   # P1: penalty for changing disparity by exactly one pixel. It discourages small fluctuations and surface noise.
    P2=32 * 1 * 5 ** 2,  # P2: penalty for changing disparity by more than one pixel. It strongly discourages sudden depth jumps.
    disp12MaxDiff=1,     # 1 enables the left-right consistency check.
    uniquenessRatio=10,  # like the ratio check: Requires the best disparity candidate to be sufficiently better than competing candidates. A value of 10 means that a match must pass a roughly 10% uniqueness margin.
    speckleWindowSize=100,  # Removes small connected regions of approximately similar disparity.
    speckleRange=32,     # Controls how much disparity variation is permitted while determining whether pixels belong to the same speckle component.
    preFilterCap=63,     
    mode=cv.STEREO_SGBM_MODE_SGBM_3WAY        
    )
    
    
    # now calculate the disparity map for the current frame
    disparity = stereo.compute(left_current, right_current).astype(np.float32) / 16.0

    matches, kp_current, kp_next  = match_features(left_current, left_next)

    left_calibration = dataset.camera_calibration(camera=2)
    right_calibration = dataset.camera_calibration(camera=3)
    K = left_calibration['K']

    T_right_left = right_calibration['T_cam_imu'] @ np.linalg.inv(left_calibration['T_cam_imu'])
    baseline = np.linalg.norm(T_right_left[:3, 3])
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    points_3d_current = []
    points_2d_next = []
    pts_i_used = []   # pixels in frame_i of the points that got a valid depth

    # Assign disparity values to matched keypoints
    for match in matches:
        pt_current = kp_current[match.queryIdx].pt
        pt_next = kp_next[match.trainIdx].pt
        u, v = int(round(pt_current[0])), int(round(pt_current[1]))
        disparity_value = disparity[v, u]
        if disparity_value > 4.0:
            Z = (fx * baseline) / disparity_value
            X = (pt_current[0] - cx) * Z / fx
            Y = (pt_current[1] - cy) * Z / fy
            points_3d_current.append([X, Y, Z])
            points_2d_next.append([pt_next[0], pt_next[1]])
            pts_i_used.append([pt_current[0], pt_current[1]])
    
    # Use the 3d points to calculate estimated camera pose
    T = None
    inliers = None
    points_3d = np.array(points_3d_current, dtype=np.float64).reshape(-1, 3)
    points_2d = np.array(points_2d_next, dtype=np.float64).reshape(-1, 2)
    pts_i_used = np.array(pts_i_used, dtype=np.float64).reshape(-1, 2)
    if len(points_2d_next) >= 4:
        N = ransac_iterations(inlier_ratio=0.50, sample_size=3, confidence=0.99)

        ok, rvec, tvec, inliers = cv.solvePnPRansac(points_3d, points_2d, K, flags=cv.SOLVEPNP_AP3P, reprojectionError=1.25, distCoeffs=None, iterationsCount=N)
        if ok and inliers is not None and len(inliers) >= 4:
            idx = inliers.flatten()
            rvec, tvec = cv.solvePnPRefineVVS(points_3d[idx], points_2d[idx], K, None, rvec, tvec)
            T = SE3.Rt(cv.Rodrigues(rvec)[0], tvec).inv()
    pnp_succeeded = T is not None
    if T is None:
        T = SE3()   # identity only for the CSV / plain callers
    # print("N: ", N, len(points_2d_next))
    # Write CSV
    headers = ['frame_i' ,'frame_j' , 'x', 'y', 'z', 'roll' ,'pitch', 'yaw']
    x, y, z = T.t
    roll, pitch, yaw = T.rpy(order='zyx')
    with open("results_relative_pose.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)  
        writer.writerow([frame_i, frame_j, x, y, z, roll, pitch, yaw])

    inlier_mask = np.zeros(len(points_2d), bool)
    if pnp_succeeded and inliers is not None:
        inlier_mask[inliers.ravel()] = True
    info = dict(
        n_matches=len(matches),       # all matches before depth filtering
        n_points_3d=len(points_3d),   # matches with valid disparity
        pts_i=pts_i_used,             # (N,2) pixels in frame_i of those points
        pts_j=points_2d,              # (N,2) matching pixels in frame_j
        inlier_mask=inlier_mask,
    )
    if return_info:
        return (T if pnp_succeeded else None), info
    return T

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

    # New relative motion for eval purposes
    relative_motion = build_consecutive_motions(dataset)
    
    # Estimate motion across non-consecutive frames to supplement the consecutive estimation.
    additional_edges = [(i, i + 3) for i in range(data.frame_count - 3)]
    additional_measurements_raw = [
        estimate_relative_pose(data, a, b) for a, b in additional_edges
    ]
    
    estimated_trajectory = [SE3()]
    
    # Convert to imu frame to be comparible with ground truth motion
    for T in relative_motion:
        # relative_motion_in_imu_frame = T_cam_imu.inv() @ T @ T_cam_imu
        # estimated_trajectory.append(estimated_trajectory[-1] @ relative_motion_in_imu_frame)
        estimated_trajectory.append(estimated_trajectory[-1] @ T)

    # Write CSV (once, after the whole trajectory is chained)
    headers = ['frame', 'x', 'y', 'z', 'roll' ,'pitch', 'yaw']
    with open("results_visual_odometry.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for frame, pose in enumerate(estimated_trajectory):
            x, y, z = pose.t
            roll, pitch, yaw = pose.rpy(order='zyx')
            writer.writerow([frame, x, y, z, roll, pitch, yaw])
    
    return None
