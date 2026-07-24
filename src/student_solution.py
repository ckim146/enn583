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
import cv2 as cv


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

    return None




#  ====================================================================================
#  ====================================================================================
def estimate_relative_pose(dataset, frame_i: int, frame_j: int):
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

    return None

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

    return None
