"""Run the ENN583 student solution on a small KITTI example.

This file is intentionally written as a simple script rather than a clever test
framework. Read the three ``check_*`` functions below to see exactly how your
three functions in ``src/student_solution.py`` are called.

Most helper code lives in ``checker_utils.py`` so that this file can focus on
the three checks that students care about:

1. feature matching;
2. frame-to-frame relative pose;
3. full visual odometry.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import math
import os
from pathlib import Path
import sys
import time
import statistics
import matplotlib.pyplot as plt

import numpy as np
import spatialmath as sm

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SUPPORT_DIR = REPO_ROOT / "support"

sys.path.insert(0, str(SUPPORT_DIR))
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(REPO_ROOT))

import kitti_utils as kitti
from checker_utils import (
    FirstNFrames,
    TeeOutput,
    choose_random_frame_pairs,
    is_number,
    plot_trajectory_comparison,
    pose_from_csv_row,
    remove_old_file,
    rotation_angle_rad,
    sampson_errors_pixels,
    trusted_fundamental_matrix,
    warn_if_angles_look_like_degrees,
)
from vo_failure_tools import (
    DIAGNOSTICS_CSV,
    FIRST_FAILURE_NPZ,
    load_diagnostics,
    plot_trajectory_colored,
    show_first_failure,
    spike_frames,
    worst_region,
    show_frame_pair,
    blur_profile,
)


solution = importlib.import_module("student_solution_eval")


MIN_TRANSLATION_FOR_PERCENT_ERROR = 0.05
MIN_ROTATION_FOR_PERCENT_ERROR = 0.001


# ====================================================================================
# Main Checking Functions, one for each of the three student functions. 
# 
# Each function calls the student function, checks that it ran without error, 
# and checks that the output CSV file exists and has the expected format.
# ====================================================================================
def check_match_features(dataset, full_dataset, frame_pairs) -> bool:
    """Check the student's ``match_features(img_i, img_j)`` function.

    The student function receives only two images.  After it writes
    ``results_matches.csv``, this checker verifies the CSV format and then
    gives feedback on geometric correctness using a trusted fundamental matrix
    computed from KITTI calibration and ground-truth left-camera poses.
    """

    print("\n" + "=" * 80)
    print("Checking match_features(img_i, img_j)")
    print("=" * 80)
    print(f"[INFO] Checking {len(frame_pairs)} randomly selected frame pair(s).")

    expected_header = ["match_id", "u_i", "v_i", "u_j", "v_j"]
    all_pairs_passed = True
    runtimes = []
    all_sampson_errors = []
    total_matches = 0

    for pair_number, (frame_i, frame_j) in enumerate(frame_pairs, start=1):
        print(f"\n[INFO] Pair {pair_number}/{len(frame_pairs)}: {frame_i} -> {frame_j}")
        output_file = remove_old_file("results_matches.csv")

        # Students get exactly the two left-camera images for this frame pair.
        img_i = dataset.stereo(frame_i)[0]
        img_j = dataset.stereo(frame_j)[0]

        # First: does the function run at all?
        start = time.perf_counter()
        try:
            solution.match_features(img_i, img_j)
            function_ran = True
            print("[PASS] match_features() ran without errors.")
        except Exception as exc:
            function_ran = False
            all_pairs_passed = False
            print("[FAIL] match_features() raised an error:")
            print(exc)
        runtime = time.perf_counter() - start
        runtimes.append(runtime)
        print(f"[INFO] Runtime: {runtime:.3f} seconds")

        # Second: did it create the expected CSV file?
        if not output_file.exists():
            all_pairs_passed = False
            print("[FAIL] match_features() must create results_matches.csv.")
            continue
        print("[PASS] Found results_matches.csv.")

        with output_file.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        # Third: is the CSV in the required format?
        if reader.fieldnames != expected_header:
            all_pairs_passed = False
            print("[FAIL] results_matches.csv must have this exact header:")
            print(",".join(expected_header))
            continue
        print("[PASS] results_matches.csv has the expected header.")

        if not rows:
            all_pairs_passed = False
            print("[FAIL] results_matches.csv must contain at least one match.")
            continue
        print(f"[PASS] results_matches.csv contains {len(rows)} match row(s).")

        rows_are_valid = True
        for expected_match_id, row in enumerate(rows):
            rows_are_valid &= row["match_id"] == str(expected_match_id)
            rows_are_valid &= is_number(row["u_i"])
            rows_are_valid &= is_number(row["v_i"])
            rows_are_valid &= is_number(row["u_j"])
            rows_are_valid &= is_number(row["v_j"])

        if rows_are_valid:
            print("[PASS] Match rows have sequential IDs and numeric coordinates.")
        else:
            all_pairs_passed = False
            print(
                "[FAIL] Match rows must use match_id values 0, 1, 2, ... and "
                "numeric u/v coordinates."
            )
            continue

        # Finally: give feedback on whether the matches obey the true epipolar
        # geometry for this KITTI frame pair.  This is feedback only in the
        # public checker; it is not currently used to assign marks.
        fundamental_matrix = trusted_fundamental_matrix(full_dataset, frame_i, frame_j)
        sampson_errors = sampson_errors_pixels(rows, fundamental_matrix)
        if not sampson_errors:
            print("[INFO] No valid Sampson errors could be calculated for this pair.")
            continue

        total_matches += len(rows)
        all_sampson_errors.extend(sampson_errors)
        pair_errors = np.asarray(sampson_errors)
        print("[INFO] Match geometry against trusted KITTI epipolar geometry:")
        print(f"       Median Sampson error: {np.median(pair_errors):.3f} px")
        print(f"       Mean Sampson error: {np.mean(pair_errors):.3f} px")
        print(f"       Matches below 1 px: {100.0 * np.mean(pair_errors <= 1.0):.1f}%")
        print(f"       Matches below 2 px: {100.0 * np.mean(pair_errors <= 2.0):.1f}%")
        print(f"       Matches below 5 px: {100.0 * np.mean(pair_errors <= 5.0):.1f}%")

    if runtimes:
        print("\n[INFO] Runtime summary for match_features():")
        print(f"       Mean runtime: {sum(runtimes) / len(runtimes):.3f} seconds")
        print(f"       Slowest pair: {max(runtimes):.3f} seconds")

    if all_sampson_errors:
        errors = np.asarray(all_sampson_errors)
        print("[INFO] Overall feature-match geometry summary:")
        print(f"       Evaluated frame pairs: {len(frame_pairs)}")
        print(f"       Evaluated matches: {total_matches}")
        print(f"       Median Sampson error: {np.median(errors):.3f} px")
        print(f"       Mean Sampson error: {np.mean(errors):.3f} px")
        print(f"       Matches below 1 px: {100.0 * np.mean(errors <= 1.0):.1f}%")
        print(f"       Matches below 2 px: {100.0 * np.mean(errors <= 2.0):.1f}%")
        print(f"       Matches below 5 px: {100.0 * np.mean(errors <= 5.0):.1f}%")

    return all_pairs_passed

# ====================================================================================
def check_estimate_relative_pose(dataset, full_dataset, frame_pairs) -> bool:
    """Check the student's ``estimate_relative_pose(dataset, frame_i, frame_j)``.

    The student function receives a restricted dataset object without ground
    truth.  This checker privately uses the full dataset only after the student
    function has finished, so it can report feedback-only RMSE values.
    """

    print("\n" + "=" * 80)
    print("Checking estimate_relative_pose(dataset, frame_i, frame_j)")
    print("=" * 80)
    print(f"[INFO] Checking {len(frame_pairs)} randomly selected frame pair(s).")

    expected_header = ["frame_i", "frame_j", "x", "y", "z", "roll", "pitch", "yaw"]
    numeric_columns = ["x", "y", "z", "roll", "pitch", "yaw"]
    all_pairs_passed = True
    runtimes = []
    translation_squared_errors = []
    rotation_squared_errors = []
    rotation_squared_errors_if_degrees = []
    translation_percentage_errors = []
    rotation_percentage_errors = []
    valid_pose_rows = []
    frame_gt_rotation_deg = []

    # The full KITTI dataset has ground truth. The restricted dataset passed to
    # student code below does not.
    #
    # KITTI ground truth is attached to the vehicle/IMU frame, but students are
    # estimating motion from the left camera images. Convert the ground-truth
    # relative motion into the left-camera frame before comparing.
    T_camera_imu = sm.SE3(
        full_dataset.camera_calibration(camera=2)["T_cam_imu"],
        check=False,
    )
    

    for pair_number, (frame_i, frame_j) in enumerate(frame_pairs, start=1):
        print(f"\n[INFO] Pair {pair_number}/{len(frame_pairs)}: {frame_i} -> {frame_j}")
        output_file = remove_old_file("results_relative_pose.csv")

        # First: call the student's function on this pair of frames.
        start = time.perf_counter()
        try:
            solution.estimate_relative_pose(dataset, frame_i, frame_j)
            function_ran = True
            print("[PASS] estimate_relative_pose() ran without errors.")
        except Exception as exc:
            function_ran = False
            all_pairs_passed = False
            print("[FAIL] estimate_relative_pose() raised an error:")
            print(exc)
        runtime = time.perf_counter() - start
        runtimes.append(runtime)
        print(f"[INFO] Runtime: {runtime:.3f} seconds")

        # Then check that the required one-row CSV file exists and is readable.
        if not output_file.exists():
            all_pairs_passed = False
            print("[FAIL] estimate_relative_pose() must create results_relative_pose.csv.")
            continue
        print("[PASS] Found results_relative_pose.csv.")

        with output_file.open(newline="", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)

        if reader.fieldnames != expected_header:
            all_pairs_passed = False
            print("[FAIL] results_relative_pose.csv must have this exact header:")
            print(",".join(expected_header))
            continue
        print("[PASS] results_relative_pose.csv has the expected header.")

        if len(rows) != 1:
            all_pairs_passed = False
            print("[FAIL] results_relative_pose.csv must contain exactly one data row.")
            continue
        row = rows[0]
        print("[PASS] results_relative_pose.csv contains one data row.")

        if row["frame_i"] != str(frame_i) or row["frame_j"] != str(frame_j):
            all_pairs_passed = False
            print(
                f"[FAIL] The row must describe frames {frame_i} -> {frame_j}, "
                f"but it describes {row['frame_i']} -> {row['frame_j']}."
            )
            continue

        if not all(is_number(row[column]) for column in numeric_columns):
            all_pairs_passed = False
            print("[FAIL] x, y, z, roll, pitch, and yaw must be numeric.")
            continue
        print("[PASS] Relative-pose row has numeric pose values.")

        # Convert the student's CSV row into an SE3 pose. Angles should be in
        # radians. We also build a second pose treating angles as degrees, only
        # to detect the common degrees-vs-radians mistake below.
        student_pose = pose_from_csv_row(row, angles_are_degrees=False)
        student_pose_if_degrees = pose_from_csv_row(row, angles_are_degrees=True)
        valid_pose_rows.append(row)

        # Convert KITTI's IMU/vehicle-frame ground truth into left-camera-frame
        # relative motion before comparing it to the student's camera motion.
        ground_truth_pose_i = full_dataset.ground_truth_pose(frame_i)
        ground_truth_pose_j = full_dataset.ground_truth_pose(frame_j)
        ground_truth_relative_pose_imu = ground_truth_pose_i.inv() @ ground_truth_pose_j
        ground_truth_relative_pose = (
            T_camera_imu @ ground_truth_relative_pose_imu @ T_camera_imu.inv()
        )

        pose_error = ground_truth_relative_pose.inv() @ student_pose
        translation_squared_errors.extend(value * value for value in pose_error.t)
        rotation_squared_errors.extend(
            value * value for value in pose_error.rpy(order="zyx")
        )

        # Percentage error compares the magnitude of the pose error to the
        # magnitude of the actual ground-truth motion for this frame pair.
        actual_translation = float(np.linalg.norm(ground_truth_relative_pose.t))
        translation_error = float(np.linalg.norm(pose_error.t))
        if actual_translation >= MIN_TRANSLATION_FOR_PERCENT_ERROR:
            translation_percentage_errors.append(
                100.0 * translation_error / actual_translation
            )

        actual_rotation = rotation_angle_rad(ground_truth_relative_pose)
        rotation_error = rotation_angle_rad(pose_error)
        frame_gt_rotation_deg.append(math.degrees(actual_rotation))  
        if actual_rotation >= MIN_ROTATION_FOR_PERCENT_ERROR:
            rotation_percentage_errors.append(100.0 * rotation_error / actual_rotation)

        pose_error_if_degrees = ground_truth_relative_pose.inv() @ student_pose_if_degrees
        rotation_squared_errors_if_degrees.extend(
            value * value for value in pose_error_if_degrees.rpy(order="zyx")
        )

    if runtimes:
        print("\n[INFO] Runtime summary for estimate_relative_pose():")
        print(f"       Mean runtime: {sum(runtimes) / len(runtimes):.3f} seconds")
        print(f"       Slowest pair: {max(runtimes):.3f} seconds")

    if translation_squared_errors and rotation_squared_errors:
        translation_rmse = math.sqrt(
            sum(translation_squared_errors) / len(translation_squared_errors)
        )
        rotation_rmse_rad = math.sqrt(
            sum(rotation_squared_errors) / len(rotation_squared_errors)
        )
        rotation_rmse_if_degrees_rad = math.sqrt(
            sum(rotation_squared_errors_if_degrees)
            / len(rotation_squared_errors_if_degrees)
        )
        rotation_rmse_deg = math.degrees(rotation_rmse_rad)

        print("[INFO] Relative-pose comparison against KITTI ground truth:")
        print(f"       Evaluated frame pairs: {len(translation_squared_errors) // 3}")
        print(f"       Translation RMSE: {translation_rmse:.6f} m")
        print(
            f"       Rotation RMSE: {rotation_rmse_rad:.6f} rad "
            f"({rotation_rmse_deg:.3f} deg)"
        )
        if translation_percentage_errors:
            print(
                "       Translation error: "
                f"{np.mean(translation_percentage_errors):.2f}% of actual motion "
                f"over {len(translation_percentage_errors)} moving pair(s)"
            )
        else:
            print(
                "       Translation error: not reported because no checked pair "
                f"moved at least {MIN_TRANSLATION_FOR_PERCENT_ERROR:.2f} m"
            )
        if rotation_percentage_errors:
            print(
                "       Rotation error: "
                f"{np.mean(rotation_percentage_errors):.2f}% of actual rotation "
                f"over {len(rotation_percentage_errors)} rotating pair(s)"
            )
        else:
            print(
                "       Rotation error: not reported because no checked pair "
                f"rotated at least {MIN_ROTATION_FOR_PERCENT_ERROR:.4f} rad"
            )
        warn_if_angles_look_like_degrees(
            "results_relative_pose.csv",
            valid_pose_rows,
            rotation_rmse_rad,
            rotation_rmse_if_degrees_rad,
        )

    return all_pairs_passed

# ====================================================================================
def check_visual_odometry(dataset, full_dataset) -> bool:
    """Check the student's ``visual_odometry(dataset)`` function.

    The expected output is one camera pose per frame in
    ``results_visual_odometry.csv``.  The public checker verifies the format and
    reports feedback-only absolute and frame-to-frame trajectory errors.
    """

    print("\n" + "=" * 80)
    print("Checking visual_odometry(dataset)")
    print("=" * 80)

    output_file = remove_old_file("results_visual_odometry.csv")
    remove_old_file(DIAGNOSTICS_CSV)
    remove_old_file(FIRST_FAILURE_NPZ)

    # The full VO function should process the whole dataset it is given.
    start = time.perf_counter()
    try:
        solution.visual_odometry(dataset)
        function_ran = True
        print("[PASS] visual_odometry() ran without errors.")
    except Exception as exc:
        function_ran = False
        print("[FAIL] visual_odometry() raised an error:")
        print(exc)
    runtime = time.perf_counter() - start
    print(f"[INFO] Runtime: {runtime:.3f} seconds")

    if not output_file.exists():
        print("[FAIL] visual_odometry() must create results_visual_odometry.csv.")
        return False
    print("[PASS] Found results_visual_odometry.csv.")

    # Read the trajectory CSV written by the student.
    with output_file.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)

    expected_header = ["frame", "x", "y", "z", "roll", "pitch", "yaw"]
    if reader.fieldnames != expected_header:
        print("[FAIL] results_visual_odometry.csv must have this exact header:")
        print(",".join(expected_header))
        return False
    print("[PASS] results_visual_odometry.csv has the expected header.")

    if not rows:
        print("[FAIL] results_visual_odometry.csv must contain at least one pose.")
        return False
    print(f"[PASS] results_visual_odometry.csv contains {len(rows)} pose row(s).")

    rows_are_valid = True
    for expected_frame, row in enumerate(rows):
        rows_are_valid &= row["frame"] == str(expected_frame)
        rows_are_valid &= is_number(row["x"])
        rows_are_valid &= is_number(row["y"])
        rows_are_valid &= is_number(row["z"])
        rows_are_valid &= is_number(row["roll"])
        rows_are_valid &= is_number(row["pitch"])
        rows_are_valid &= is_number(row["yaw"])

    if rows_are_valid:
        print("[PASS] Trajectory rows have sequential frame IDs and numeric poses.")
    else:
        print(
            "[FAIL] Trajectory rows must use frame values 0, 1, 2, ... and "
            "numeric x/y/z/roll/pitch/yaw values."
        )
        return False

    student_poses = []
    student_poses_if_degrees = []
    for row in rows:
        student_poses.append(pose_from_csv_row(row, angles_are_degrees=False))
        student_poses_if_degrees.append(pose_from_csv_row(row, angles_are_degrees=True))

    # KITTI ground truth is attached to the vehicle/IMU frame. Convert it into
    # the left-camera frame and normalize it so frame 0 is the identity pose,
    # matching the required CSV convention.
    T_camera_imu = sm.SE3(
        full_dataset.camera_calibration(camera=2)["T_cam_imu"],
        check=False,
    )
    ground_truth_camera_poses = []
    for frame in range(len(student_poses)):
        ground_truth_imu_pose = full_dataset.ground_truth_pose(frame)
        ground_truth_camera_poses.append(
            T_camera_imu @ ground_truth_imu_pose @ T_camera_imu.inv()
        )

    first_ground_truth_pose_inverse = ground_truth_camera_poses[0].inv()
    ground_truth_poses_relative_to_frame_zero = [
        first_ground_truth_pose_inverse @ pose for pose in ground_truth_camera_poses
    ]

    comparison_plot = "results_visual_odometry_comparison.png"
    plot_trajectory_comparison(
        student_poses,
        ground_truth_poses_relative_to_frame_zero,
        comparison_plot,
    )
    print(f"[INFO] Wrote trajectory comparison plot to {comparison_plot}.")

    # Absolute trajectory error: compare every submitted pose to the
    # corresponding ground-truth camera pose relative to frame 0.
    translation_squared_errors = []
    rotation_squared_errors = []
    rotation_squared_errors_if_degrees = []
    position_errors = []  # ||p_est - p_gt|| per frame, for ATE / final error / drift
    for student_pose, student_pose_if_degrees, ground_truth_pose in zip(
        student_poses,
        student_poses_if_degrees,
        ground_truth_poses_relative_to_frame_zero,
    ):
        pose_error = ground_truth_pose.inv() @ student_pose
        translation_squared_errors.extend(value * value for value in pose_error.t)
        rotation_squared_errors.extend(
            value * value for value in pose_error.rpy(order="zyx")
        )
        position_errors.append(float(np.linalg.norm(pose_error.t)))

        pose_error_if_degrees = ground_truth_pose.inv() @ student_pose_if_degrees
        rotation_squared_errors_if_degrees.extend(
            value * value for value in pose_error_if_degrees.rpy(order="zyx")
        )

    translation_rmse = math.sqrt(
        sum(translation_squared_errors) / len(translation_squared_errors)
    )
    rotation_rmse_rad = math.sqrt(
        sum(rotation_squared_errors) / len(rotation_squared_errors)
    )
    rotation_rmse_if_degrees_rad = math.sqrt(
        sum(rotation_squared_errors_if_degrees)
        / len(rotation_squared_errors_if_degrees)
    )
    rotation_rmse_deg = math.degrees(rotation_rmse_rad)

    print("[INFO] Trajectory comparison against KITTI ground truth:")
    print(f"       Evaluated frames: {len(student_poses)}")
    print(f"       Translation RMSE: {translation_rmse:.6f} m")
    print(
        f"       Rotation RMSE: {rotation_rmse_rad:.6f} rad "
        f"({rotation_rmse_deg:.3f} deg)"
    )
    warn_if_angles_look_like_degrees(
        "results_visual_odometry.csv",
        rows,
        rotation_rmse_rad,
        rotation_rmse_if_degrees_rad,
    )

    if len(student_poses) >= 2:
        # Relative trajectory error: compare each submitted frame-to-frame
        # motion against the corresponding ground-truth frame-to-frame motion.
        relative_translation_squared_errors = []
        relative_rotation_squared_errors = []
        relative_translation_percentage_errors = []
        relative_rotation_percentage_errors = []
        frame_rotation_errors_deg = []    # one value per frame pair (k -> k+1)
        frame_translation_errors_m = []
        frame_heading_errors_deg = []     # signed, about camera y (vertical) axis
        for frame in range(len(student_poses) - 1):
            student_relative_pose = student_poses[frame].inv() @ student_poses[frame + 1]
            ground_truth_relative_pose = (
                ground_truth_poses_relative_to_frame_zero[frame].inv()
                @ ground_truth_poses_relative_to_frame_zero[frame + 1]
            )
            relative_pose_error = ground_truth_relative_pose.inv() @ student_relative_pose
            relative_translation_squared_errors.extend(
                value * value for value in relative_pose_error.t
            )
            relative_rotation_squared_errors.extend(
                value * value for value in relative_pose_error.rpy(order="zyx")
            )

            actual_translation = float(np.linalg.norm(ground_truth_relative_pose.t))
            translation_error = float(np.linalg.norm(relative_pose_error.t))
            if actual_translation >= MIN_TRANSLATION_FOR_PERCENT_ERROR:
                relative_translation_percentage_errors.append(
                    100.0 * translation_error / actual_translation
                )

            actual_rotation = rotation_angle_rad(ground_truth_relative_pose)
            rotation_error = rotation_angle_rad(relative_pose_error)
            if actual_rotation >= MIN_ROTATION_FOR_PERCENT_ERROR:
                relative_rotation_percentage_errors.append(
                    100.0 * rotation_error / actual_rotation
                )

            frame_rotation_errors_deg.append(math.degrees(rotation_error))
            frame_translation_errors_m.append(translation_error)
            theta, axis = relative_pose_error.angvec()
            frame_heading_errors_deg.append(math.degrees(theta * axis[1]))

        relative_translation_rmse = math.sqrt(
            sum(relative_translation_squared_errors)
            / len(relative_translation_squared_errors)
        )
        relative_rotation_rmse_rad = math.sqrt(
            sum(relative_rotation_squared_errors)
            / len(relative_rotation_squared_errors)
        )
        relative_rotation_rmse_deg = math.degrees(relative_rotation_rmse_rad)

        # Median Values
        relative_translation_rmse_med = math.sqrt(
            statistics.median(relative_translation_squared_errors)
        )
        relative_rotation_rmse_rad_med = math.sqrt(
            statistics.median(relative_rotation_squared_errors)
        )
        relative_rotation_rmse_deg_med = math.degrees(relative_rotation_rmse_rad_med)

        print("[INFO] Average frame-to-frame relative pose error:")
        print(f"       Evaluated frame pairs: {len(student_poses) - 1}")
        print(f"       Translation RMSE: {relative_translation_rmse:.6f} m")
        print(
            f"       Rotation RMSE: {relative_rotation_rmse_rad:.6f} rad "
            f"({relative_rotation_rmse_deg:.3f} deg)"
        )
        if relative_translation_percentage_errors:
            print(
                "       Translation error: "
                f"{np.mean(relative_translation_percentage_errors):.2f}% "
                "of actual frame-to-frame motion "
                f"over {len(relative_translation_percentage_errors)} moving pair(s)"
            )
        else:
            print(
                "       Translation error: not reported because no frame pair "
                f"moved at least {MIN_TRANSLATION_FOR_PERCENT_ERROR:.2f} m"
            )
        if relative_rotation_percentage_errors:
            print(
                "       Rotation error: "
                f"{np.mean(relative_rotation_percentage_errors):.2f}% "
                "of actual frame-to-frame rotation "
                f"over {len(relative_rotation_percentage_errors)} rotating pair(s)"
            )
        else:
            print(
                "       Rotation error: not reported because no frame pair "
                f"rotated at least {MIN_ROTATION_FOR_PERCENT_ERROR:.4f} rad"
            )
        print("[INFO] Median frame-to-frame relative pose error:")
        print(f"       Translation RMSE: {relative_translation_rmse_med:.6f} m")
        print(
            f"       Rotation RMSE: {relative_rotation_rmse_rad_med:.6f} rad "
            f"({relative_rotation_rmse_deg_med:.3f} deg)"
        )

        # ---------------------------------------------------------------
        # Failure analysis (report table + figures)
        # ---------------------------------------------------------------
        diagnostics = load_diagnostics()
        rejected = [r for r in diagnostics if r["accepted"] == "0"]
        rejected_frames = [int(r["frame_j"]) for r in rejected
                           if int(r["frame_j"]) < len(student_poses)]

        rot_err = np.asarray(frame_rotation_errors_deg)
        trans_err = np.asarray(frame_translation_errors_m)
        heading = np.asarray(frame_heading_errors_deg)
        pos_err = np.asarray(position_errors)
        path_length = sum(
            float(np.linalg.norm(
                ground_truth_poses_relative_to_frame_zero[k + 1].t
                - ground_truth_poses_relative_to_frame_zero[k].t))
            for k in range(len(student_poses) - 1)
        )
        start, end, growth = worst_region(pos_err)
        worst_pair = int(np.argmax(rot_err))
        rot_spikes = spike_frames(rot_err)
        trans_spikes = spike_frames(trans_err)

        print("[INFO] Failure analysis (values for the report table):")
        print(f"       Frames processed: {len(student_poses)}")
        if diagnostics:
            print(f"       Rejected frame-to-frame estimates: {len(rejected)} "
                  f"of {len(diagnostics)}")
        else:
            print(f"       Rejected estimates: unknown ({DIAGNOSTICS_CSV} not written)")
        print(f"       Final trajectory error: {pos_err[-1]:.3f} m "
              f"({100 * pos_err[-1] / max(path_length, 1e-9):.2f}% of "
              f"{path_length:.1f} m path)")
        print(f"       ATE (RMSE of position error): "
              f"{math.sqrt(np.mean(pos_err ** 2)):.3f} m")
        print(f"       Median frame-to-frame rotation error: {np.median(rot_err):.4f} deg")
        print(f"       Median frame-to-frame translation error: "
              f"{np.median(trans_err):.4f} m")
        print(f"       Worst single pair (rotation): {worst_pair} -> {worst_pair + 1} "
              f"({rot_err[worst_pair]:.3f} deg)")
        print(f"       Fastest error growth: frames {start}-{end} "
              f"(+{growth:.3f} m)")
        print(f"       Error spikes (> median + 5 MAD): rotation at pairs "
              f"{rot_spikes.tolist()}, translation at pairs {trans_spikes.tolist()}")
        print(f"       Signed heading error: mean {heading.mean():+.4f} deg, "
              f"std {heading.std():.4f} deg "
              "(|mean| large vs std suggests systematic bias)")

        silent = sorted(set(rot_spikes.tolist()) | set(trans_spikes.tolist()))
        silent = [k for k in silent if (k + 1) not in rejected_frames]
        if silent:
            print(f"       Spikes NOT rejected by the pipeline (silent failures): "
                  f"pairs {silent}")

        plot_path = plot_trajectory_colored(
            student_poses,
            ground_truth_poses_relative_to_frame_zero,
            rot_err,
            rejected_frames,
        )
        print(f"[INFO] Wrote error-colored trajectory to {plot_path}.")

        # Visualise the worst pair: re-run the student's estimator on it to get the matches
        try:
            _, pair_info = solution.estimate_relative_pose(
                dataset, worst_pair, worst_pair + 1, return_info=True
            )
            pair_plot = show_frame_pair(
                full_dataset, worst_pair, worst_pair + 1, pair_info,
                title_note=f"rotation error {rot_err[worst_pair]:.3f} deg",
            )
            print(f"[INFO] Wrote worst frame pair to {pair_plot}.")

            frames, blur = blur_profile(full_dataset, len(student_poses))
            k = int(np.flatnonzero(frames == worst_pair)[0])
            print(f"       Blur score at frame {worst_pair}: {blur[k]:.0f} "
                  f"(sequence median {np.median(blur):.0f}, "
                  f"min {blur.min():.0f} at frame {frames[np.argmin(blur)]})")
        except Exception as exc:
            print(f"[INFO] Could not plot worst frame pair: {exc}")

        failure_plot = show_first_failure(full_dataset)
        if failure_plot:
            print(f"[INFO] Wrote first rejected frame pair to {failure_plot}.")
        else:
            print(f"[INFO] No rejected frame pair ({FIRST_FAILURE_NPZ} not written).")

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(frame_gt_rotation_deg, frame_heading_errors_deg, s=8)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xlabel("True frame-to-frame rotation [deg]")
        ax.set_ylabel("Signed heading error [deg]")
        ax.set_title("Heading error vs. turn rate")
        fig.tight_layout()
        fig.savefig("results_vo_rot_err_vs_rotation.png", dpi=200)
        plt.close(fig)
        print("[INFO] Wrote heading error vs. rotation to results_vo_rot_err_vs_rotation.png.")

    return function_ran



# ====================================================================================
# ====================================================================================
# ====================================================================================
def main() -> None:
    """Parse command-line options, load KITTI, and run the three checks."""

    parser = argparse.ArgumentParser(
        description="Check src/student_solution.py on a small KITTI example.",
    )
    parser.add_argument(
        "--sequence",
        default="2011_09_26_drive_0035",
        help="KITTI sequence name to load.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Optional data directory containing or receiving the KITTI data.",
    )
    parser.add_argument("--frame-i", type=int, default=0)
    parser.add_argument("--frame-j", type=int, default=1)
    parser.add_argument(
        "--frame-pairs",
        dest="frame_pairs",
        type=int,
        default=10,
        help=(
            "Number of random frame pairs to check with "
            "match_features() and estimate_relative_pose()."
        ),
    )
    parser.add_argument(
        "--skip-frames",
        type=int,
        default=1,
        help=(
            "Frame gap for sampled pairs. The default 1 checks adjacent pairs "
            "i -> i+1; --skip-frames 5 checks pairs i -> i+5."
        ),
    )
    parser.add_argument(
        "--relative-pose-pairs",
        dest="frame_pairs",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=0,
        help="Random seed used when selecting frame pairs.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help=(
            "Optional quick-test limit. If set, visual_odometry() only sees "
            "the first N frames."
        ),
    )
    parser.add_argument(
        "--solution-module",
        default="student_solution_eval",
        help=(
            "Python module containing match_features(), estimate_relative_pose(), "
            "and visual_odometry(). Defaults to student_solution."
        ),
    )
    parser.add_argument(
        "--check-matches",
        action="store_true",
        help="Run only/also the match_features() check.",
    )
    parser.add_argument(
        "--check-relative-pose",
        action="store_true",
        help="Run only/also the estimate_relative_pose() check.",
    )
    parser.add_argument(
        "--check-visual-odometry",
        action="store_true",
        help="Run only/also the visual_odometry() check.",
    )
    parser.add_argument(
        "--report-file",
        default="check_student_solution_report.txt",
        help=(
            "Write the same checker output to this text file. "
            "Use --no-report-file to disable this."
        ),
    )
    parser.add_argument(
        "--no-report-file",
        action="store_true",
        help="Do not write a text report file.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable coloured terminal output.",
    )
    args = parser.parse_args()

    selected_checks = {
        "match_features": args.check_matches,
        "estimate_relative_pose": args.check_relative_pose,
        "visual_odometry": args.check_visual_odometry,
    }
    if not any(selected_checks.values()):
        selected_checks = {name: True for name in selected_checks}

    # Replace stdout with a small wrapper that adds icons/colour and optionally
    # writes the same output to a report file.
    original_stdout = sys.stdout
    report_file = None
    if not args.no_report_file:
        report_path = Path(args.report_file)
        report_file = report_path.open("w", encoding="utf-8")
    use_color = original_stdout.isatty() and not args.no_color
    sys.stdout = TeeOutput(original_stdout, report_file, use_color)

    global solution
    solution = importlib.import_module(args.solution_module)

    if args.data_dir is not None:
        os.environ["ENN583_DATA_DIR"] = str(Path(args.data_dir).expanduser().resolve())

    print(f"Using solution module: {args.solution_module}")
    print("Loading KITTI data...")
    full_dataset = kitti.load_kitti_dataset(name=args.sequence)

    # Student code receives this restricted view. It contains images,
    # calibration, and sequence length, but not ground-truth poses.
    dataset = full_dataset.student_view(include_ground_truth=False)

    if args.max_frames is not None:
        dataset = FirstNFrames(dataset, args.max_frames)
        full_dataset = FirstNFrames(full_dataset, args.max_frames)

    if not (0 <= args.frame_i < len(dataset)) or not (0 <= args.frame_j < len(dataset)):
        raise SystemExit(
            f"Frame indices must be between 0 and {len(dataset) - 1}; "
            f"got {args.frame_i} and {args.frame_j}."
        )
    checks_need_frame_pairs = (
        selected_checks["match_features"] or selected_checks["estimate_relative_pose"]
    )
    if checks_need_frame_pairs:
        if args.frame_pairs < 1:
            raise SystemExit("--frame-pairs must be at least 1.")
        if args.skip_frames < 1:
            raise SystemExit("--skip-frames must be at least 1.")
        if len(dataset) <= args.skip_frames:
            raise SystemExit(
                f"--skip-frames {args.skip_frames} is too large for a dataset with "
                f"{len(dataset)} frame(s)."
            )

    img_i = dataset.stereo(args.frame_i)[0]
    img_j = dataset.stereo(args.frame_j)[0]
    selected_frame_pairs = []
    if checks_need_frame_pairs:
        selected_frame_pairs = choose_random_frame_pairs(
            len(dataset),
            args.frame_pairs,
            args.random_seed,
            args.skip_frames,
        )

    # These first two images are printed only to make the example concrete for
    # students. The feature-matching and relative-pose checks below use the
    # randomly selected frame pairs.
    print(f"Loaded sequence: {args.sequence}")
    print(f"Number of frames: {len(dataset)}")
    print(f"Example image frames: {args.frame_i} and {args.frame_j}")
    print(
        "Selected checks: "
        + ", ".join(name for name, is_selected in selected_checks.items() if is_selected)
    )
    if checks_need_frame_pairs:
        print("Using these frame pairs for the selected pair-based check(s):")
        print(
            "    "
            + ", ".join(f"{frame_i}->{frame_j}" for frame_i, frame_j in selected_frame_pairs)
        )
        print(f"Frame skip: {args.skip_frames}")
    print(f"Random seed: {args.random_seed}")
    print(f"Image {args.frame_i} shape: {img_i.shape}")
    print(f"Image {args.frame_j} shape: {img_j.shape}")
    print("\nThis checker does not assign marks.")
    print("It shows whether your functions run and whether the CSV files look right.")

    results = {}
    if selected_checks["match_features"]:
        results["match_features"] = check_match_features(
            dataset,
            full_dataset,
            selected_frame_pairs,
        )
    if selected_checks["estimate_relative_pose"]:
        results["estimate_relative_pose"] = check_estimate_relative_pose(
            dataset,
            full_dataset,
            selected_frame_pairs,
        )
    if selected_checks["visual_odometry"]:
        results["visual_odometry"] = check_visual_odometry(dataset, full_dataset)

    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    for check_name, passed in results.items():
        print(f"{check_name}(): {'PASS' if passed else 'FAIL'}")

    if not all(results.values()):
        if report_file is not None:
            print(f"\nReport written to: {Path(args.report_file).resolve()}")
            sys.stdout.flush()
            report_file.close()
        sys.stdout = original_stdout
        raise SystemExit(1)

    if report_file is not None:
        print(f"\nReport written to: {Path(args.report_file).resolve()}")
        sys.stdout.flush()
        report_file.close()
    sys.stdout = original_stdout


if __name__ == "__main__":
    main()
