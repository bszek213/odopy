#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for vedb odo from folder and from cloud
@author: brianszekely
"""
# import sys
from os import getcwd, path
from odopy import headCalibrate
import matplotlib.pyplot as plt
import numpy as np

def quat_to_euler(quaternions):
    """
    Convert quaternions to Euler angles.
    
    Roll (φ) = atan2(2(w x + y z), 1 - 2(x^2 + y^2))
    Pitch (θ) = asin(2(w y - z x))
    Yaw (ψ) = atan2(2(w z + x y), 1 - 2(y^2 + z^2))
    """
    q_w, q_x, q_y, q_z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
    #Roll
    roll = np.degrees(np.arctan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2)))
    #Pitch
    pitch = np.degrees(np.arcsin(2 * (q_w * q_y - q_z * q_x)))
    #Yaw
    yaw = np.degrees(np.arctan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2)))

    return np.column_stack((roll, pitch, yaw))


# def extract_unzip(file):
#     zip_url = f"https://osf.io/pcxsj/download/{file}.zip"
#     extract_dir = "/media/bszekely/BrianDrive"
#     # Download the zip file
#     wget -O - https://osf.io/pcxsj/download/2022_08_10_15_20_14.zip
#     download_command = ["wget", "-P", extract_dir, zip_url]
#     subprocess.run(download_command)
#     zip_file = os.path.join(extract_dir, f"{file}.zip")
#     extract_command = ["unzip", zip_file, "-d", extract_dir]
#     subprocess.run(extract_command)
#     input()

if __name__ == "__main__":
    odo = headCalibrate.headCalibrate()
    curr_dir = getcwd()
    folder = path.join(curr_dir, 'test_data')
    # folder = '/media/bszekely/BrianDrive/brian_walk_test'
    # folder = "/media/bszekely/BrianDrive/2022_02_09_13_40_13_test_walk_session"
    folder = '/home/bszekely/Desktop/projects/eye_pipeline/002_2022_12_12_10_56_40'
    #vedb takes
    # folder = "/media/bszekely/BrianDrive/2022_08_10_13_39_57" #VEDB WALKING TAKE
    odo.set_odometry_local(folder)
    odo_data = odo.get_odometry()
    # print(odo_data.orientation.values)
    # np.save('orientation_values_walk.npy',odo_data.orientation.values)
    odo.start_end_plot()
    odo.t265_to_head_trans()
    odo.calc_head_orientation()
    rbm_tree = odo.get_rbm()
    # head_roll, head_pitch, head_yaw = odo.get_head_orientation()
    # out = quat_to_euler(odo.get_odometry().orientation.values)
    # plt.plot(out,linewidth=3,marker='*')
    # plt.plot(odo.get_calibrated_odo().ang_pos[0:100,1].values,linewidth=3)
    odo.plot()
