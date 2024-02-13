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

if __name__ == "__main__":
    odo = headCalibrate.headCalibrate()
    curr_dir = getcwd()
    folder = path.join(curr_dir, 'test_data')
    odo.set_odometry_local(folder)
    odo.start_end_plot()
    odo.t265_to_head_trans()
    odo.calc_head_orientation()
    # head_roll, head_pitch, head_yaw = odo.get_head_orientation()
    # out = quat_to_euler(odo.get_odometry().orientation.values)
    # plt.plot(out,linewidth=3,marker='*')
    # plt.plot(odo.get_calibrated_odo().ang_pos[0:100,1].values,linewidth=3)
    odo.plot()
