#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for vedb odo from folder and from cloud
@author: brianszekely
"""
import sys
from os import getcwd, path
from vedb_odometry import vedb_calibration
if __name__ == "__main__":
    odo = vedb_calibration.vedbCalibration()
    curr_dir = getcwd()
    folder = path.join(curr_dir, 'test_data')
    odo.set_odometry_local(folder)
    # odo.set_odometry_cloud('2021_05_21_13_56_00')
    odo.start_end_plot()
    odo.t265_to_head_trans()
    odo.plot()
