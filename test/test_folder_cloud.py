#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test for vedb odo from folder and from cloud
@author: brianszekely
"""
import sys
sys.path.insert(0, '/home/brianszekely/Desktop/ProjectsResearch/vedb/odometry/')
from vedb_odometry import vedb_calibration

if __name__ == "__main__":
    odo = vedb_calibration.vedbCalibration()
    odo.set_odometry_folder('/media/brianszekely/TOSHIBA EXT/test_odo/')
    # odo.set_odometry_cloud('2021_05_21_13_56_00')
    odo.start_end_plot()