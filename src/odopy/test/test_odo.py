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
if __name__ == "__main__":
    odo = headCalibrate.headCalibrate()
    curr_dir = getcwd()
    folder = path.join(curr_dir, 'test_data')
    odo.set_odometry_local(folder)
    odo.start_end_plot()
    odo.t265_to_head_trans()
    odo.calc_head_orientation()
    head_roll, head_pitch, head_yaw = odo.get_head_orientation()
    odo.plot()
