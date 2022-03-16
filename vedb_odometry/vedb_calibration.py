#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration of VEDB T265
@author: brianszekely
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pupil_recording_interface as pri
import rigid_body_motion as rbm
import vedb_store
import os

#methods should be written from longest to shortest in length
class vedbCalibration():
    def __init__(self, name=None):
        self.name = name 
        self.odometry = None
        self.accel = None
        
    def set_odometry(self, date_lookup):
        dbi = vedb_store.docdb.getclient()
        sessions = dbi.query('+date', type="Session")
        # print(sessions.sort(key=myFunc))
        for i in range(len(sessions)):
            temp_date = sessions[i]['date']
            if temp_date == date_lookup:
                temp_path = sessions[i].paths["odometry"][1]
                temp_path = '/'.join(temp_path.split('/')[0:-1])
                self.odometry = pri.load_dataset(temp_path,
                                                 odometry='recording',
                                                 cache=False)
                self.accel = pri.load_dataset(temp_path,
                                              accel="recording", 
                                              cache=False)

    def set_calibration(self):
        sine_wave_samps = np.linspace(0, 
                                       len(self.odometry.position[:,0].values), 
                                       int(1*len(self.odometry.position[:,0].values)), endpoint=False)
        amplitude = np.sin(sine_wave_samps)
        plt.plot(self.odometry.time.values, amplitude)
        plt.show()
    
    def t265_to_head_trans(self):
        pass
    
    def calc_gait_variablity(self):
        pass
    
    def calc_inst_force(self):
        pass
    
    def plot(self, accel, odo):
        pass
        
    def get_accel(self):
        return self.accel
    
    def get_odometry(self):
        return self.odometry
    
        
        
if __name__ == "__main__":
    odo = vedbCalibration()
    odo.set_odometry('2021_05_21_13_56_00')
    odo.set_calibration()

