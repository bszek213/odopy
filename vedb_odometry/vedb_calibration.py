#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration of VEDB T265
@author: brianszekely, christiansinnott
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

        # Pull annotated calibration segment times from yaml (LEGACY FROM OLD ANALYSIS, NEEDS CHANGED)
        with open(folder / subject / "meta.yaml") as f:
            times = yaml.safe_load(f.read())

        for annotation in ("pre_calib", "post_calib", "experiment"):
            if annotation in times:
                times[annotation] = {
                    k: pd.to_datetime(times["date"]) + pd.to_timedelta(v)
                    for k, v in times[annotation].items()
                }

        for annotation in ("re_calib", "exclude"):
            if annotation in times:
                for idx, segment in enumerate(times[annotation]):
                    times[annotation][idx] = {
                        k: pd.to_datetime(times["date"]) + pd.to_timedelta(v)
                        for k, v in times[annotation][idx].items()
                    }

        # Set-up Reference Frames
        R_WORLD_ODOM = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        R_IMU_ODOM = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        rbm.register_frame("world", update=True)

        rbm.ReferenceFrame.from_rotation_matrix(R_WORLD_ODOM, name="t265_world", parent="world").register(update=True)

        rbm.ReferenceFrame.from_dataset(self.odometry, "position", "orientation", "time", parent="t265_world",
                                        name="t265_odom")

        rbm.ReferenceFrame.from_rotation_matrix(R_IMU_ODOM, name="t265_imu", parent="t265_odom").register(update=True)

        rbm.ReferenceFrame.from_rotation_matrix(R_WORLD_ODOM, name="t265_vestibular", parent="t265_odom", inverse=True
                                                ).register(update=True)

        # Define first calibrated frame using calibration segments identified in set_calibration method
        segments = [times["pre_calib"]] + [times["re_calib"]]
        rotations = np.zeros(len(segments), 4)
        timestamps = np.zeros(len(segments), dtype=pd.Timestamp)

        for idx, calib_segment in enumerate(segments):

            omega = rbm.transform_vectors(self.odometry.angular_velocity, outof="t265_world", into="t265_vestibular")
            omega_pitch = omega.sel(time=slice(calib_segment["pitch_start"], calib_segment["pitch_end"]))
            omega_yaw = omega.sel(time=slice(calib_segment["yaw_start"], calib_segment["yaw_end"]))

            omega_pitch_target = xr.zeros_like(omega_pitch)
            omega_pitch_target[:, 1] = np.sign(omega_pitch[:, 1]) * omega_pitch.reduce(np.linalg.norm, "cartesian_axis")

            omega_yaw_target = xr.zeros_like(omega_yaw)
            omega_yaw_target[:, 2] = np.sign(omega_yaw[:, 2]) * omega_yaw.reduce(np.linalg.norm, "cartesian_axis")

            rotations[idx, :] = rbm.best_fit_rotation(xr.concat((omega_pitch, omega_yaw), "time"),
                                                      xr.concat((omega_pitch_target, omega_yaw_target), "time"))

            timestamps[idx] = min(calib_segment["pitch_start"], calib_segment["yaw_start"])

        # Construct discrete reference frame

        pass
    
    def calc_gait_variability(self):
        pass
    
    def calc_inst_force(self):
        pass

    def calc_head_orientation(self):
        x = self.accel.sel(cartesian_axis="x")
        y = self.accel.sel(cartesian_axis="y")
        z = self.accel.sel(cartesian_axis="z")
        norm = self.accel.reduce(np.linalg.norm, "cartesian_axis")

        head_roll = np.rad2deg(np.arctan(y/z))
        head_pitch = np.rad2deg(-np.arctan(x/norm))

        pass

    def calc_heading(self):
        x = self.accel.linear_vel.sel(cartesian_axis="x")
        y = self.accel.linear_vel.sel(cartesian_axis="y")
        z = self.accel.linear_vel.sel(cartesian_axis="z")
        norm = self.accel.linear_vel.reduce(np.linalg.norm, "cartesian_axis")

        heading_azimuth = np.rad2deg(-np.arctan2(y, x))
        heading_elevation = np.rad2deg(np.arcsin(z/norm))

        pass
    
    def plot(self, accel, odo):
        pass

    def get_head_orientation(self):
        return self.head_roll, self.head_pitch

    def get_heading(self):
        return self.heading_azimuth, self.heading_elevation
        
    def get_accel(self):
        return self.accel
    
    def get_odometry(self):
        return self.odometry
    
        
        
if __name__ == "__main__":
    odo = vedbCalibration()
    odo.set_odometry('2021_05_21_13_56_00')
    odo.set_calibration()

