#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calibration of VEDB T265
@author: brianszekely, christiansinnott
"""
# from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import pupil_recording_interface as pri
import rigid_body_motion as rbm
import vedb_store
import os
from scipy.signal import savgol_filter
# import file_io
from datetime import datetime
import plotly.graph_objects as go

# def show(fig):
#     import io
#     import plotly.io as pio
#     from PIL import Image
#     buf = io.BytesIO()
#     pio.write_image(fig, buf)
#     img = Image.open(buf)
#     img.show() 
#methods should be written from longest to shortest in length
class vedbCalibration():

    def __init__(self, name=None):
        self.name = name 
        self.odometry = None
        self.accel = None
        self.fps = 200 
    
    def set_odometry_local(self, folder):
        isdir = os.path.isdir(folder) 
        if isdir == False:
            print('Folder does not exist. Check the path')
        elif isdir == True:
            # accel_local = os.path.join(folder, 'accel.pldata')
            # odo_local = os.path.join(folder, 'odometry.pldata')
            self.odometry = pri.load_dataset(folder,odometry='recording',cache=False)
            self.accel = pri.load_dataset(folder,accel='recording',cache=False)
            
    def set_odometry_cloud(self, date_lookup):
        pass
        # dbi = vedb_store.docdb.getclient()
        # sessions = dbi.query('+date', type="Session")
        # # print(sessions.sort(key=myFunc))
        # for i in range(len(sessions)):
        #     temp_date = sessions[i]['date']
        #     if temp_date == date_lookup:
        #         temp_path = sessions[i].paths["odometry"][1]
        #         temp_path = '/'.join(temp_path.split('/')[0:-1])
        #         self.odometry = pri.load_dataset(temp_path,
        #                                          odometry='recording',
        #                                          cache=False)
        #         self.accel = pri.load_dataset(temp_path,
        #                                       accel="recording", 
        #                                       cache=False)

    def set_calibration(self):
        sine_wave_samps = np.linspace(0, 
                                       len(self.odometry.position[:,0].values), 
                                       int(1*len(self.odometry.position[:,0].values)), endpoint=False)
        amplitude = np.sin(sine_wave_samps)
        plt.plot(self.odometry.time.values, amplitude)
        plt.show()
        
    def start_end_plot(self):
        #smooth data for better viewing purposes
        pitch_vel = savgol_filter(self.odometry.angular_velocity[:, 0], 201, 2)
        yaw_vel = savgol_filter(self.odometry.angular_velocity[:, 1], 201, 2)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=pitch_vel,
            name="pitch"       # this sets its legend entry
            ))
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=yaw_vel,
            name="yaw"       # this sets its legend entry
            ))
        fig.update_layout(
            title="Angular velocity odometry",
            xaxis_title="Time stamps (datetime)",
            yaxis_title="Angular Velocity (radians/second)",
            )
        fig.show()
        # show(fig)
        
        pitch_start = input('Pitch Timestamp Start (HH:mm:ss format): ')
        pitch_end = input('Pitch Timestamp End (HH:mm:ss format): ')
        yaw_start = input('Yaw Timestamp Start (HH:mm:ss format): ')
        yaw_end = input('Yaw Timestamp End (HH:mm:ss format): ')
        df_time = pd.Series(self.odometry.time[0].values)
        self.pitch_start = datetime.combine(df_time.dt.date.values[0],
                                     datetime.strptime(pitch_start, '%H:%M:%S').time())
        self.pitch_end = datetime.combine(df_time.dt.date.values[0],
                                     datetime.strptime(pitch_end, '%H:%M:%S').time())
        self.yaw_start = datetime.combine(df_time.dt.date.values[0],
                                     datetime.strptime(yaw_start, '%H:%M:%S').time())
        self.yaw_end = datetime.combine(df_time.dt.date.values[0],
                                     datetime.strptime(yaw_end, '%H:%M:%S').time())
        self.times = {'calibration':
            {'pitch_start': self.pitch_start,
             'pitch_end': self.pitch_end,
             'yaw_start': self.yaw_start,
             'yaw_end': self.yaw_end
             }}

    def t265_to_head_trans(self):

        # Pull annotated calibration segment times from yaml (LEGACY FROM OLD ANALYSIS, NEEDS CHANGED)
        # with open(folder / subject / "meta.yaml") as f:
        #     times = yaml.safe_load(f.read())

        # for annotation in ("pre_calib", "post_calib", "experiment"):
        #     if annotation in times:
        #         times[annotation] = {
        #             k: pd.to_datetime(times["date"]) + pd.to_timedelta(v)
        #             for k, v in times[annotation].items()
        #         }

        # for annotation in ("re_calib", "exclude"):
        #     if annotation in times:
        #         for idx, segment in enumerate(times[annotation]):
        #             times[annotation][idx] = {
        #                 k: pd.to_datetime(times["date"]) + pd.to_timedelta(v)
        #                 for k, v in times[annotation][idx].items()
        #             }
        # Extract calibration segment times:
        # Set-up Reference Frames
        R_WORLD_ODOM = np.array([[0, 0, -1], [-1, 0, 0], [0, 1, 0]])
        R_IMU_ODOM = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

        rbm.register_frame("world", update=True)

        rbm.ReferenceFrame.from_rotation_matrix(R_WORLD_ODOM, name="t265_world", parent="world").register(update=True)

        rbm.register_frame(
            "t265_odom",
            translation=self.odometry.position.values,
            rotation=self.odometry.orientation.values,
            timestamps=self.odometry.time.values,
            parent="t265_world",
            update=True,
        )
        rbm.ReferenceFrame.from_rotation_matrix(R_IMU_ODOM, name="t265_imu", parent="t265_odom").register(update=True)
        rbm.ReferenceFrame.from_rotation_matrix(R_WORLD_ODOM, name="t265_vestibular", parent="t265_odom", inverse=True
                                                ).register(update=True)
        print(rbm.render_tree("world"))
        # Define first calibrated frame using calibration segments identified in set_calibration method
        # segments = [times["pre_calib"]] + [times["re_calib"]]
        segments = self.times["calibration"]
        rotations = np.zeros([len(segments), 4]) #Original has 4 - yaw, pitch, Reid's, hallway
        timestamps = np.zeros(len(segments), dtype=pd.Timestamp)
        
        for idx, seg in self.times["calibration"].items():
            self.times["calibration"][idx] = np.datetime64(seg)
        omega = rbm.transform_vectors(self.odometry.angular_velocity, outof="t265_world", into="t265_vestibular")
        omega_pitch = omega.sel(time=slice(self.times["calibration"]["pitch_start"], 
                                           self.times["calibration"]["pitch_end"]))
        omega_yaw = omega.sel(time=slice(self.times["calibration"]["yaw_start"], 
                                         self.times["calibration"]["yaw_end"]))

        omega_pitch_target = xr.zeros_like(omega_pitch)
        omega_pitch_target[:, 1] = np.sign(omega_pitch[:, 1]) * omega_pitch.reduce(np.linalg.norm,
                                                                                   "cartesian_axis")
        omega_yaw_target = xr.zeros_like(omega_yaw)
        omega_yaw_target[:, 2] = np.sign(omega_yaw[:, 2]) * omega_yaw.reduce(np.linalg.norm,
                                                                             "cartesian_axis")
        rotations = rbm.best_fit_rotation(xr.concat((omega_pitch, omega_yaw), "time"),
                                                  xr.concat((omega_pitch_target, omega_yaw_target), "time"))

        timestamps = min(self.times["calibration"]["pitch_start"],
                              self.times["calibration"]["yaw_start"])

        argmin_time = min(self.odometry.time.values, key=lambda x: abs(x - timestamps))

        # for idx, calib_segment in enumerate(self.times["calibration"].items()):
        #     dt64 = np.datetime64(calib_segment[1])
        #     omega = rbm.transform_vectors(self.odometry.angular_velocity, outof="t265_world", into="t265_vestibular")
        #     omega_pitch = omega.sel(time=slice(calib_segment["pitch_start"], calib_segment["pitch_end"]))
        #     omega_yaw = omega.sel(time=slice(calib_segment["yaw_start"], calib_segment["yaw_end"]))

        #     omega_pitch_target = xr.zeros_like(omega_pitch)
        #     omega_pitch_target[:, 1] = np.sign(omega_pitch[:, 1]) * omega_pitch.reduce(np.linalg.norm, "cartesian_axis")

        #     omega_yaw_target = xr.zeros_like(omega_yaw)
        #     omega_yaw_target[:, 2] = np.sign(omega_yaw[:, 2]) * omega_yaw.reduce(np.linalg.norm, "cartesian_axis")

        #     rotations[idx, :] = rbm.best_fit_rotation(xr.concat((omega_pitch, omega_yaw), "time"),
        #                                               xr.concat((omega_pitch_target, omega_yaw_target), "time"))

        #     timestamps[idx] = min(calib_segment["pitch_start"], calib_segment["yaw_start"])
            
        # print(argmin_time) #IS THERE A REASON THIS ARRAY IS ONE VALUE?
        # print(pd.to_datetime(argmin_time))
        # print(np.array([argmin_time]))
        # print(np.array([pd.to_datetime(argmin_time)]))

        # Construct discrete reference frame
        # Note - may need to change discrete to false, as final version of calibration script this is ported from
        # incorporates an additional step using Reid's plane, that results in a final continuous calibration frame.

        # rbm.register_frame(rotation=np.array([rotations]), #Hacky but passes, otherwise get ValueError: Expected rotation to be of shape (1, 4), got (4,)
        #                    # timestamps=xr.DataArray(argmin_time),
        #                    # timestamps=np.array([pd.to_datetime(argmin_time)]), #pd.to_datetime(argmin_time)
        #                    name="t265_calib", parent="t265_vestibular", inverse=True, discrete=False, update=True)

        rbm.register_frame(rotation=rotations,
                           name="t265_calib", parent="t265_vestibular", inverse=True, discrete=False, update=True)

        record_time = slice(str(self.odometry.orientation.time[0].values), str(self.odometry.orientation.time[-1].values)) #Just selecting entire recording for now, but need it as a slice object

        # Express data in calibrated frame (probably can use iteration to make this cleaner)
        self.calib_ang_pos = rbm.transform_quaternions(self.odometry.orientation.sel(time=record_time),
                                                       outof="t265_world",
                                                       into="t265_calib")
        
        self.calib_lin_pos = rbm.transform_points(self.odometry.position.sel(time=record_time),
                                                    outof="t265_world",
                                                    into="t265_calib")

        #Get TypeError on into="t265_calib": float() argument must be a string or a number, not 'TimeStamp'

        self.calib_ang_vel = rbm.transform_vectors(self.odometry.angular_velocity.sel(time=record_time),
                                                    outof="t265_world",
                                                    into="t265_calib")

        self.calib_lin_vel = rbm.transform_vectors(self.odometry.linear_velocity.sel(time=record_time),
                                                    outof="t265_world",
                                                    into="t265_calib")

        self.calib_ang_acc = rbm.transform_vectors(self.odometry.angular_acceleration.sel(time=record_time),
                                                    outof="t265_world",
                                                    into="t265_calib")

        self.calib_lin_acc = rbm.transform_vectors(self.odometry.linear_acceleration.sel(time=record_time), #Could use accelerometer measurement here as well, to decide later
                                                    outof="t265_world",
                                                    into="t265_calib")

        # print(self.calib_ang_pos)
        # print(self.calib_lin_pos)
        # print(self.calib_ang_vel)
        # print(self.calib_lin_vel)
        # print(self.calib_ang_acc)
        # print(self.calib_lin_acc)

        # Return data expressed in calibrated frame

        self.calib_odo = xr.Dataset(
            {"ang_pos": self.calib_ang_pos, 
            "lin_pos": self.calib_lin_pos,
            "ang_vel": self.calib_ang_vel,
            "lin_vel": self.calib_lin_vel,
            "ang_acc": self.calib_ang_acc,
            "lin_acc": self.calib_lin_acc}
            )
        print(f'PRINT THE CALIBRATED FRAME: {self.calib_odo}')
    
    def calc_gait_variability(self):
        """
        this article is used as a reference for the gait variability metric:
        https://jneuroengrehab.biomedcentral.com/track/pdf/10.1186/1743-0003-2-19.pdf
        """
        plt.plot(self.odometry.position[:,1])
        plt.show()
        # peaks_fast, _= find_peaks(fast[:,1],distance=fps / 3)
        # for i in range(len(peaks_slow) - 1):
        #     step_time_slo[i] = (peaks_slow[i+1] - peaks_slow[i]) * (1/fps)
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
    
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.odometry.time.values,self.odometry.angular_velocity[:, 0],label='uncalibrated pitch velocity')
        ax.plot(self.odometry.time.values,self.odometry.angular_velocity[:, 1],label='uncalibrated yaw velocity')
        ax.plot(self.odometry.time.values,self.calib_odo.ang_vel[:, 0],label='calibrated pitch velocity')
        ax.plot(self.odometry.time.values,self.calib_odo.ang_vel[:, 1],label='calibrated yaw velocity')
        plt.legend()
        plt.ylabel('Angular velocity (rad/s)')
        plt.xlabel('Time')
        plt.show()

    def get_head_orientation(self):
        return self.head_roll, self.head_pitch

    def get_heading(self):
        return self.heading_azimuth, self.heading_elevation
        
    def get_accel(self):
        return self.accel
    
    def get_odometry(self):
        return self.odometry
    
        
    

