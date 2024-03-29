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
import os
import yaml
from scipy.signal import savgol_filter
from datetime import datetime
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R
import hvplot 
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'
class headCalibrate():
    def __init__(self):
        self.name = None 
        self.odometry = None
        self.accel = None
        self.fps = 200
    
    def set_odometry_local(self, folder):
        isdir = os.path.isdir(folder) 
        if isdir == False:
            print('Folder does not exist. Check the path')
        elif isdir == True:
            self.folder = folder
            # accel_local = os.path.join(folder, 'accel.pldata')
            # odo_local = os.path.join(folder, 'odometry.pldata')
            self.odometry = pri.load_dataset(folder,odometry='recording',cache=False)
            self.accel = pri.load_dataset(folder,accel='recording',cache=False)
            
    # def set_odometry_cloud(self, date_lookup):
    #     dbi = vedb_store.docdb.getclient()
    #     sessions = dbi.query('+date', type="Session")
    #     # # print(sessions.sort(key=myFunc))
    #     for i in range(len(sessions)):
    #         temp_date = sessions[i]['date']
    #         if temp_date == date_lookup:
    #             try:
    #                 temp_path = sessions[i].paths["odometry"][1]
    #                 self.folder = os.path.dirname(sessions[i].paths["odometry"][1])
    #                 temp_path = '/'.join(temp_path.split('/')[0:-1])
    #                 self.odometry = pri.load_dataset(temp_path,
    #                                                   odometry='recording',
    #                                                   cache=False)
    #                 self.accel = pri.load_dataset(temp_path,
    #                                               accel="recording", 
    #                                               cache=False)
    #             except:
    #                 print('=================================')
    #                 print('Sessions returned no paths. Please mount the VEDB to hard drive')
    #                 print('=================================')
        
    def start_end_plot(self):
        path_odo = os.path.join(self.folder, 'odo_times.yaml')
        path_exists = os.path.exists(path_odo)
        if path_exists == True:
            print('Found odo_times.yaml.')
            with open(path_odo) as file:
                time_list = yaml.load(file, Loader=yaml.FullLoader)
            try:
                self.pitch_start = time_list[0]['calibration']['pitch_start']
                self.pitch_end = time_list[0]['calibration']['pitch_end']
                self.yaw_start = time_list[0]['calibration']['yaw_start']
                self.yaw_end = time_list[0]['calibration']['yaw_end']
            except:
                print('wrong index. try without indexing [0]')
                self.pitch_start = time_list['pitch_start']
                self.pitch_end = time_list['pitch_end']
                self.yaw_start = time_list['yaw_start']
                self.yaw_end = time_list['yaw_end']
            self.times = {'calibration':
            {'pitch_start': self.pitch_start,
             'pitch_end': self.pitch_end,
             'yaw_start': self.yaw_start,
             'yaw_end': self.yaw_end
             }}
        else:
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
            save_to_yaml = {'pitch_start': self.pitch_start,
                            'pitch_end': self.pitch_end,
                            'yaw_start': self.yaw_start,
                            'yaw_end': self.yaw_end
                            }
            yaml_file = os.path.join(self.folder, 'odo_times.yaml')
            # if os.path.exists(yaml_file):
            #     raise ValueError('File %s already exists! Please rename or remove it if you wish to overwrite.'%(str(yaml_file)))
            # else:
            with open(yaml_file, mode='w') as fid:
                yaml.dump(save_to_yaml, fid)
            self.times = {'calibration':
                {'pitch_start': self.pitch_start,
                'pitch_end': self.pitch_end,
                'yaw_start': self.yaw_start,
                'yaw_end': self.yaw_end
                }}

    def calculate_world_coordinate_system(self):
        #TODO: check the means of each axis for every take and save those to notes
        # print(f'Mean X,Y,Z accel: [{np.mean(self.accel.linear_acceleration[:,0].values)},{np.mean(self.accel.linear_acceleration[:,1].values)},{np.mean(self.accel.linear_acceleration[:,2].values)}]')
        # input()
        path_odo = os.path.join(self.folder, 'reids_plane.yaml')
        self.path_exists_reids = os.path.exists(path_odo)
        if self.path_exists_reids == True:
            with open(path_odo) as file:
                time_reids = yaml.load(file, Loader=yaml.FullLoader)
            print('Read reids_plane.yaml file')
            mean_g = self.accel.linear_acceleration.sel(
                time=slice(time_reids["reids_start"], time_reids["reids_end"])
            ).reduce(np.linalg.norm, "cartesian_axis").mean()
            print(f'mean g over reids plane: {mean_g.values}')
        else:
            mean_g = self.accel.linear_acceleration.reduce(np.linalg.norm, "cartesian_axis").mean()
        g_world = xr.DataArray(
            [0, 0, mean_g],
            coords={"cartesian_axis": ["x", "y", "z"]},
            dims="cartesian_axis",
            name="gravity",
        )
        g_target = xr.broadcast(self.accel.linear_acceleration, g_world)[1] #this makes g_worlds have the same shape as the data
        #this transformation does nothing as the data are already in vestibular coordinates
        # print(f'g_world before out of world and into t265_calib: {g_world}')
        gravity_world_coords = rbm.transform_vectors(g_world, outof="world", into="t265_calib") #was t265_calib in old code

        length_gravity_world_coords = len(gravity_world_coords)
        length_g_target = len(g_target)
        downsampling_factor = length_gravity_world_coords // length_g_target
        #Downsample gravity_world_coords to match the length of g_target
        gravity_world_coords = gravity_world_coords.isel(time=slice(0, length_g_target * downsampling_factor, downsampling_factor))
    
        self.rot_world = rbm.qmean(rbm.shortest_arc_rotation(gravity_world_coords, g_target).values)
        # print(self.rot_world)
        # print(f'before weirdness: {self.rot_world}')
        # self.rot_world[[1, 3]] = 0
        self.rot_world /= np.linalg.norm(self.rot_world)
        # print(f'after weirdness: {self.rot_world}')
        rbm.register_frame(
            rotation=self.rot_world, name="world_coord", parent="t265_calib", update=True, inverse=True,
        )

        if self.path_exists_reids == True:
            gravity = rbm.transform_vectors(g_world, outof="world", into="world_coord")
        else:
            gravity = rbm.transform_vectors(g_world, outof="world", into="t265_calib")
        norm = gravity.reduce(np.linalg.norm, "cartesian_axis")

        #save head pitch and roll
        self.head_roll = np.rad2deg(np.arctan(gravity[:,1]/gravity[:,2]))
        self.head_pitch = np.rad2deg(-np.arcsin(gravity[:,0]/norm))

        # calib_lin_acc = rbm.transform_vectors(self.odometry.linear_velocity, #Could use accelerometer measurement here as well, to decide later
        #                                             outof="t265_world",
        #                                             into="world_coord")
        
        # plt.plot(self.odometry.time.values, gravity[:, 0], color='red', label='x', linewidth=3,alpha=0.7)
        # plt.plot(gravity[:,1], gravity[:, 0], color='blue', label='y', linewidth=3, alpha=0.7)
        # plt.plot(self.odometry.time.values, gravity[:, 1], color='blue', label='y', linewidth=3, alpha=0.7)
        # plt.plot(self.odometry.time.values, gravity[:, 2], color='green', label='z', linewidth=3, alpha=0.7)
        # plt.title('linear accel in world... supposedly')

        # plt.scatter(head_roll, head_pitch, color='blue',s=1,alpha=0.5)
        # plt.ylabel('Pitch wrt to gravity (degrees)')
        # plt.xlabel('Roll wrt to gravity (degrees)')
        # # plt.legend()
        # plt.show()
        

    def t265_to_head_trans(self):
        # Pull annotated calibration segment times from yaml (LEGACY Fself.timesROM OLD ANALYSIS, NEEDS CHANGED)
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

        #old way
        self.R_WORLD_ODOM = np.array([[0, 0, -1], 
                                [-1, 0, 0], 
                                [0, 1, 0]])
        self.R_IMU_ODOM = np.array([[-1, 0, 0], 
                              [0, 1, 0], 
                               [0, 0, -1]])
        #new way
        # R_WORLD_ODOM = np.array([[0, -1, 0], 
        #                          [0, 0, 1], 
        #                          [-1, 0, 0]])
        # R_IMU_ODOM = np.array([[0, 1, 0], 
        #                        [0, 0, 1], 
        #                        [1, 0, 0]])


        rbm.register_frame("world",update=True)
        rbm.ReferenceFrame.from_rotation_matrix(self.R_WORLD_ODOM, name="t265_world",
                                                parent="world").register(update=True)

        # rbm.register_frame(
        #     "t265_odom",
        #     translation=self.odometry.position.values,
        #     rotation=self.odometry.orientation.values,
        #     timestamps=self.odometry.time.values,
        #     parent="t265_world",
        #     update=True,
        # )
        
        rbm.ReferenceFrame.from_dataset(self.odometry, translation = "position",
                                        rotation = "orientation", timestamps = "time",
                                        parent = "t265_world", name = "t265_odom").register(update=True)
        
        rbm.ReferenceFrame.from_rotation_matrix(self.R_IMU_ODOM, name="t265_imu", parent="t265_odom").register(update=True)
        rbm.ReferenceFrame.from_rotation_matrix(self.R_WORLD_ODOM, name="t265_vestibular", 
                                                parent="t265_odom", 
                                                inverse=True
                                                ).register(update=True)
        

        # Define first calibrated frame using calibration segments identified in set_calibration method
        # segments = [times["pre_calib"]] + [times["re_calib"]]
        segments = self.times["calibration"]
        rotations = np.zeros([len(segments), 4]) #Original has 4 - yaw, pitch, Reid's
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
                            name="t265_calib", parent="t265_vestibular",
                            inverse=True, discrete=False, update=True)
        
        #calculate world coordinate system from gravity
        self.calculate_world_coordinate_system()
        print(rbm.render_tree("world"))

        #explicity transform the sensor coordinate frame to calibrated coordinate frame 
        #WORK IN PROGRESS
        # print(f'Quaternion that will transform sensor to head coordinates: {rotations}')
        # print(f'Rotation Matrix that will transform sensor to head coordinates: {self.quat_to_rot_mat(rotations)}')
        # print(f'Rotation Matrix that will transform calibrated rf to world coordinates: {self.quat_to_rot_mat(self.rot_world)}')
        # og_euler = self.quat_to_euler(self.odometry.orientation.values)

        # calibrated_orientation = rbm.qmul(self.odometry.orientation.values,rotations)
        # calibrated_orientation_world = rbm.qmul(calibrated_orientation,self.rot_world)
        
        # # calib_euler = self.quat_to_euler(calibrated_orientation_world)

        # #swap the george w with zed axis
        # calibrated_orientation_world[:, [0, -1]] = calibrated_orientation_world[:, [-1, 0]]

        # euler_angles = []
        # for q in (calibrated_orientation):
        #     r = R.from_quat(q)
        #     euler_angles.append(r.as_euler('xyz',degrees=True))
        # # euler_angles = np.vstack([np.pad(arr, (0, max(map(len, euler_angles)) - len(arr))) for arr in euler_angles])
        # euler_angles = np.vstack(euler_angles)
        # euler_angles = euler_angles * -1
        # plt.figure()
        # plt.scatter(euler_angles[:,0], euler_angles[:,1], color='blue',s=1,alpha=0.5,label='quaternion method')
        # plt.scatter(self.head_roll, self.head_pitch, color='red',s=1,alpha=0.5,label='acceleration method')
        # plt.title('Quaternion transformation to euler')
        # plt.ylabel('Pitch wrt to gravity (degrees)')
        # plt.xlabel('Roll wrt to gravity (degrees)')
        # plt.legend()
        # plt.tight_layout()
        # plt.show()
        #STILL WORKING ON TRANSFORMING HEAD TO GRAVITY VIA QUATERNIONS

        # Create a figure and three subplots
        # fig, axs = plt.subplots(3, 1, figsize=(10, 10))

        # # Plot roll
        # axs[0].plot(self.odometry.time.values, og_euler[:, 0], color='red', label='roll before transformation', linewidth=3)
        # axs[0].plot(self.odometry.time.values, euler_angles[:, 0], color='blue', label='roll after transformation', linewidth=3, alpha=0.5)
        # axs[0].set_title('Roll')
        # axs[0].set_ylabel('Degrees')
        # axs[0].legend()

        # # Plot pitch
        # axs[1].plot(self.odometry.time.values, og_euler[:, 1], color='red', label='pitch before transformation', linewidth=3)
        # axs[1].plot(self.odometry.time.values, euler_angles[:, 1], color='blue', label='pitch after transformation', linewidth=3, alpha=0.5)
        # axs[1].set_title('Pitch')
        # axs[1].set_ylabel('Degrees')
        # axs[1].legend()

        # # Plot yaw
        # axs[2].plot(self.odometry.time.values, og_euler[:, 2], color='red', label='yaw before transformation', linewidth=3)
        # axs[2].plot(self.odometry.time.values, euler_angles [:, 2 ], color='blue', label='yaw after transformation', linewidth=3, alpha=0.5)
        # axs[2].set_title('Yaw')
        # axs[2].set_ylabel('Degrees')
        # axs[2].legend()

        # plt.tight_layout()
        # plt.show()
        # # plt.savefig('roll_pitch_yaw.png',dpi=400)
        # plt.close()

        # # Create a figure and three subplots
        # fig, axs = plt.subplots(4, 1, figsize=(10, 10))

        # # Plot roll
        # axs[0].plot(self.odometry.time.values, self.odometry.orientation.values[:, 0], color='red', label='w before transformation', linewidth=3)
        # axs[0].plot(self.odometry.time.values, calibrated_orientation[:, 0], color='blue', label='w after transformation', linewidth=3, alpha=0.3)
        # axs[0].set_title('w')
        # axs[0].set_ylabel('Degrees')
        # axs[0].legend()

        # # Plot pitch
        # axs[1].plot(self.odometry.time.values, self.odometry.orientation.values[:, 1], color='red', label='x before transformation', linewidth=3)
        # axs[1].plot(self.odometry.time.values, calibrated_orientation[:, 1], color='blue', label='x after transformation', linewidth=3, alpha=0.3)
        # axs[1].set_title('x')
        # axs[1].set_ylabel('Degrees')
        # axs[1].legend()

        # # Plot yaw
        # axs[2].plot(self.odometry.time.values, self.odometry.orientation.values[:, 2], color='red', label='y before transformation', linewidth=3)
        # axs[2].plot(self.odometry.time.values, calibrated_orientation[:, 2], color='blue', label='y after transformation', linewidth=3, alpha=0.3)
        # axs[2].set_title('y')
        # axs[2].set_ylabel('Degrees')
        # axs[2].legend()

        # # Plot z
        # axs[3].plot(self.odometry.time.values, self.odometry.orientation.values[:, 3], color='red', label='z before transformation', linewidth=3)
        # axs[3].plot(self.odometry.time.values, calibrated_orientation[:, 3], color='blue', label='z after transformation', linewidth=3, alpha=0.3)
        # axs[3].set_title('z')
        # axs[3].set_ylabel('Degrees')
        # axs[3].legend()

        # plt.tight_layout()
        # plt.show()
        # plt.savefig('quaternion_transformation.png',400)
        # plt.close()

        record_time = slice(str(self.odometry.orientation.time[0].values), str(self.odometry.orientation.time[-1].values)) #Just selecting entire recording for now, but need it as a slice object

        # Express data in calibrated frame (probably can use iteration to make this cleaner)
        self.calib_ang_pos = rbm.transform_quaternions(self.odometry.orientation.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_quaternions(self.odometry.orientation.sel(time=record_time), outof="t265_world", into="t265_calib")

        self.calib_lin_pos = rbm.transform_points(self.odometry.position.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_points(self.odometry.position.sel(time=record_time), outof="t265_world", into="t265_calib")
        #Get TypeError on into="t265_calib": float() argument must be a string or a number, not 'TimeStamp'
        self.calib_ang_vel = rbm.transform_vectors(self.odometry.angular_velocity.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_vectors(self.odometry.angular_velocity.sel(time=record_time), outof="t265_world", into="t265_calib")

        self.calib_lin_vel = rbm.transform_vectors(self.odometry.linear_velocity.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_vectors(self.odometry.linear_velocity.sel(time=record_time), outof="t265_world", into="t265_calib")

        self.calib_ang_acc = rbm.transform_vectors(self.odometry.angular_acceleration.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_vectors(self.odometry.angular_acceleration.sel(time=record_time), outof="t265_world", into="t265_calib")

        self.calib_lin_acc = rbm.transform_vectors(self.odometry.linear_acceleration.sel(time=record_time), outof="t265_world", into="world_coord") if self.path_exists_reids == True else rbm.transform_vectors(self.odometry.linear_acceleration.sel(time=record_time), outof="t265_world", into="t265_calib")

        # Return data expressed in calibrated frame
        self.calib_odo = xr.Dataset(
            {"ang_pos": self.calib_ang_pos, 
            "lin_pos": self.calib_lin_pos,
            "ang_vel": self.calib_ang_vel,
            "lin_vel": self.calib_lin_vel,
            "ang_acc": self.calib_ang_acc,
            "lin_acc": self.calib_lin_acc}
            )
        print('odometry data are now calibrated')
    
    def calc_gait_variability(self):
        """
        this article is used as a reference for the gait variability metric:
        https://jneuroengrehab.biomedcentral.com/track/pdf/10.1186/1743-0003-2-19.pdf
        """
        # plt.plot(self.odometry.position[:,1])
        # plt.show()
        # peaks_fast, _= find_peaks(fast[:,1],distance=fps / 3)
        # for i in range(len(peaks_slow) - 1):
        #     step_time_slo[i] = (peaks_slow[i+1] - peaks_slow[i]) * (1/fps)
        pass
    
    def calc_inst_force(self):
        pass

    def quat_to_rot_mat(self,quaternion):
        """
        Parameters: 
        -array([-9.99428637e-01, -1.27234994e-02,  3.98607922e-04,  3.13105822e-02]) : Coordinates:'w' 'x' 'y' 'z'

        Returns:
        - numpy.ndarray - 3x3 rotation matrix.
        Output: [[ 9.98038977e-01  6.25752416e-02 -1.59352069e-03]
                [-6.25955284e-02  9.97715520e-01 -2.54074980e-02]
                [ 3.78386558e-16  2.54574206e-02  9.99675907e-01]]
        
        Equation:
        R = | 1 - 2(q_y^2 + q_z^2)  2(q_x*q_y - q_w*q_z)  2(q_x*q_z + q_w*q_y) |
            | 2(q_x*q_y + q_w*q_z)  1 - 2(q_x^2 + q_z^2)  2(q_y*q_z - q_w*q_x) |
            | 2(q_x*q_z - q_w*q_y)  2(q_y*q_z + q_w*q_x)  1 - 2(q_x^2 + q_y^2) |
        """
        q_w, q_x, q_y, q_z = quaternion
        rot_mat_sensor_to_head = np.array([
            [1 - 2 * (q_y**2 + q_z**2), 2 * (q_x*q_y - q_w*q_z), 2 * (q_x*q_z + q_w*q_y)],
            [2 * (q_x*q_y + q_w*q_z), 1 - 2 * (q_x**2 + q_z**2), 2 * (q_y*q_z - q_w*q_x)],
            [2 * (q_x*q_z - q_w*q_y), 2 * (q_y*q_z + q_w*q_x), 1 - 2 * (q_x**2 + q_y**2)]
        ])
        return rot_mat_sensor_to_head

    def calc_head_orientation(self):
        pass
        # x = self.accel.sel(cartesian_axis="x")
        # y = self.accel.sel(cartesian_axis="y")
        # z = self.accel.sel(cartesian_axis="z")
        # self.head_roll = np.rad2deg(np.arctan2(y,x))
        # self.head_pitch = np.rad2deg(np.arctan2(y,z))
        # self.head_yaw = np.rad2deg(np.arctan2(z,x))
        #ASSUMES SPHERICAL COORDINATES
        # norm = self.accel.reduce(np.linalg.norm, "cartesian_axis")
        # self.head_roll = np.rad2deg(np.arctan(y/z))
        # self.head_pitch = np.rad2deg(-np.arcsin(x/norm))
        # self.head_yaw = np.rad2deg(-np.arctan2(y, x))

    def calc_heading(self):
        x = self.accel.linear_vel.sel(cartesian_axis="x")
        y = self.accel.linear_vel.sel(cartesian_axis="y")
        z = self.accel.linear_vel.sel(cartesian_axis="z")
        norm = self.accel.linear_vel.reduce(np.linalg.norm, "cartesian_axis")
        heading_azimuth = np.rad2deg(-np.arctan2(y, x))
        heading_elevation = np.rad2deg(np.arcsin(z/norm))
        return heading_azimuth, heading_elevation
    
    def quat_to_euler(self,quaternions):
        """
        Convert quaternions to Euler angles.
        
        Roll (φ) = atan2(2(w x + y z), 1 - 2(x^2 + y^2))
        Pitch (θ) = asin(2(w y - z x))
        Yaw (ψ) = atan2(2(w z + x y), 1 - 2(y^2 + z^2))

        our setup returns data like this [yaw,pitch,roll]
        """
        q_w, q_x, q_y, q_z = quaternions[:, 0], quaternions[:, 1], quaternions[:, 2], quaternions[:, 3]
        #Roll
        roll = np.degrees(np.arctan2(2 * (q_w * q_x + q_y * q_z), 1 - 2 * (q_x**2 + q_y**2)))
        #Pitch
        pitch = np.degrees(np.arcsin(2 * (q_w * q_y - q_z * q_x)))
        #Yaw
        yaw = np.degrees(np.arctan2(2 * (q_w * q_z + q_x * q_y), 1 - 2 * (q_y**2 + q_z**2)))

        return np.column_stack((yaw, roll, pitch)) #roll, pitch, yaw

    def plot(self):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=self.odometry.angular_velocity[:, 0],
            name="uncalibrated pitch velocity"       # this sets its legend entry
            ))
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=self.odometry.angular_velocity[:, 1],
            name="uncalibrated yaw velocity"       # this sets its legend entry
            ))
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=self.calib_odo.ang_vel[:, 1],
            name="calibrated pitch velocity"       # this sets its legend entry
            ))
        fig.add_trace(go.Scatter(
            x=self.odometry.time.values,
            y=self.calib_odo.ang_vel[:, 2],
            name="calibrated yaw velocity"       # this sets its legend entry
            ))
        fig.update_layout(
            title="Angular velocity odometry",
            xaxis_title="Time stamps (datetime)",
            yaxis_title="Angular Velocity (radians/second)",
            )
        fig.show()
        # fig, ax = plt.subplots(figsize=(20,8))
        # ax.plot(self.odometry.time.values,self.odometry.angular_velocity[:, 0],label='uncalibrated pitch velocity')
        # ax.plot(self.odometry.time.values,self.odometry.angular_velocity[:, 1],label='uncalibrated yaw velocity')
        # ax.plot(self.odometry.time.values,self.calib_odo.ang_vel[:, 1],label='calibrated pitch velocity')
        # ax.plot(self.odometry.time.values,self.calib_odo.ang_vel[:, 2],label='calibrated yaw velocity')
        # plt.legend()
        # plt.ylabel('Angular velocity (rad/s)')
        # plt.xlabel('Time')
        # plt.show()

    def get_head_orientation(self):
        return self.head_roll, self.head_pitch
    
    def get_rbm(self):
        return rbm

    def get_heading(self):
        return self.heading_azimuth, self.heading_elevation
        
    def get_accel(self):
        return self.accel
    
    def get_odometry(self):
        return self.odometry

    def get_calibrated_odo(self):
        return self.calib_odo
