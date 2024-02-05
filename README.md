# VEDB Odometry README

## Overview

This repository, is for transforming the t265 data within the Visual Experience Database with the `headCalibrate.py` script. This script is designed to perform the calibration of VEDB T265 odometry data from sensor coordinates to head coordinates. It uses the Pupil Recording Interface (`pupil_recording_interface`) and Rigid Body Motion (`rigid_body_motion`) custom libraries, along with other Python modules, to process and calibrate the T265 data.

## Dependencies

Ensure you have the following dependencies installed:

- `numpy`
- `pandas`
- `xarray`
- `matplotlib`
- `pupil_recording_interface`
- `rigid_body_motion`
- `os`
- `yaml`
- `scipy`
- `datetime`
- `plotly`

You can install them using:

```bash
pip install numpy pandas xarray matplotlib pupil_recording_interface rigid_body_motion scipy plotly
```
This can be done in a conda environment
note: use pip 21.3.1, not version 22

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/bszek213/odopy.git
   ```

2. Import the `headCalibrate` class into your Python script:

   ```python
   from odopy import headCalibrate
   ```

3. Create an instance of the `headCalibrate` class:

   ```python
   odo = headCalibrate.headCalibrate()
   ```

4. Set the path to the folder containing T265 data:

   ```python
   curr_dir = getcwd()
   folder = path.join(curr_dir, 'your_data_folder')
   odo.set_odometry_local(folder)
   ```

5. Plot and select start and end timestamps for calibration:

   ```python
   """
   This is only necessary if there is no yaml of start and end segments of head shakes and nods.
   the code expects the yaml file (odo_times.yaml) to be formated as follows:
   - calibration:
        pitch_end: 2022-05-31 17:12:49
        pitch_start: 2022-05-31 17:12:35
        yaw_end: 2022-05-31 17:12:49
        yaw_start: 2022-05-31 17:12:50
   """
   odo.start_end_plot()
   ```

6. Perform T265 to head transformation:

   ```python
   odo.t265_to_head_trans()
   ```

7. Calculate head orientation:

   ```python
   odo.calc_head_orientation()
   head_roll, head_pitch, head_yaw = odo.get_head_orientation()
   ```

8. Visualize calibrated data:

   ```python
   odo.plot()
   ```

## Additional Functionality

The script also provides additional functionality, such as calculating gait variability and instantaneous force. You can explore and use these features based on your specific needs.

## Example

An example usage script is provided in the `test_odo.py` file. You can modify this script to suit your dataset and requirements.

## Contributors

- [brianszekely](https://github.com/bszek213)
- [christiansinnott](https://github.com/csinnott91)

Feel free to contribute or report issues by creating pull requests or raising issues.
