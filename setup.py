#from setuptools import setup, find_packages
from setuptools import setup

setup(
   name='vedb_odometry',
   version='0.1.0',
   author='Brian Szekely and Christian Sinnott',
   author_email='bszekely@nevada.unr.edu',
   packages=['vedb_odometry', 'vedb_odometry.test'],
   scripts=['src/vedb_odometry/vedb_calibration.py'],
   #url='http://pypi.python.org/pypi/PackageName/',
   license='LICENSE.txt',
   description='VEDB odometry package that handles the T265 data',
   long_description=open('README.md').read(),
   install_requires=[
   "numpy >= 1.22.2",
   #"rigid-body-motion >= 0.9.1",# Not pip installable, must be installed from conda or git
   #"pupil_recording_interface >= 0.5.0", # Not pip installable, must be installed from conda or git
   "xarray >= 0.21.1",
   "scipy >= 1.8.0",
   #"vedb-store >= 0.0.1", # Not pip installable, must be installed from git
   "plotly >= 5.6.0",
   "pandas >= 1.4.1",
   "matplotlib==3.5.2"
   ],
)
