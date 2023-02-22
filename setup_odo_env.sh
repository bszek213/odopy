# Check for interactive environment
set -e

if [[ $- == *i* ]]
then
	echo 'Interactive mode detected.'
else
	echo 'Not in interactive mode! Please run with `bash -i`'
	exit
fi
echo ">>> update base conda"
conda update -n base conda
# Assure mamba is installed
if hash mamba 2>/dev/null; then
		echo ">>> Found mamba installed."
        mamba update mamba
    else
        conda install mamba -n base -c conda-forge
		mamba update mamba
fi
# Check if the environment exists
if conda env list | grep -q "vedb_analysis38"; then
    echo "vedb_analysis38 exists"
    # Update the existing environment with the packages in env.yaml
    mamba env update -n vedb_analysis38 -f environment.yaml
    echo "Updated existing environment vedb_analysis38 with packages from env.yaml"
else
    # Create a new environment with the packages in env.yaml
    mamba create -n vedb_analysis38 -f environment.yaml
    echo "Created new environment vedb_analysis38 with packages from env.yaml"
fi
#echo ">>> Install from yaml"
#mamba env update -n veb_analysis38 -f environment.yaml
echo ">>> python setup.py install"
conda activate vedb_analysis38
python setup.py install
#echo ">>> Install rigid_body_motion and pupil_recording_interface"
#mamba install -c phausamann -c conda-forge rigid-body-motion
#mamba install -c conda-forge msgpack-python
#conda install -c vedb pupil_recording_interface
#mamba install -c vedb pupil_recording_interface
echo "odopy,rigid_body_motion, and pupil_recording_interface are now installed"
