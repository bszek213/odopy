from setuptools import setup, find_packages

requirements = []

setup(
    name="vedb_odometry",
    version="0.0.0",
    packages=find_packages(),
    long_description=open("README.md").read(),
    install_requires=requirements,
    include_package_data=True,
    package_data={
        'vedb_odometry':[
            #'defaults.cfg',
            #'config/*yaml'
        ]
    }
)
