from distutils.core import setup
from setuptools import find_packages
from setuptools.command.install import install
import os

setup(
    name='graspnetAPI',
    version='1.2.11',
    description='graspnet API',
    author='Hao-Shu Fang, Chenxi Wang, Minghao Gou',
    author_email='gouminghao@gmail.com',
    url='https://graspnet.net',
    packages=find_packages(),
    install_requires=[
        # allow NumPy 2.x; upstream’s ==1.23.4 pin is what breaks modern stacks
        "numpy>=1.20",
        "scipy>=1.6",
        # Open3D is sensitive; 0.18+ works on recent Python/NumPy
        "open3d",
        "trimesh>=3.9",
        "tqdm>=4.60",
        "pillow>=10",           # dedupe Pillow/pillow, prefer modern
        "matplotlib>=3.5",
        "pywavefront>=1.3",
        "scikit-image>=0.19",
        "dill>=0.3.4",
        "h5py>=3.6",
        "scikit-learn>=1.1",
        # leave opencv and autolab* out of core (see extras below)
        # keep grasp_nms but don’t hard-pin
        "grasp_nms>=1.0.2",
        "opencv-python-headless>=4.8",
        "transforms3d>=0.4.2",
        "autolab_core",
        "autolab-perception",
        "cvxopt"
    ],
    extras_require={
        # Dev/test docs etc. if you want them
        "dev": ["pytest", "black", "isort", "mypy"],
    },
    include_package_data=True,
    zip_safe=False
)
