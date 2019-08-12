from setuptools import setup
from setuptools import find_packages
import os


here = os.path.abspath(os.path.dirname(__file__))

setup(
    name='dance_sim_tools',
    version='0.0.0',
    description='Package for creation and analysis of simulated dance on S^1',
    long_description=__doc__,
    author='Annie Rak',
    author_email='arak@caltech',
    license='MIT',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Biology',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.5',
    ],

    packages=find_packages(exclude=['examples',]),
)
