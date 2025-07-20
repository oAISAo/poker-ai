# filepath: setup.py
from setuptools import setup, find_packages

setup(
    name='poker-ai',  # Replace with your project's name
    version='0.1.0',  # Replace with your project's version
    packages=find_packages(),
    install_requires=[
        'pytest'  # Add any dependencies here, including pytest
    ],
)