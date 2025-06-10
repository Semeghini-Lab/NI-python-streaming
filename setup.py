from setuptools import setup, find_packages

setup(
    name="nistreamer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pyzmq",
        "nidaqmx",  # for NI-DAQ functionality
    ],
    python_requires=">=3.7",
    author="Marcin Kalinowski & Yi Zhu",
    description="NI-DAQ streaming package for atom array control",
    long_description=open("README.md").read() if open("README.md").read() else "",
    long_description_content_type="text/markdown",
) 