from setuptools import find_packages, setup

setup(
    name="app",
    version="1.0",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[],
)