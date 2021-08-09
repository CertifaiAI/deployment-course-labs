from setuptools import setup, find_packages

# List of requirements
requirements = []
# Package (minimal) configuration
setup(
    name="app",
    version="1.0.0",
    description="Object detector web app",
    packages=find_packages(),  # __init__.py folders search
    install_requires=requirements,
)
