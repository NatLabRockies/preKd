from io import open
from os import path

from setuptools import find_packages, setup

import versioneer

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="prekd",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    description=(
        "Graph neural networks for predicting the partition coefficient (Kd) of a "
        "target compound in biphasic solvent systems"
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/NatLabRockies/preKd",
    author="Jeff Law",
    author_email="jeffrey.law@nlr.gov",
    license="BSD-3-Clause",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Chemistry",
    ],
    packages=find_packages(exclude=["docs", "tests", "example_data"]),
    project_urls={
        "Source": "https://github.com/NatLabRockies/preKd",
    },
)
