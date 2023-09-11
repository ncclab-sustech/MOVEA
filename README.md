# MOVEA: Multi-Objective Optimization via Evolutionary Algorithm for High-Definition Transcranial Electrical Stimulation of the Human Brain

## Introduction
This repository contains the official implementation of the MOVEA algorithm for optimizing high-definition transcranial electrical stimulation (HD-tES) of the human brain. MOVEA is a multi-objective optimization approach that utilizes an evolutionary algorithm to find optimal stimulation parameters for a given target brain area. The code has been tested on a Linux system.

## Dependencies
To run the code, you need to have the following dependencies installed:
- pip install geatpy

## Quick Start
To get started, you can run the `main.py` script with the following command-line options:
- `-t`: Stimulation method (e.g., tdcs, ti)
- `-m`: Head model (e.g., hcp4)
- `-p`: Target name or MNI coordinate (e.g., motor)
- `-g`: Maximum epochs (default: 100)

### Input Data
The input consists of two parts: the lead field matrix file and the voxel position file. These files are used to calculate the electric field and to determine the voxel indices corresponding to MNI coordinates. You can create these files yourself or get them from an h5py file in the output folder of SimNIBS by running convert.ipynb. In this repository, we provide files for the HCP4 head model from the Human Connectome Project (HCP) dataset.

### Output
The output files are located in the "Output" folder and 'pic' folder. The "fitness" file contains the results of the objectives, and the "in" file contains information about the montage and current. Each line represents a solution. The relationship between array indices and activated channels is specified in the "hcp4.csv" file. Note that the Cz channel is not included.

### Example Command
Here is an example command to optimize the motor area for the HCP4 head model using the tTIS stimulation method:
```bash
python main.py -t ti -m hcp4 -p motor
```
You will get two files in the "Output" folder. In the first line of the "fitness" file, you may see values such as 1.972386587771203 1.819407772791436506e-01, which represent the two objectives defined in util.py. In the first line of the "in" file, you may see values such as 46 1.0 2 -1.0 74 1.0 23 -1.0, indicating that the Cp2 channel (47th row in "hcp4.csv") injects 1 mA and so on. For HD-based methods, the currents for all channels are displayed.

### Citation
If you find this code useful for your research, please consider citing the following paper: [https://doi.org/10.48550/arXiv.2211.05658](https://doi.org/10.1016/j.neuroimage.2023.120331)

### Contact
We provide some results from the paper in 'others', for any questions or issues related to this implementation, please contact the authors.
