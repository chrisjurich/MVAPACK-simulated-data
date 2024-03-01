# Simulated LC-MS Data Used in Benchmarking of MVAPACK 

This repository contains the workflow for creating the simulated data used 
in the MVAPACK simulated LC-MS data project. Below is a brief description of the 
relevant files in this repository.


## `./main.py`

This is the driver script for creating the nine versions of the simulated dataset. The 
script is not run with any commandline arguments. It has been rng seeded to ensure the same
data is created each time. Revolves around the `ConditionManager` class defined in 
`condition_manager.py`, which takes the `ClassEic`'s, significance mapper, percent missing level,
and noise level, to output the full 20 replicates for each condition and outputs the respective `.mzML` 
files. Requires the following packages to run:
    + `matplotlib`
    + `numpy`
    + `psims`

## `utils.py`

Contains convenience functions and `namedtuple()` data holders that ease the creation of the synthetic 
dataset. There is a lot of talk about "significant" vs "non-significant" chemicals. This refers to 
chemicals that have variantion across replicates that is statistically significant versus ones that do not.

## `norm_eics.py`
Helper script used just once to normalize the maximum peak of the compound to 1.0. Loads in eics from the `data/eics-cpy2.pickle`
file and saves them back to that file.


## `mults.py`

Auxiliary script that creates the multipliers that will be combined with the eics to create the master replicate. 
By default makes 3000 non-significant multiplier sets and 1000 significant multiplier sets. Saves the results to
`multipliers.csv`


## `vimms/`

The specially modified version of the ViMMS software package used in this data generation protocol to pull the EIC
waveforms out of memory directly.


## `peak_graphs`

Contains the individual peak waveforms that were analyzed during the course of creating the initial compound
list for the simulated dataset.



## `vimms_data_generation`

This directory contains a number of scripts critical to creating the original EICs and idealizing them.
