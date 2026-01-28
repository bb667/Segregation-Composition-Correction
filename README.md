# Segregation-Composition-Correction
Replication code and data for *When All Measures Fail the Same Way*

## Segregation Metrics and Composition Adjustment

This repository contains code and data used in the analysis of segregation metrics and their dependence on group composition, including simulations and U.S. Census data applications.

## Repository Structure

- `git_segregation.py`  
  Core Python module implementing segregation indices, composition-adjusted indices, Schelling simulations, neighborhood partitioning, and plotting utilities.

- `Schelling_example.ipynb`  
  Demonstration notebook showing how to run Schelling simulations and compute segregation measures.

- `Schelling_paper_results.ipynb`  
  Analysis notebook reproducing simulation results used in the paper, including results for cities at different compositions and neighborhood subset analyses.  
  Note: This notebook requires downloading simulation snapshot data from Google Drive:  
  [https://drive.google.com/drive/folders/1KZu9CG0nNOJ44mHkN1TkMhavsJacjj_0](https://drive.google.com/drive/folders/1KZu9CG0nNOJ44mHkN1TkMhavsJacjj_0?usp=sharing)  
  After downloading, place the `Simulation Results/` folder in the same directory as `Schelling_paper_results.ipynb`.

- `Census_data_results.ipynb`  
  Census-based analysis of historical segregation trends and subset-based metro-level composition experiments.

- `Census Data/`  
  Documentation and geographic crosswalks for Census data. Raw NHGIS tabulations are not redistributed; see the folder README for download instructions.

- `Simulation Results/`  
  Documentation for large simulation outputs (hosted externally due to size).

## Data Availability

Raw block-group Census tabulations were obtained from IPUMS NHGIS and are not redistributed due to licensing terms. File names and extraction parameters are documented in `Census Data/README.md`.  
Large simulation outputs are hosted externally due to size; see `Simulation Results/README.md` for download links.

## Requirements

The code uses standard scientific Python packages (NumPy, pandas, matplotlib).  
Notebooks were developed in Python 3.10+.
