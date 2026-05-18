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
  [https://drive.google.com/drive/folders/1lQVadXv8ohc5x0LQHIKx0Z-IaAR3janM?usp=sharing](https://drive.google.com/drive/folders/1lQVadXv8ohc5x0LQHIKx0Z-IaAR3janM )  
  After downloading, place the `Simulation Results/` folder in the same directory as `Schelling_paper_results.ipynb`.
- `Census_data_results.ipynb`  
  Census-based analysis of historical segregation trends and subset-based metro-level composition experiments.
- `Census Data/`  
  Documentation and geographic crosswalks for Census data. Raw NHGIS tabulations are not redistributed; see the folder README for download instructions.
- `Simulation Results/`  
  Documentation for large simulation outputs (hosted externally due to size).

## Beta Convenience Modules
Two lightweight, self-contained modules are provided for users who want to apply the composition-invariance correction to their own data without working through the full replication code:

- `compositional_correction_beta.py` — Python module implementing all six two-group segregation indices (D, S, H, R, G, Iso) with the I-projection correction. Accepts data as a `pandas.DataFrame`, a 2-column matrix, or two vectors. Supports both the pointwise correction (`target = mu*`) and the full composition curve, with an optional plotting helper.
- `compositional_correction_beta.R` — R port of the same module, with the same API. Base R only; no extra packages required.

**These are beta versions.** They were written after the paper to make the method easier to apply to new data, and were **not** used to produce any of the results reported in the paper. If you find any issues or have questions about applying the correction, please contact me at **barron@demogr.mpg.de**.

## Data Availability
Raw block-group Census tabulations were obtained from IPUMS NHGIS and are not redistributed due to licensing terms. File names and extraction parameters are documented in `Census Data/README.md`.  

## Requirements
The code uses standard scientific Python packages (NumPy, pandas, matplotlib, SciPy).  
Notebooks were developed in Python 3.10+.

The beta convenience modules have lighter requirements:
- Python (`compositional_correction_beta.py`): NumPy, pandas, SciPy (matplotlib only for the plotting helper).
- R (`compositional_correction_beta.R`): base R 4.0+; no additional packages.
