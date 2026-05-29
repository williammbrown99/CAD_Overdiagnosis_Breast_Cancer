# Risk-Adjusted Machine Learning for Breast Cancer Overdiagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-Zenodo-green.svg)](https://zenodo.org/records/20436617?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIzMGY4MWI5LWJmMjUtNDBmNi05NGFhLTIzMjQ4ODE3NTQwNyIsImRhdGEiOnt9LCJyYW5kb20iOiJmYzA1NWM0YzBjMTY1ODcyNTA5ZTg5MGE5ZTFiMzU2NCJ9.bcbccspIANXQrBkAYO-b9MOSZKPAbXL77guIdGb_sWbEWgqeS_bFZCinIBOCQr-J_lvL7l9wcZ6bGiBstc7pXg)

## Overview
This repository contains the data engineering, machine learning, and statistical visualization pipeline for predicting breast cancer trajectories (Aggressive vs. Indolent/Overdiagnosed) across a 5-to-10-year survival continuum. 

To ensure patient safety and avoid the undertreatment of lethal disease, this pipeline utilizes an **Extreme Gradient Boosting (XGBoost)** classifier equipped with a dynamic, sliding-scale asymmetric loss function. By heavily penalizing False Positives (misclassifying a lethal tumor as safe), the algorithm establishes a "mathematical firewall" that achieves 100% Specificity and a Precision of 1.00 up to a full decade post-diagnosis.

## Data Availability (Zenodo)
Due to GitHub's file size restrictions, the primary SEER clinical dataset (`SEER_10436_in_situ_included.csv`) is hosted externally on Zenodo. 

**To run this pipeline locally:**
1. Download the dataset from the Zenodo repository: [Zenodo Data Link](https://zenodo.org/records/20436617?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIzMGY4MWI5LWJmMjUtNDBmNi05NGFhLTIzMjQ4ODE3NTQwNyIsImRhdGEiOnt9LCJyYW5kb20iOiJmYzA1NWM0YzBjMTY1ODcyNTA5ZTg5MGE5ZTFiMzU2NCJ9.bcbccspIANXQrBkAYO-b9MOSZKPAbXL77guIdGb_sWbEWgqeS_bFZCinIBOCQr-J_lvL7l9wcZ6bGiBstc7pXg)
2. Create a folder named `INPUT/` in the root directory of this project.
3. Place the downloaded `.csv` file directly into the `INPUT/` folder.

*(Note: The `INPUT/` directory and all `.csv` files are added to `.gitignore` to prevent accidental large-file commits).*

## Repository Structure
The codebase is divided into modular scripts corresponding to the methodology and results of the study:

* **`main.py`**: The core XGBoost classification script. Handles feature engineering (sequential backfilling, NaN handling for sparsity-aware splitting), temporal thresholding, and runs the dynamic sliding-scale optimization to guarantee zero false positives across the 10-year continuum.
* **`Performance_Visualization.py`**: Generates publication-ready 1x3 grouped bar charts comparing the three cost-optimization strategies (Max Clinical Safety, 1-to-1 Balanced Cost, Max OD Detection).
* **`P_value_Table.py`**: Conducts baseline demographic and statistical analyses (Mann-Whitney U tests for continuous variables, Chi-square for categorical) to generate the baseline characteristics table.
* **`Feature_Importance_CSS.py`**: Extracts XGBoost relative feature importance metrics and generates Kaplan-Meier Cancer-Specific Survival (CSS) curves with log-rank testing to clinically validate the predicted cohorts.

## Requirements and Installation
This project requires Python 3.8 or higher. To install the required dependencies, run the following command in your terminal:

    pip install -r requirements.txt

**Core Libraries Used:**
* `xgboost` (Classification and sparsity-aware splitting)
* `scikit-learn` (Hyperparameter tuning, cross-validation, preprocessing)
* `pandas` & `numpy` (Data wrangling and harmonization)
* `scipy` (Statistical testing)
* `lifelines` (Kaplan-Meier survival analysis and log-rank tests)
* `matplotlib` & `seaborn` (Publication styling and visualization)

## Usage
Ensure your clinical data is correctly placed in `INPUT/SEER_10436_in_situ_included.csv`. 

To train the models and output the confusion matrices and dynamic penalty ratios for years 5 through 10, execute the main pipeline:

    python main.py

To generate the specific figures and tables used in the manuscript, execute the visualization scripts sequentially:

    python P_value_Table.py
    python Performance_Visualization.py
    python Feature_Importance_CSS.py

All generated figures will be saved as high-resolution PNG/PDF files in the local directory.

## Citation
If you utilize this pipeline or the accompanying dataset in your research, please cite our manuscript:

> *(Citation details will be updated upon publication)*

## License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
