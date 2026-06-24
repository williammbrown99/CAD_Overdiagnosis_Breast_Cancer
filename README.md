# Risk-Adjusted Machine Learning for Breast Cancer Overdiagnosis

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![DOI](https://img.shields.io/badge/DOI-Zenodo-green.svg)](https://zenodo.org/records/20836585?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIzMGY4MWI5LWJmMjUtNDBmNi05NGFhLTIzMjQ4ODE3NTQwNyIsImRhdGEiOnt9LCJyYW5kb20iOiJmYzA1NWM0YzBjMTY1ODcyNTA5ZTg5MGE5ZTFiMzU2NCJ9.bcbccspIANXQrBkAYO-b9MOSZKPAbXL77guIdGb_sWbEWgqeS_bFZCinIBOCQr-J_lvL7l9wcZ6bGiBstc7pXg)

## Overview
This repository contains the data engineering, machine learning, and statistical visualization pipeline for predicting breast cancer trajectories (Malignant Aggressive vs. Malignant Indolent/Overdiagnosed) across a 5-to-10-year survival continuum. 

To accurately isolate the natural biological progression of the disease without survival bias, the dataset explicitly evaluates untreated patients and includes living cohorts. To ensure patient safety and avoid the undertreatment of lethal disease, this pipeline utilizes an **Extreme Gradient Boosting (XGBoost)** classifier equipped with a dynamic, sliding-scale asymmetric loss function. By heavily penalizing False Positives (misclassifying a lethal tumor as safe) on an intuitive 100-point scale, the algorithm establishes a clinician-calibrated "mathematical firewall" that achieves 1.00 Specificity and zero lethal errors up to a full decade post-diagnosis.

## Data Availability (Zenodo)
Due to GitHub's file size restrictions, the primary SEER clinical dataset (`SEER_11760_alive_and_in_situ_included.csv`) is hosted externally on Zenodo. 

**To run this pipeline locally:**
1. Download the dataset from the Zenodo repository: [Zenodo Data Link](https://zenodo.org/records/20836585?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjIzMGY4MWI5LWJmMjUtNDBmNi05NGFhLTIzMjQ4ODE3NTQwNyIsImRhdGEiOnt9LCJyYW5kb20iOiJmYzA1NWM0YzBjMTY1ODcyNTA5ZTg5MGE5ZTFiMzU2NCJ9.bcbccspIANXQrBkAYO-b9MOSZKPAbXL77guIdGb_sWbEWgqeS_bFZCinIBOCQr-J_lvL7l9wcZ6bGiBstc7pXg)
2. Create a folder named `INPUT/` in the root directory of this project.
3. Place the downloaded `.csv` file directly into the `INPUT/` folder.

*(Note: The `INPUT/` directory and all `.csv` files are added to `.gitignore` to prevent accidental large-file commits).*

## Repository Structure
The codebase is divided into modular scripts corresponding to the methodology and results of the study:

* **`main.py`**: The core XGBoost classification script. Handles feature engineering (sequential backfilling, NaN handling for sparsity-aware splitting), temporal thresholding, and runs the sliding-scale optimization to export the raw performance metrics.
* **`Cost_Optimizations.py`**: Analyzes the sliding-scale data to extract the exact thresholds for the three primary clinical strategies: Max Clinical Safety (0 FP), 1-to-1 Balanced Cost, and Max Indolence (Bounded by a $\ge$ 50% Specificity constraint).
* **`Clincal_Trade_Off.py`**: Executes the Marginal Benefit Analysis, calculating the clinical exchange rate of avoided overdiagnoses versus introduced lethal errors, and plots the exponential decay curves.
* **`Master_Optimization_All_Years.csv`**: The exported raw optimization data utilized by the trade-off and plotting scripts.
* **`Performance_Visualization.py`**: Generates publication-ready 1x3 grouped bar charts comparing the three cost-optimization strategies across the temporal continuum.
* **`P_value_Table.py`**: Conducts baseline demographic and statistical analyses (Mann-Whitney U tests for continuous variables, Chi-square for categorical) to generate the baseline characteristics table.
* **`Feature_Importance_CSS.py`**: Extracts XGBoost relative feature importance metrics and generates Kaplan-Meier Cancer-Specific Survival (CSS) curves with log-rank testing to clinically validate the safety of the predicted cohorts.

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
Ensure your clinical data is correctly placed in `INPUT/SEER_11760_alive_and_in_situ_included.csv`. 

To train the models, extract the cost optimizations, and generate the master dataset for the visualizations, execute the pipeline:

    python main.py
    python Cost_Optimizations.py

To generate the specific figures and tables used in the manuscript, execute the visualization and statistical scripts:

    python Clincal_Trade_Off.py
    python Performance_Visualization.py
    python P_value_Table.py
    python Feature_Importance_CSS.py

All generated figures will be saved as high-resolution PNG/PDF files in the local directory.

## Citation
If you utilize this pipeline or the accompanying dataset in your research, please cite our manuscript:

> *(Citation details will be updated upon publication)*

## License
This project is licensed under the GNU General Public License v3.0 - see the `LICENSE` file for details.
