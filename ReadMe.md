# 📈 Can We Save CAPM? Predicted Beta Shows Statistical Significance

## Overview
This repository contains the downstream analysis code from my thesis project.  
We investigate whether **Random Forest-predicted betas** can recover the predictive power of CAPM by showing significant correlation with future stock returns.

The project includes:
- Predicting beta ($\beta$) using Random Forest
- Performing **Fama-MacBeth style regressions** on predicted beta
- Computing **Newey-West adjusted** t-statistics
- Visualizing rolling p-values over time

## Dataset

🔗 Download data here:  
[Google Drive Link](https://drive.google.com/drive/folders/1m9imy7ZAv9dD1f1mUNBJ2MBL6IwSYa0b?usp=sharing)

## How It Works

1. **Load Data**  
   Load CRSP and Compustat merged features.

2. **Predict Beta**  
   Use Random Forest to predict next-period beta (`b12d`) based on firm characteristics.

3. **Downstream Regression**  
   Regress future returns (`r12`) on:
   - True beta (calculated contemporaneously)
   - Predicted beta (from Random Forest)
   - Previous beta (historical estimate)

4. **Significance Testing**  
   Compute **Newey-West** adjusted standard errors and t-statistics to assess the predictive power.

5. **Visualization**  
   Plot rolling p-values over time for true beta, predicted beta, and previous beta.

## Key Findings

- **True betas** show strong significance, as expected.
- **Predicted betas** (from Random Forest) show **borderline significance (around 10% p-value)**, better than using naive historical betas.
- Prediction performance **deteriorates during crisis periods**.

## Folder Structure

```
.
├── data/          # Folder for CRSP and Compustat data (download separately)
├── utils.py       # Helper functions for loading and preparing data
├── ml_methods.py  # Random Forest model wrapper
├── downstream.ipynb  # Main notebook performing prediction and downstream regression
```

## Requirements

- Python 3.10+
- pandas
- numpy
- scikit-learn
- statsmodels
- matplotlib
- scipy

Install dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib scipy
```

## Acknowledgment
This project was completed as part of my empirical asset pricing research on testing the validity of CAPM under non-linear beta modeling.

---