# Project: Chest X-ray Biomarkers for Diagnosis

## Overview

In this project, you will apply techniques and methods learned during the practical sessions to address a clinical problem using medical imaging data.

---

## Dataset

We will use a subset of the **MIMIC-CXR dataset**. The MIMIC Chest X-ray (MIMIC-CXR) Database v2.0.0 is a large, publicly available dataset of chest radiographs in DICOM format, accompanied by free-text radiology reports. It contains 377,110 images from 227,835 radiographic studies conducted at the Beth Israel Deaconess Medical Center in Boston, MA.

The dataset has been de-identified in compliance with the US Health Insurance Portability and Accountability Act of 1996 (HIPAA) Safe Harbor requirements. All protected health information (PHI) has been removed.

More details: [https://mimic.mit.edu/docs/iv/modules/cxr/](https://mimic.mit.edu/docs/iv/modules/cxr/)

A subset of this dataset with **XXX participants** and **XXX images** is provided. You can download it from the following link:  
`[INSERT LINK HERE]`  
Alternatively, a code cell in the first notebook will download the required data.

---

## Work Plan

This project is divided into weekly tasks aligned with the lessons:

### Week 2: Explore, Clean, and Organize the Data

- Load and examine the dataset  
- Clean metadata (remove NaNs, anonymize DICOMs if needed)  
- Extract features from images and/or reports  
- Standardize features and identify outliers  
- Visualize and reduce dimensions using PCA or t-SNE  
- Select relevant features and split the dataset (train/val/test)

### Week 3: Machine Learning Models

- Train models for disease diagnosis  
- Optimize hyperparameters with cross-validation  
- Evaluate models with appropriate metrics  
- Write a benchmark analysis

### Week 4: Deep Learning Models

- Train a CNN on chest X-ray images  
- Optimize hyperparameters  
- Evaluate performance using proper metrics

### Week 5: Fairness and Uncertainty

- Use a pre-trained model (e.g., TorchXRayVision) to segment anatomical structures  
- Evaluate model performance with segmentation masks  
- Investigate Clever Hans effects and retrain if necessary  
- Compute fairness metrics using demographic data

### Week 6: Foundation Models

- Extract features using GLORIA or similar foundation models  
- Train a linear probe for disease diagnosis  
- Compare performance with:  
  - Machine learning models  
  - CNNs (with/without masking)  
  - Foundation model features

---

## Week 2 

### Detailed Tasks

#### Data Exploration

- Load the dataset  
- Identify available data (images, segmentation masks, radiology reports)  
- Check for missing participants/data   
- Drop participants having one or more of these previously mentioned elements missing. 

#### Metadata Cleaning

- Identify the target label column `icd_code` in the diagnosis table
- Drop participants with missing target labels  
- Impute or drop missing values in other features  
- Remove any sensitive info from DICOMs if necessary  
- Save cleaned metadata in a `metadata_clean.csv` file

#### Feature Extraction

- Use segmentation masks to extract radiomics  
- Extract data from radiology reports  
- Save new features in a `features_raw.csv` file

#### Standardization and Outlier Detection

- Standardize all numeric features  
- Identify and handle outliers

#### Dimensionality Reduction

- Apply PCA and/or t-SNE  
- Examine loadings and explained variance  
- Select relevant features  
- Save selected features in `features_selected.csv`

---

### Expected Outcomes

- `metadata_clean.csv`: Metadata from participants having at least:  
  - 1 chest X-ray image  
  - 1 segmentation mask  
  - A valid (non-NaN) target label


- `features_raw.csv`: Extracted features from images and/or reports  
    
- `features_selected.csv`: Final feature set after PCA/t-SNE selection
