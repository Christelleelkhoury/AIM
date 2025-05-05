# AI for Medical Diagnosis & Prediction: Practical Exercises

This repository contains all hands-on practical exercises for the **AI for Medical Diagnosis and Prediction** course. Each week's folder includes starter Jupyter notebooks, sample data pipelines, and instructions needed to complete the formative labs.

## 📂 Repository Structure

```bash
/
├── environment.yml               			# Conda environment definition
├── requirements.txt              			# pip dependencies
├── data/                         			# Sample data subsets (placeholders)
│   ├── ReMIND/ 							# Brain MRI DICOMs
│   ├── PKG - ReMIND_NRRD_Seg_Sep_2023/		# Tumor segmentation in NRRD format
│   ├── mimic_cxr/                			# Chest X‑ray DICOMs + reports
│   └── mimic_iv/                 			# EHR CSV snippets
├── notebooks/                    			# Jupyter notebooks by week
│   ├── week1/
│   └── week2/
	│   ├── lesson1.ipynb         			# Practical exercises for lesson #1 of week #2
	│   └── lesson2.ipynb		  			# Practical exercises for lesson #2 of week #2
├── utils/                        			# Helper scripts
│   ├── data_loader.py            			# DICOM & CSV loading utilities
│   └── metrics.py                			# Clinical metric 
├── LICENSE
└── README.md                     			# This file
```

## 🚀 Getting Started

### Step 1: Clone the Repository
```bash
git clone https://github.com/albarqounilab/AIM.git
cd AIM
```

### Step 1: Clone the Repository

Using Conda (recommended)
```bash
conda env create -f environment.yml
conda activate aim-course
```
Using pip
```bash
pip install -r requirements.txt
```

### Step 3: Prepare Sample Data




### Step 4: Launch JupyterLab OR Google Colab 

```bash
jupyter lab
```
Open the notebook corresponding to the week you're working on, and follow the instructions within each notebook.

## 🛠️ Utilities Provided

- utils/data_loader.py: Load and preprocess DICOM images and CSV tables.
- utils/metrics.py: Calculate clinical metrics such as sensitivity, specificity, AUC, and Dice scores.
