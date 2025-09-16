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

---
## Week 3

### Detailed Tasks

#### Data & Split (patient-level)
- Load the prepared tabular features and labels:
  - `radiomics.csv` (feature matrix), `labels.csv` (targets/metadata)
- Create a **patient-level** split (e.g., 70/30) using unique `subject_id`
  - Save: `train_labels.csv`, `test_labels.csv`
- Inspect class balance on the resulting `train_df` / `test_df`

#### Features
- From the radiomics table, drop ID/meta columns (e.g., `subject_id`, `study_id`, paths/aux fields)
- Define `X_train`, `y_train`, `X_test`, `y_test`

#### Models & Cross-Validation
- Baselines: **Logistic Regression**, **Random Forest**, **K-Nearest Neighbors**
- Cross-validation: **StratifiedKFold(n_splits=3, shuffle=True, random_state=42)**
- For each model, build a single pipeline:  
  **StandardScaler → SMOTE → classifier** (using `imblearn.Pipeline`)
- Hyperparameter search via **RandomizedSearchCV(n_iter=10, scoring='recall', refit=True)** with small grids:
  - Logistic Regression: `classifier__C ∈ {0.01, 1, 10}`, `classifier__solver ∈ {lbfgs, liblinear}`, `max_iter=1000`
  - Random Forest: `classifier__n_estimators ∈ {10, 30, 50}`, `classifier__max_depth ∈ {None, 5, 10}`
  - KNN: `classifier__n_neighbors ∈ {5, 10, 20}`

#### Selection, Persistence, and Reporting
- Record mean CV **recall** ± std per model (table printed in-notebook)
- Persist the **best pipeline** (scaler + SMOTE + classifier) per model with `joblib`
- Pick one model for downstream evaluation by loading its persisted pipeline
- On the **held-out test** set:
  - Print `classification_report`
  - Plot **ROC** and **DET** curves
  - Plot **confusion matrix**

---

### Expected Outcomes
- Split metadata: `train_labels.csv`, `test_labels.csv`
- Persisted pipelines (trained with SMOTE inside CV):
  - `best_model_smote_logistic_regression.pkl`
  - `best_model_smote_random_forest.pkl`
  - `best_model_smote_nearest_neighbors.pkl`
- In-notebook outputs:
  - CV comparison table (mean recall ± std)
  - Test **classification report**
  - **ROC** / **DET** plots and **confusion matrix**

---
## Week 4

### Detailed Tasks

#### Dataset (patient-level, images)
- Download inputs used in this notebook:
  - `MIMIC-CXR-png.zip` (PNG images) → extracted into `./MIMIC-CXR-png/`
  - `radiomics.csv`, `labels.csv`, and the Week-3 split files `train_labels.csv`, `test_labels.csv`
- Build **patient-level** labels by aggregating to one label per `subject_id` (`max` over 0/1 rows).
- Recreate splits and carve out a **validation** set from the training patients (stratified by label).
- Construct image paths (`img_path`) to `./MIMIC-CXR-png/files-png/<dicom_id>.png`.
- Verify file existence for train/val/test and drop any rows with missing images.

#### Classification (EfficientNet-B4, fine-tuning)
- Dataset loader for classification (grayscale → 1-channel), with:
  - `Resize` → `CenterCrop(image_size)` → `ToTensor` → `Normalize`
- Model: **`torchvision.models.efficientnet_b4`** (ImageNet weights), final classifier replaced with a **2-class** head.
- Training:
  - Loss: **CrossEntropyLoss**
  - Optimizer: **Adam**
  - LR scheduler (stepped each epoch)
  - Selection by **validation accuracy**; best weights saved as  
    `./efficientnetb4-pathology-cxr.pth`

#### Segmentation + Classification (U-Net multitask)
- Dataset loader for segmentation uses images + masks (`./MIMIC-CXR-png/segmentation/...`) if present:
  - Image transforms: `Resize(size=256)` → `ToTensor` → `Normalize`
  - Mask transforms: `Resize(size=256)` → `ToTensor`
- Model: custom **U-Net** with
  - **Segmentation head** (binary mask)
  - **Classification head** (global pooling + linear → 2 classes)
- Training (multi-task):
  - Losses: **BCEWithLogitsLoss** (seg) + **CrossEntropyLoss** (cls)
  - Optimizer: **Adam**
  - Validation prints per-epoch classification loss, segmentation loss, and accuracy
  - Best (by val classification loss) saved as  
    `./unet_seg_cls_pathology_cxr.pth`

#### Evaluation
- Test-time inference utilities to collect **softmax probabilities** and **labels**.
- Plot **ROC curves** on the test set comparing:
  - **EfficientNet-B4** (fine-tuned classifier)
  - **U-Net (multitask) classifier head**
- (Additionally in-notebook) load Week-3 `.pkl` baselines and render **ROC/DET** overlays for comparison.

---

### Expected Outcomes
- Data split artifacts (reused): `train_labels.csv`, `test_labels.csv`
- Saved models:
  - `efficientnetb4-pathology-cxr.pth`
  - `unet_seg_cls_pathology_cxr.pth`
- In-notebook outputs:
  - Sample image visualizations from the training loader
  - Per-epoch validation prints (loss/accuracy) for both models
  - **ROC** plot(s) comparing EfficientNet-B4 vs U-Net; optional overlay with Week-3 baselines (ROC/DET)
---
## Week 5

### Detailed Tasks

#### Dataset exploration & fairness proxies
- Reload the **patient-level** splits from Week 3/4 (`train_df`, `val_df`, `test_df`) and attach image paths.
- Derive metadata-based subgroup attributes from DICOM tags:
  - **projection** (`PA`, `AP`, `LAT`, `OTHER`) parsed from `ViewCodeSequence_CodeMeaning`
  - **is_portable** (boolean) from portable markers in metadata
  - auxiliary flags: `has_lateral`, `projection_portable`, `procedure_family`
- Inspect subgroup distributions (`value_counts`) and visualize **disease prevalence** (positive rate) as a heatmap over `(projection × is_portable)`.

#### Performance across subgroups
- Load the Week-4 classifier (**EfficientNet-B4**, 2-class head) from disk.
- Define `evaluate_subgroup(...)` to compute, per subgroup:
  - **AUC**, **F1**, **recall**, **precision**, **accuracy**, and **n_samples**
- Iterate over test subgroups `(projection, is_portable)`, collect metrics into a table, and display results.

#### Mitigation A — Targeted oversampling (implemented end-to-end)
- Identify the **underperforming subgroup** (`projection="PA" & is_portable=False`).
- Create a balanced training set by **oversampling with replacement** that subgroup  
  (e.g., `oversample_factor=2`) → `train_df_fairer`.
- Re-train EfficientNet-B4 on `train_df_fairer` with the existing training loop:
  - Loss: `CrossEntropyLoss`
  - Optimizer: `Adam`
  - Scheduler: `CosineAnnealingLR`
  - Model selection by **validation accuracy**
- Save weights as: `./fair-efficientnetb4-pathology-cxr2.pth`
- Re-evaluate subgroup metrics on the **test** set and compare to the original model.

#### (Scaffold) Mitigation B — Weighted loss (prepared in notebook)
- Add a `sample_weight` column to `train_df` assigning higher weight to the target subgroup.
- Switch loss definition to `CrossEntropyLoss(reduction='none')` to enable per-sample weighting.

#### Visualization
- Plot the **prevalence heatmap** (subgroup positive rates).
- Render **ROC** per subgroup from collected probabilities.
- Generate **Grad-CAM++** overlays for representative samples to sanity-check model focus.

---

### Expected Outcomes
- In-notebook tables/plots:
  - Subgroup counts and **prevalence heatmap**
  - Per-subgroup **AUC/F1/recall/precision/accuracy** (before vs after oversampling)
  - Grad-CAM++ visualizations
- Saved models:
  - Original (from Week 4): `efficientnetb4-pathology-cxr2.pth` (loaded)
  - **Oversampled** retrain: `fair-efficientnetb4-pathology-cxr2.pth`
  - (Prototype) Weighted-loss run: `fair-weights-efficientnetb4-pathology-cxr2.pth` *(weights prepared; loss not yet applying them)*
---
## Week 6

### Detailed Tasks

#### Setup & Data
- Install and import libs used in-notebook: `torch`, `torchvision`, `transformers`, `peft`, `pandas`, `scikit-learn`, etc.
- Reuse the Week-3/4 **patient-level** splits and image paths.
- Build lightweight metadata features from DICOM-derived proxies:
  - `projection` → one-hot (e.g., `AP`, `PA`)
  - `is_portable` → boolean flag
- Helper: `preprocess_tabular(projection, is_portable)` returns a small tabular tensor.

---

### Part A — Multi-Modal Fusion (from scratch)

#### Datasets & Loaders
- `MultiModalScratchDataset`: returns `(image_tensor, tabular_tensor, label)`  
  - Image transforms: `Resize(224×224)`, `ToTensor`, normalization.
  - Tabular: one-hot `projection` + `is_portable` (size = **TABULAR_FEATURE_SIZE**).
- Train/val/test `DataLoader`s created with reasonable batch sizes.

#### Models
- **Backbone**: `resnet50(pretrained=True)` with frozen weights; `fc` replaced by `Identity()` to output image features.
- **ConcatFusionModel**:  
  - MLP on tabular (→ 64) + concat with image features → MLP classifier → sigmoid.
- **CrossAttentionModel**:  
  - Project image features → 512, tabular → 512, apply **MultiheadAttention** (num_heads=4) between them; concat(attended_img, tabular) → classifier.

#### Training & Evaluation
- Utility: `train_and_evaluate(model, name, train_loader, test_loader)`
  - Optimizer: `Adam` (lr=1e-4), loss: `BCEWithLogitsLoss`
  - **2 epochs** (quick run)
  - Metrics: **AUC**, **Accuracy**, **F1** (threshold 0.5)
- Append per-model dicts to `results`.

---

### Part B — Foundation Model Adaptation

#### Base model
- **MedCLIP**:  
  - `processor = AutoProcessor.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")`  
  - `medclip_model = AutoModel.from_pretrained("flaviagiammarino/pubmed-clip-vit-base-patch32")`
  - Freeze **vision backbone** (`vision_model`) and **visual projection**.
- `MultiModalFMDataset`: uses `processor(images=…)` to produce `pixel_values`; returns `(pixel_values, tabular_tensor, label)`.
- DataLoaders: batch size 16.

#### Adaptation variants (image+tabular fusion)
- **LinearProbeModel**  
  - Use frozen MedCLIP image embedding; concat with tabular; small MLP → 1-logit output.
- **PartialFTModel**  
  - Similar head; minimal trainable parameters (classifier only), backbone frozen.
- **LoRAModel** (PEFT)  
  - Apply **LoRA** to (selected) CLIP layers while keeping the backbone otherwise frozen; concat frozen image embedding with tabular → small MLP.

#### Training & Evaluation
- `train_and_evaluate_fm(model, name, fm_train_loader, fm_test_loader)`
  - Optimizer: `Adam` (lr=5e-4), loss: `BCEWithLogitsLoss`
  - **2 epochs** per variant
  - Metrics: **AUC**, **Accuracy**, **F1**
- Append per-model dicts (including a count of trainable params) to `results`.

---

### Final Comparison
- Aggregate `results` into a DataFrame and print a compact table: columns **AUC / ACC / F1 / params** (params formatted with thousands separators).
- Visualize bar charts for **AUC**, **Accuracy**, **F1** across all trained models (scratch fusion + foundation adaptations).

---

### Expected Outcomes
- In-notebook artifacts:
  - `results` table comparing **Concatenation (Scratch)**, **Cross-Attention (Scratch)**, **Linear Probing (MedCLIP)**, **Partial FT (MedCLIP)**, **LoRA (MedCLIP)**.
  - Bar plots for **AUC**, **Accuracy**, **F1**.

