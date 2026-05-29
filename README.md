# Interpretable Multi-View MRI Analysis for Knee Injury Detection

## Overview

This project explores the use of deep learning and medical image analysis techniques for knee injury detection using MRI scans from the MRNet dataset.

The primary goal is to develop interpretable machine learning models capable of identifying common knee injuries such as:

* ACL Tears
* Meniscus Tears
* General Knee Abnormalities

In addition to classification, the project investigates visualization and explainability techniques to better understand how deep learning models make diagnostic decisions.

## Dataset

This project uses the MRNet dataset released by Stanford University.

The dataset contains:

* 1,130 training MRI exams
* 120 validation MRI exams
* Sagittal MRI volumes
* Coronal MRI volumes
* Axial MRI volumes

Labels are provided for:

* ACL Tear
* Meniscus Tear
* Abnormality

**Note:** MRI data is not included in this repository.

---

## Current Progress

### Data Exploration

* Explored MRI volume structure
* Compared sagittal, coronal, and axial views
* Built slice-scrolling visualization tools
* Created side-by-side healthy vs injured MRI viewers
* Generated MRI intensity histograms

### Preprocessing

* Analyzed dataset statistics
* Normalized MRI intensities to the range [0,1]
* Investigated volume shape variability across patients

### Deep Learning

* Implemented baseline ACL tear classification CNN
* Trained sagittal-view model
* Evaluated performance using:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
  * ROC-AUC

Current best validation ROC-AUC:

**0.7643**

### Visualization

* Interactive MRI slice viewers
* Thresholding experiments
* 3D MRI volume rendering
* Preliminary heatmap visualization

---

## Project Structure

```text
Programs/
│
├── ACL Classification/
├── Data Exploration/
├── Preprocess/
│
CSV Files/
│
├── train-acl.csv
├── train-meniscus.csv
├── train-abnormal.csv
├── valid-acl.csv
├── valid-meniscus.csv
└── valid-abnormal.csv
```

---

## Future Work

### Classification

* Multi-view MRI classification
* Combined sagittal + coronal + axial models
* Multi-task learning for simultaneous injury prediction

### Explainability

* Grad-CAM heatmaps
* Attention visualization
* Clinically interpretable model outputs

### Segmentation

* Explore ACL localization and segmentation datasets
* Investigate weakly supervised localization methods
* Compare segmentation and classification approaches

---

## Technologies

* Python
* NumPy
* Pandas
* PyTorch
* Matplotlib
* Plotly
* OpenCV
* Scikit-Learn

---

## Repository Notes

The following items are intentionally excluded from the repository:

* MRI datasets
* Normalized MRI datasets
* Virtual environments
* Trained model weights

To reproduce results, download the MRNet dataset separately and place the data in the appropriate project directories.
