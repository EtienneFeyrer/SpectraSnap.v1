# **SpectraSnap : Fast molecule retrieval**

This repository contains Python code to apply [JESTR](https://doi.org/10.1093/bioinformatics/btaf354) on real world data.

---

## **Overview**
With this code JESTR is applied on the Real-world-dataset of [FLARE](https://doi.org/10.64898/2026.01.27.702086). We provide:
- **Pretrained weights** from the MassSpecGym dataset
- **Precomputed index** for fast retrieval with FAISS available upon request
  - data/precursor_indexes
- **Dataset** for primary and metastatic cancer analysis
  - data/medical/Inhousematch.csv Inhouse matches used as groundtruth for evaluation
  - data/medical/RFA MSMS.txt preprocessed data from MS2 scan
  - data/medical/ST003752_AN006162_Results.txt preprocessed features from MS1 scan
- **Preprocesseing notebook** to standardize real world data
  - datascripts/parse_to_json.py to bring Spectra data into json dictionary
  - datascripts/preprocess_precursor.ipynb filter through MS2 data to match MS1 data
- **Analysis notebook** evaluation and feature analysis
  - datascripts/feature_analysis.ipynb

JESTR uses [PyTorch](https://pytorch.org). The released weights were trained on NVIDIA A100 (CUDA 11.8). Please ensure your environment supports GPU execution.

---

## **Quickstart**

### **1) Set up the environment**
- need to installl conda miniforge
- ensure conda-forge is the only channel and configuration ( > conda config --show channels)
- in the setup_env.sh set the conda.sh location


```bash
# Create and activate environment
#1. Setup environment
bash setup_env.sh
#2. Activate environment
conda activate jestr
```

### **2) Prepare the data**
1. Dowload the large files here and place it into data/medical and data/precursor
2. run cd data_scripts/medical
 python -m parse_to_mgf.py "RFA MSMS.txt" "RFA MSMS.mgf"
3. execute preprocess_precursor.ipynb

### **3) Run SpectraSnap**
1. run cd jestr
2. run python -m params_pub_faiss_inference

### **4) Data Analysis**
execute data/medical/feature_analysis.ipynb


## **Troubleshooting**
- **GPU/driver errors**: Ensure CUDA toolkit and drivers compatible with your GPU (e.g., CUDA 11.8 for A100) are installed and visible to PyTorch.

---

## **License**
This project is licensed under the MIT license.
