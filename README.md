# **JESTR: Joint Embedding Space Technique for Ranking Candidate Molecules for the Annotation of Untargeted Metabolomics Data with FAISS implementation**

This repository contains Python code to train and test the JESTR model, and run inference with FAISS

---

## **Overview**
JESTR learns a joint embedding of spectra and candidate molecules to rank candidates for untargeted metabolomics annotation. We provide:
- **Pretrained weights** for the MassSpecGym dataset
- **Sample data** from the MassSpecGym dataset including preprocessing
- **Training and testing scripts**

JESTR uses [PyTorch](https://pytorch.org). The released weights were trained on NVIDIA A100 (CUDA 11.8). Please ensure your environment supports GPU execution.

---
## **Quickstart**

### **1) Set up the environment**
- need to installl conda miniforge
- ensure conda-forge is the only channel and configuration ( > conda config --show channels)
- Create environment using  mini-forge and pip

```bash
# Create and activate environment
#1. Setup environment
bash setup_env.sh
#2. Activate environment
conda activate jestr
```

### **2) Run the tests:**
```bash
pytest -v
```

### **3) Demo**
- take a look into the demo.ipynb and follow the instructions
- demo will explain input files for inference


### **4) Configure `params.yaml`**
Create or edit `params.yaml`. Key fields:
- `run_name`: experiement directory name
- `checkpoint_pth_spec_enc`: spectra encoder checkpoint
- `checkpoint_pth_mol_enc`: molecule encoder checkpoint
- `candidates_pth`: path to your identifier_to_candidate.json
- `dataset_pth`: path to your data.tsv

## **Training**
1. Prepare data in the same format as the provided data/sample/data.tsv or the [MassSpecGym data](https://huggingface.co/datasets/roman-bushuiev/MassSpecGym).
2. Edit `params.yaml`, make sure all checkpoint paths are empty if training from scratch. 

3. Train and evaluate
```bash
# train
python train.py

# test/evaluate
python inference.py
```

## **Notes on licensing and datasets**
We release the NPLIB1 dataset and pretrained weights. Other datasets may be subject to licensing restrictions and are not included. If you test other datasets, confirm you have appropriate licenses and access.

---

## **Troubleshooting**
- **GPU/driver errors**: Ensure CUDA toolkit and drivers compatible with your GPU are installed and visible to PyTorch.
- **Dependency conflicts**: torchdata has to be <= 0.7.0 because of DGL graphbolt, change to faiss-gpu tested and possible

---

## **License**
This project is licensed under the MIT license.

