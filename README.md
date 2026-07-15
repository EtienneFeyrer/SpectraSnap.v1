# **SpectraSnap — Fast MS/MS Molecule Retrieval with JESTR**

SpectraSnap applies the **JESTR** joint‑embedding model to real‑world MS/MS data, including the breast cancer dataset from **FLARE**. It provides pretrained weights, preprocessing scripts, FAISS‑based retrieval, and analysis notebooks.  
**On a consumer‑grade laptop (4 CPU cores), SpectraSnap annotates ~9,000 molecules in under 13 minutes — making it one of the fastest open MS/MS retrieval pipelines available.**

---
---

## Included resources

**Model & index**
- Pretrained JESTR weights (trained on MassSpecGym)
- FAISS precursor index for fast retrieval  
  - `data/precursor_indexes/` (large files — downloaded via GitHub Release)

**Real‑world cancer dataset(not provided)**
- `data/medical/Inhousematch.csv` — ground‑truth matches  
- `data/medical/RFA MSMS.txt` — MS2 scans  
- `data/medical/ST003752_AN006162_Results.txt` — MS1 features  
- `data/medical/Sample_ID.csv` — condition label  

**Preprocessing tools**
- `datascripts/parse_to_json.py` — convert MS2 scans into JSON  
- `datascripts/parse_to_mgf.py` — convert MS2 scans into MGF  
- `datascripts/preprocess_precursor.ipynb` — align MS1/MS2 data  

**Analysis**
- `data/medical/feature_analysis.ipynb` — retrieval evaluation & biological insights  

---

## 1) Set up the environment

**Steps:**
1. Install Miniforge.
2. Configure `conda` to use `conda‑forge` as the only active channel.
3. Edit `setup_env.sh` so it points to your `conda.sh`.
4. Run `setup_env.sh` to create the `jestr` environment.
5. Activate the `jestr` environment.

---

## 2) Get medical and PubChemLite data

The **medical** and **pubchemlite** datasets are only available upon request and are **not** part of the GitHub Release.

**Steps:**
1. Open the provided SharePoint link for SpectraSnap data.
2. Download the `medical` folder.
3. Download the `pubchemlite` folder.
4. Place both folders inside `SpectraSnap/data/`.

---


## 3) Download FAISS precursor index from the release

You can download the FAISS index tarballs from the public release.

**Steps:**
1. Ensure you are inside the `SpectraSnap` repository directory.
2. Use Browser or GitHub CLI to download all release assets matching the precursor index tarballs (pattern `batch_*.tar`) for tag `v1.0.0`.
3. Verify that all tarball files are present in your local `data` directory.

---

### 5) Reconstruct the original `data/precursor_indexes` directory

The tarballs contain the original FAISS precursor index structure.  
You must unpack them into the correct location.

---

### **Option A — Using the provided unpack script**

Run:

```bash
cd data
bash unpack_index.sh
```

### **Option B — direct download**

```bash
bash tar -xf precursor_indexes.tar
```

---

## 6) Optional: Rebuild input files

If you want to regenerate JSON and MGF files:

**Steps:**
```bash
#1.Step
cd datascripts/medical
#2.Step converts MS2 scans without precursor to MGF (SIRIUS input).
python -m parse_to_mgf.py "RFA MSMS.txt" "RFA MSMS.mgf" 
#3.Step convert MS2 scan to json (all with adduct)
python -m parse_to_json.py "RFA MSMS.txt" "RFA MSMS_n.json"
```
4. Execute `datascripts/preprocess_precursor.ipynb` to realign MS1/MS2 data.

---

## 7) Run SpectraSnap retrieval

**Steps:**
```bash
cd jestr
python -m pub_faiss_inference
```

---

## 8) Perform downstream analysis

**Steps:**
1. Open `ground_truth_precursor.ipynb`.
   - update experiment paths with the newly create results (search for "experiments/")
   - Evaluate retrieval performance.
2. Open `data/medical/feature_analysis.ipynb`.
   - update experiment paths with the newly create results (search for "experiments/")
   - Inspect volcano plots and embedding structure.
   - Analyze PubChemLite features and natural product clusters.

---

## License

MIT License.
