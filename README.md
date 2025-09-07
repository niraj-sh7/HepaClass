# HepaClass — HBV-HCC Classifier (TCGA-LIHC)

**Goal:** Separate HBV-driven hepatocellular carcinoma (HBV-HCC) from non‑viral HCC with a lightweight, reproducible baseline.

## Quickstart
```bash
# 1) Create and activate a virtual env (recommended)
python -m venv .venv && source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2) Install deps
pip install -r requirements.txt

# 3) Add data
# Place a clinical CSV at data/tcga_lihc_clinical.csv with columns:
# patient_id,hbv_status,age,sex,stage
# - hbv_status: HBV or NONVIRAL  (others will be dropped)
# - sex: M/F
# - stage: I,II,III,IV  (or strings containing these numerals)

# 4) Run
python main.py

# Outputs -> ./outputs/{figures, tables}
```

## Files
- `main.py` — runs the whole baseline.
- `src/load.py` — loading CSVs.
- `src/preprocess.py` — filtering labels, simple cleaning/encoding.
- `src/train.py` — trains LR and RF.
- `src/eval.py` — metrics + confusion matrix + ROC.
- `src/plots.py` — plotting helpers.
- `outputs/` — saved figures and tables for Devpost/report.
- `report/` — Devpost blurb + quant table template.

## Notes
- Keep **patients unique** between train/test (done by the split).
- We use **class weights** to handle imbalance.
- Extend by adding expression data and setting a combined feature set.
