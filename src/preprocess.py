import re
import pandas as pd

def _stage_to_num(val: str) -> float:
    if pd.isna(val): return None
    s = str(val).upper()
    for k,v in {"I":1,"II":2,"III":3,"IV":4}.items():
        if re.search(rf'\b{k}\b', s):
            return v
    m = re.search(r'(1|2|3|4)', s)
    return float(m.group(1)) if m else None

def clean_and_label(clin: pd.DataFrame) -> pd.DataFrame:
    df = clin.copy()

    df = df[df["hbv_status"].isin(["HBV","NONVIRAL"])].copy()
    df["label"] = (df["hbv_status"] == "HBV").astype(int)

    df["sex_binary"] = df["sex"].astype(str).str.upper().map({"M":1,"MALE":1,"F":0,"FEMALE":0})
    df["stage_numeric"] = df["stage"].apply(_stage_to_num)

    feats = ["age","sex_binary","stage_numeric"]
    df = df.dropna(subset=feats + ["label"])

    return df[["patient_id","label"] + feats]
