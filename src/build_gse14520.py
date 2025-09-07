from pathlib import Path
import re
import pandas as pd
import gzip

DATA_DIR = Path("data/gse14520")
OUT_EXPR = Path("data/gse14520_expression.csv")
OUT_META = Path("data/gse14520_labels.csv")

def pick_series_matrix() -> Path:
    cand = sorted(list(DATA_DIR.glob("GSE14520*series_matrix*.txt")) +
                  list(DATA_DIR.glob("GSE14520*series_matrix*.txt.gz")))
    if not cand:
        raise FileNotFoundError(
            f"No Series Matrix found in {DATA_DIR}. "
            "Put GSE14520-GPL3921_series_matrix.txt(.gz) or GSE14520-GPL571_series_matrix.txt(.gz) there."
        )
    for c in cand:
        if "GPL3921" in c.name:
            return c
    return cand[0]

def read_text_any(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()
    else:
        with open(path, "r", encoding="utf-8", errors="ignore") as fh:
            return fh.readlines()

def parse_series_matrix(path: Path):
    lines = read_text_any(path)

    meta = {}
    table_start = None
    for i, line in enumerate(lines):
        if line.startswith("!Sample_title"):
            meta["title"] = [x.strip().strip('"') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_geo_accession"):
            meta["gsm"] = [x.strip().strip('"') for x in line.split("\t")[1:]]
        elif line.startswith("!Sample_characteristics_ch1"):
            vals = [x.strip().strip('"') for x in line.split("\t")[1:]]
            meta.setdefault("char", []).append(vals)
        elif line.startswith("!series_matrix_table_begin"):
            table_start = i + 1
            break
    if table_start is None:
        raise RuntimeError("Could not find expression table in series matrix.")

    expr_rows, colnames = [], None
    for j in range(table_start, len(lines)):
        if lines[j].startswith("!series_matrix_table_end"):
            break
        parts = [p.strip().strip('"') for p in lines[j].rstrip("\n").split("\t")]
        if colnames is None:
            colnames = parts
        else:
            expr_rows.append(parts)

    df_expr = pd.DataFrame(expr_rows, columns=colnames)
    df_expr.rename(columns={df_expr.columns[0]: "feature"}, inplace=True)
    for c in df_expr.columns[1:]:
        df_expr[c] = pd.to_numeric(df_expr[c], errors="coerce")

    # Build labels from characteristics
    samples = df_expr.columns[1:].tolist()
    char_lines = meta.get("char", [])
    labels = {}
    for idx, gsm in enumerate(meta["gsm"]):
        fields = []
        for row in char_lines:
            if idx < len(row):
                fields.append(row[idx])
        blob = " ".join(fields).lower()
        if re.search(r"adjacent|non[-\s]*tumou?r|normal", blob):
            labels[gsm] = "ADJACENT_NORMAL"
        elif re.search(r"tumou?r", blob):
            labels[gsm] = "HBV_TUMOR"
        else:
            labels[gsm] = None

    meta_df = pd.DataFrame({"sample_id": meta["gsm"], "label": [labels[g] for g in meta["gsm"]]})
    meta_df = meta_df[meta_df["sample_id"].isin(samples)].dropna(subset=["label"])

    keep_cols = ["feature"] + meta_df["sample_id"].tolist()
    df_expr = df_expr[keep_cols].set_index("feature")

    OUT_EXPR.parent.mkdir(parents=True, exist_ok=True)
    df_expr.to_csv(OUT_EXPR)
    meta_df.to_csv(OUT_META, index=False)
    print(f"Using: {path.name}")
    print(f"Saved expression: {OUT_EXPR.resolve()}  shape={df_expr.shape}")
    print(f"Saved labels:     {OUT_META.resolve()}  n={len(meta_df)}  (labels: {meta_df['label'].value_counts().to_dict()})")

if __name__ == "__main__":
    sm = pick_series_matrix()
    parse_series_matrix(sm)
