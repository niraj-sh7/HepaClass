from pathlib import Path
import pandas as pd

def load_clinical(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Clinical file not found: {path}")
    df = pd.read_csv(path)
    expected = {"patient_id","hbv_status","age","sex","stage"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Clinical CSV is missing columns: {missing}")
    return df
