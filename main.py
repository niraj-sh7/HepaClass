from pathlib import Path
from src.load import load_clinical
from src.preprocess import clean_and_label
from src.train import split_data, train_models
from src.eval import evaluate_and_save

DATA = Path("data/tcga_lihc_clinical.csv")
OUT  = Path("outputs")

def main():
    clin = load_clinical(DATA)
    df = clean_and_label(clin)
    X_train, X_test, y_train, y_test = split_data(df, test_size=0.3, seed=42)
    models = train_models(X_train, y_train)
    qt, top = evaluate_and_save(models, X_train, X_test, y_train, y_test, OUT)
    print("Saved: outputs/tables/quant_table.csv, outputs/tables/top_features.csv, outputs/figures/*")

if __name__ == "__main__":
    main()
