from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt

EXPR = Path("data/gse14520_expression.csv")
META = Path("data/gse14520_labels.csv")
OUTF = Path("outputs"); (OUTF/"figures").mkdir(parents=True, exist_ok=True); (OUTF/"tables").mkdir(parents=True, exist_ok=True)

def top_var(X, k=2000):
    var = X.var(axis=1)
    keep = var.sort_values(ascending=False).head(min(k, len(var))).index
    return X.loc[keep]

def main():
    Xfull = pd.read_csv(EXPR, index_col=0)
    meta = pd.read_csv(META)

    samples = [s for s in meta["sample_id"].tolist() if s in Xfull.columns]
    y = (meta.set_index("sample_id").loc[samples, "label"] == "HBV_TUMOR").astype(int).values

    X = Xfull.loc[:, samples]
    X = top_var(X, k=2000).T.values  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train); Xte_s = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=800, class_weight="balanced")
    lr.fit(Xtr_s, y_train)

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)

    def evaluate(model, Xte, name):
        proba = model.predict_proba(Xte)[:,1]
        pred = (proba >= 0.5).astype(int)
        return {"Model": name,
                "ROC-AUC": roc_auc_score(y_test, proba),
                "F1": f1_score(y_test, pred),
                "Accuracy": accuracy_score(y_test, pred),
                "CM": confusion_matrix(y_test, pred, labels=[0,1])}

    res_lr = evaluate(lr, Xte_s, "LogReg (expr)")
    res_rf = evaluate(rf, X_test, "RandForest (expr)")

    qt = pd.DataFrame([{k:v for k,v in res_lr.items() if k!="CM"} | {"Split":"Stratified 70/30"},
                       {k:v for k,v in res_rf.items() if k!="CM"} | {"Split":"Stratified 70/30"}])
    qt.to_csv(OUTF/"tables/quant_table_gse14520.csv", index=False)

    fig = plt.figure()
    RocCurveDisplay.from_estimator(lr, Xte_s, y_test); plt.title("ROC — GSE14520 (test, LR)")
    plt.savefig(OUTF/"figures/roc_gse14520_lr.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    cm = res_lr["CM"]; fig = plt.figure()
    plt.imshow(cm)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i,j]), ha="center", va="center")
    plt.xticks([0,1], ["Pred 0","Pred 1"]); plt.yticks([0,1], ["True 0","True 1"])
    plt.title("Confusion Matrix — GSE14520 (test, LR)"); plt.colorbar()
    plt.savefig(OUTF/"figures/cm_gse14520_lr.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    Xtop = top_var(pd.read_csv(EXPR, index_col=0).loc[:, samples], k=2000)
    feats = Xtop.index.tolist()
    lr_top = pd.Series(lr.coef_[0], index=feats).abs().sort_values(ascending=False).head(10).rename("coef_abs")
    rf_top = pd.Series(rf.feature_importances_, index=feats).sort_values(ascending=False).head(10).rename("importance")
    pd.concat([lr_top, rf_top], axis=1).to_csv(OUTF/"tables/top_features_gse14520.csv")

    print("Saved: outputs/tables/quant_table_gse14520.csv, top_features_gse14520.csv, and figures.")

if __name__ == "__main__":
    main()
