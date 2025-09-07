import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
from pathlib import Path

def evaluate_and_save(models, X_train, X_test, y_train, y_test, out_dir: str | Path):
    out_dir = Path(out_dir); (out_dir/"figures").mkdir(parents=True, exist_ok=True); (out_dir/"tables").mkdir(parents=True, exist_ok=True)

    Xtr_s = models.scaler.transform(X_train)
    Xte_s = models.scaler.transform(X_test)

    def metrics_from(model, Xte, name):
        proba = model.predict_proba(Xte)[:,1]
        pred = (proba >= 0.5).astype(int)
        return {
            "Model": name,
            "ROC-AUC": roc_auc_score(y_test, proba),
            "F1": f1_score(y_test, pred),
            "Accuracy": accuracy_score(y_test, pred),
            "CM": confusion_matrix(y_test, pred, labels=[0,1])
        }

    res_lr = metrics_from(models.lr, Xte_s, "Logistic Regression (L2)")
    res_rf = metrics_from(models.rf, X_test, "Random Forest")
    qt = pd.DataFrame([
        {k:v for k,v in res_lr.items() if k != "CM"} | {"Split":"Stratified 70/30"},
        {k:v for k,v in res_rf.items() if k != "CM"} | {"Split":"Stratified 70/30"},
    ])
    qt.to_csv(out_dir/"tables/quant_table.csv", index=False)

    fig = plt.figure()
    RocCurveDisplay.from_estimator(models.lr, Xte_s, y_test)
    plt.title("ROC — Logistic Regression (test)")
    plt.savefig(out_dir/"figures/roc_lr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    cm = res_lr["CM"]
    fig = plt.figure()
    import matplotlib.ticker as mticker
    plt.imshow(cm, cmap="Blues")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.xticks([0,1], ["Pred 0","Pred 1"])
    plt.yticks([0,1], ["True 0","True 1"])
    plt.title("Confusion Matrix — Logistic Regression (test)")
    plt.colorbar()
    plt.savefig(out_dir/"figures/cm_lr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    feat_names = ["age","sex_binary","stage_numeric"]

    import numpy as np
    lr_coefs = pd.Series(models.lr.coef_[0], index=feat_names).abs().sort_values(ascending=False).head(10).rename("coef_abs")
    rf_imps  = pd.Series(models.rf.feature_importances_, index=feat_names).sort_values(ascending=False).head(10).rename("importance")
    top = pd.concat([lr_coefs, rf_imps], axis=1)
    top.to_csv(out_dir/"tables/top_features.csv")
    return qt, top
