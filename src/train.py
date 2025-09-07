from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

@dataclass
class Models:
    lr: LogisticRegression
    rf: RandomForestClassifier
    scaler: StandardScaler

def split_data(df: pd.DataFrame, test_size=0.3, seed=42):
    X = df[["age","sex_binary","stage_numeric"]].astype(float).values
    y = df["label"].astype(int).values
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=y)

def train_models(X_train, y_train) -> Models:
    scaler = StandardScaler()
    Xtr_s = scaler.fit_transform(X_train)

    lr = LogisticRegression(max_iter=500, class_weight="balanced")
    lr.fit(Xtr_s, y_train)

    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced_subsample", random_state=42)
    rf.fit(X_train, y_train)

    return Models(lr=lr, rf=rf, scaler=scaler)
