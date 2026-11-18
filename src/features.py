"""Variables tarifaires : regroupement en classes, comme dans un tarif réel."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CAT_FEATURES = ["DrivAgeBand", "VehAgeBand", "VehPowerBand", "Area", "VehBrand", "VehGas", "Region"]
NUM_FEATURES = ["BonusMalus", "LogDensity"]


def add_rating_factors(df):
    df = df.copy()
    df["DrivAgeBand"] = pd.cut(df.DrivAge, [17, 21, 25, 30, 40, 50, 60, 70, 120]).astype(str)
    df["VehAgeBand"] = pd.cut(df.VehAge, [-1, 0, 2, 5, 10, 15, 200]).astype(str)
    df["VehPowerBand"] = df.VehPower.clip(upper=12).astype(int).astype(str)
    df["BonusMalus"] = df.BonusMalus.clip(upper=150).astype(float)
    df["LogDensity"] = np.log(df.Density.astype(float))
    return df


def make_preprocessor():
    """One-hot sur les classes (une modalité de référence supprimée), variables continues centrées-réduites."""
    return ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", StandardScaler(), NUM_FEATURES),
    ])
