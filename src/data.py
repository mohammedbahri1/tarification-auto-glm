"""Chargement et nettoyage des données freMTPL2 (assurance auto, France)."""
import numpy as np
import pandas as pd

OPENML_FREQ = 41214  # freMTPL2freq : un contrat par ligne
OPENML_SEV = 41215   # freMTPL2sev : un sinistre par ligne


def load_fremtpl2():
    """Télécharge freMTPL2freq et freMTPL2sev depuis OpenML (connexion internet requise)."""
    from sklearn.datasets import fetch_openml

    freq = fetch_openml(data_id=OPENML_FREQ, as_frame=True).frame
    sev = fetch_openml(data_id=OPENML_SEV, as_frame=True).frame
    return freq, sev


def make_demo_data(n=60_000, seed=0):
    """Jeu synthétique au même format que freMTPL2, pour tester le code hors ligne."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "IDpol": np.arange(1, n + 1),
        "Exposure": rng.uniform(0.05, 1.0, n),
        "Area": rng.choice(list("ABCDEF"), n),
        "VehPower": rng.integers(4, 16, n),
        "VehAge": rng.integers(0, 25, n),
        "DrivAge": rng.integers(18, 90, n),
        "BonusMalus": np.clip(rng.gamma(2, 15, n) + 50, 50, 230).astype(int),
        "VehBrand": rng.choice([f"B{i}" for i in range(1, 15)], n),
        "VehGas": rng.choice(["Diesel", "Regular"], n),
        "Density": np.exp(rng.normal(6, 2, n)).astype(int) + 1,
        "Region": rng.choice([f"R{i}" for i in range(11, 95, 4)], n),
    })
    lin = (-2.6 + 0.012 * (df.BonusMalus - 50) + 0.25 * (df.DrivAge < 25)
           + 0.05 * np.log(df.Density) - 0.01 * df.VehAge)
    df["ClaimNb"] = rng.poisson(np.exp(lin) * df.Exposure)
    ids = np.repeat(df.IDpol.values, df.ClaimNb.values)
    sev = pd.DataFrame({"IDpol": ids, "ClaimAmount": rng.gamma(1.5, 1400, len(ids))})
    return df, sev


def clean(freq, sev, cap_claims=4, cap_amount=200_000):
    """Nettoyage classique :
    - exposition plafonnée à 1 an, nombre de sinistres plafonné (erreurs de saisie probables) ;
    - montants écrêtés : les sinistres graves sont traités à part (surcharge),
      sinon ils déstabilisent le modèle de coût moyen ;
    - rattachement des montants aux contrats.
    """
    freq = freq.copy()
    freq["IDpol"] = freq["IDpol"].astype(int)
    freq["ClaimNb"] = freq["ClaimNb"].astype(int).clip(upper=cap_claims)
    freq["Exposure"] = freq["Exposure"].astype(float).clip(upper=1.0)

    sev = sev.copy()
    sev["IDpol"] = sev["IDpol"].astype(int)
    sev["ClaimAmount"] = sev["ClaimAmount"].astype(float).clip(upper=cap_amount)
    agg = sev.groupby("IDpol")["ClaimAmount"].agg(["sum", "count"])
    agg.columns = ["ClaimAmount", "NbAmounts"]

    df = freq.merge(agg, left_on="IDpol", right_index=True, how="left")
    df[["ClaimAmount", "NbAmounts"]] = df[["ClaimAmount", "NbAmounts"]].fillna(0)
    # On garde les contrats cohérents : un sinistre déclaré doit avoir un montant
    df = df[~((df.ClaimNb > 0) & (df.ClaimAmount == 0))]
    df["ClaimNb"] = np.where(df.ClaimAmount > 0, np.maximum(df.ClaimNb, 1), 0)
    for c in ["Area", "VehBrand", "VehGas", "Region"]:
        df[c] = df[c].astype(str).str.strip("'")
    return df.reset_index(drop=True)
