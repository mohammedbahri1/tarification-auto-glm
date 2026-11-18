"""Modèles fréquence x coût moyen et indicateurs de performance."""
import numpy as np
import pandas as pd
from sklearn.linear_model import GammaRegressor, PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.pipeline import Pipeline

from .features import make_preprocessor

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # compatibilité numpy 1.x / 2.x


def fit_frequency(train, alpha=1e-4):
    """GLM Poisson, lien log : E[N] = exposition x exp(X beta).
    On modélise N / exposition avec l'exposition en poids : équivalent à un offset log(exposition).
    """
    model = Pipeline([("prep", make_preprocessor()),
                      ("glm", PoissonRegressor(alpha=alpha, max_iter=1000))])
    model.fit(train, train.ClaimNb / train.Exposure, glm__sample_weight=train.Exposure)
    return model


def fit_severity(train, alpha=1e-3):
    """GLM Gamma, lien log, sur les contrats sinistrés : coût moyen pondéré par le nombre de sinistres."""
    claims = train[train.ClaimNb > 0]
    model = Pipeline([("prep", make_preprocessor()),
                      ("glm", GammaRegressor(alpha=alpha, max_iter=1000))])
    model.fit(claims, claims.ClaimAmount / claims.ClaimNb, glm__sample_weight=claims.ClaimNb)
    return model


def pure_premium(freq_model, sev_model, df):
    """Prime pure annuelle = fréquence annuelle x coût moyen."""
    return freq_model.predict(df) * sev_model.predict(df)


def poisson_deviance(df, freq_pred):
    return mean_poisson_deviance(df.ClaimNb / df.Exposure, freq_pred, sample_weight=df.Exposure)


def lorenz_curve(y_true, y_pred, exposure):
    """Contrats triés du moins risqué au plus risqué selon le modèle ; part cumulée de la charge."""
    order = np.argsort(y_pred)
    cum_exp = np.cumsum(exposure[order]) / exposure.sum()
    cum_loss = np.cumsum(y_true[order]) / y_true.sum()
    return np.r_[0, cum_exp], np.r_[0, cum_loss]


def gini(y_true, y_pred, exposure):
    """Gini = 1 - 2 x aire sous la courbe de Lorenz : capacité du tarif à classer les risques."""
    x, y = lorenz_curve(np.asarray(y_true), np.asarray(y_pred), np.asarray(exposure))
    return 1 - 2 * _trapezoid(y, x)


def relativities(freq_model):
    """Relativités exp(beta) par modalité : lecture « tarif » du GLM (1 = modalité de référence)."""
    prep = freq_model.named_steps["prep"]
    coefs = freq_model.named_steps["glm"].coef_
    names = prep.get_feature_names_out()
    tab = pd.DataFrame({"variable": names, "coef": coefs})
    tab["relativite"] = np.exp(tab.coef)
    tab["variable"] = tab.variable.str.replace("cat__", "").str.replace("num__", "")
    return tab


def lift_table(df, pred, n_bins=10):
    """Déciles de prime prédite : charge observée vs prédite par exposition."""
    d = pd.DataFrame({"pred": pred * df.Exposure, "obs": df.ClaimAmount, "exp": df.Exposure})
    d["decile"] = pd.qcut(pred, n_bins, labels=False, duplicates="drop") + 1
    g = d.groupby("decile")[["pred", "obs", "exp"]].sum()
    g["prime_pure_predite"] = g.pred / g.exp
    g["prime_pure_observee"] = g.obs / g.exp
    return g[["exp", "prime_pure_predite", "prime_pure_observee"]].rename(columns={"exp": "exposition"})
