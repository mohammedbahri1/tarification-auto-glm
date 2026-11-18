"""Tarification auto par GLM fréquence x coût moyen sur freMTPL2.

Usage :
    python main.py           # données réelles freMTPL2 (téléchargement OpenML)
    python main.py --demo    # données synthétiques, sans connexion
"""
import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from src.data import clean, load_fremtpl2, make_demo_data
from src.features import add_rating_factors
from src.models import (fit_frequency, fit_severity, gini, lift_table, lorenz_curve,
                        poisson_deviance, pure_premium, relativities)

OUT = Path("outputs")


def main(demo=False):
    OUT.mkdir(exist_ok=True)
    freq, sev = make_demo_data() if demo else load_fremtpl2()
    df = add_rating_factors(clean(freq, sev))
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    # 1. Statistiques descriptives du portefeuille
    print(f"Contrats : {len(df):,} | Exposition : {df.Exposure.sum():,.0f} années")
    print(f"Fréquence observée : {df.ClaimNb.sum() / df.Exposure.sum():.4f} sinistre/an")
    print(f"Coût moyen observé : {df.ClaimAmount.sum() / df.ClaimNb.sum():,.0f} €")
    print(f"Prime pure observée : {df.ClaimAmount.sum() / df.Exposure.sum():,.1f} €/an\n")

    # 2. Modèles
    f_model = fit_frequency(train)
    s_model = fit_severity(train)

    # 3. Évaluation sur l'échantillon test
    f_null = np.full(len(test), train.ClaimNb.sum() / train.Exposure.sum())
    f_pred = f_model.predict(test)
    print(f"Déviance Poisson (test) : modèle nul {poisson_deviance(test, f_null):.4f} "
          f"-> GLM {poisson_deviance(test, f_pred):.4f}")
    pp = pure_premium(f_model, s_model, test)
    g = gini(test.ClaimAmount.values, pp, test.Exposure.values)
    print(f"Gini de la prime pure (test) : {g:.3f}")
    ratio = (pp * test.Exposure).sum() / test.ClaimAmount.sum()
    print(f"Charge prédite / charge observée (test) : {ratio:.3f}\n")

    # 4. Sorties
    rel = relativities(f_model)
    rel.to_csv(OUT / "relativites_frequence.csv", index=False)
    lift = lift_table(test, pp)
    lift.to_csv(OUT / "lift_deciles.csv")
    print("Prime pure par décile de risque (test) :")
    print(lift.round(1).to_string(), "\n")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    x, y = lorenz_curve(test.ClaimAmount.values, pp, test.Exposure.values)
    ax[0].plot(x, y, label=f"GLM (Gini = {g:.3f})")
    ax[0].plot([0, 1], [0, 1], "k--", label="Tarif uniforme")
    ax[0].set(title="Courbe de Lorenz", xlabel="Part cumulée de l'exposition",
              ylabel="Part cumulée de la charge sinistres")
    ax[0].legend()
    ax[1].plot(lift.index, lift.prime_pure_predite, "o-", label="Prédite")
    ax[1].plot(lift.index, lift.prime_pure_observee, "s-", label="Observée")
    ax[1].set(title="Prime pure par décile de risque", xlabel="Décile", ylabel="€ / an")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(OUT / "performance.png", dpi=150)

    bm = rel[rel.variable.str.startswith("DrivAgeBand")]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(bm.variable.str.replace("DrivAgeBand_", ""), bm.relativite)
    ax.axhline(1, color="k", lw=0.8)
    ax.set(title="Relativités de fréquence par âge du conducteur (réf. 17-21 ans)",
           ylabel="exp(beta)")
    fig.tight_layout()
    fig.savefig(OUT / "relativites_age.png", dpi=150)
    print(f"Résultats enregistrés dans {OUT}/")

    # 5. Mise à jour automatique de la section « Résultats » du README
    top = rel.assign(ecart=(rel.relativite - 1).abs()).nlargest(8, "ecart")
    md = [
        f"*Résultats obtenus sur {'données synthétiques' if demo else 'freMTPL2'} "
        f"(échantillon test de {len(test):,} contrats).*",
        "",
        "| Indicateur | Valeur |",
        "|---|---|",
        f"| Contrats | {len(df):,} |",
        f"| Exposition totale | {df.Exposure.sum():,.0f} années |",
        f"| Fréquence observée | {df.ClaimNb.sum() / df.Exposure.sum():.2%} par an |",
        f"| Coût moyen observé | {df.ClaimAmount.sum() / df.ClaimNb.sum():,.0f} € |",
        f"| Prime pure moyenne observée | {df.ClaimAmount.sum() / df.Exposure.sum():,.1f} € / an |",
        f"| Déviance de Poisson test : modèle nul → GLM | {poisson_deviance(test, f_null):.4f} → {poisson_deviance(test, f_pred):.4f} |",
        f"| Indice de Gini de la prime pure (test) | {g:.3f} |",
        f"| Charge prédite / charge observée (test) | {ratio:.3f} |",
        "",
        "**Prime pure par décile de risque (test)**",
        "",
        "| Décile | Exposition | Prime pure prédite (€) | Prime pure observée (€) |",
        "|---|---|---|---|",
        *[f"| {i} | {r.exposition:,.0f} | {r.prime_pure_predite:,.1f} | {r.prime_pure_observee:,.1f} |"
          for i, r in lift.iterrows()],
        "",
        "**Principales relativités du modèle de fréquence**",
        "",
        "| Variable | Relativité exp(β) |",
        "|---|---|",
        *[f"| {r.variable} | {r.relativite:.3f} |" for r in top.itertuples()],
        "",
        "![Courbe de Lorenz et lift](outputs/performance.png)",
        "",
        "![Relativités par âge du conducteur](outputs/relativites_age.png)",
    ]
    (OUT / "resultats.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    readme = Path("README.md")
    if readme.exists():
        txt = readme.read_text(encoding="utf-8")
        new = re.sub(r"(<!-- RESULTATS:DEBUT -->).*?(<!-- RESULTATS:FIN -->)",
                     lambda m: m.group(1) + "\n" + "\n".join(md) + "\n" + m.group(2), txt, flags=re.S)
        readme.write_text(new, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", action="store_true", help="données synthétiques (hors ligne)")
    main(parser.parse_args().demo)
