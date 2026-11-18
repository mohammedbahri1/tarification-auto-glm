*Résultats obtenus sur freMTPL2 (échantillon test de 133,780 contrats).*

| Indicateur | Valeur |
|---|---|
| Contrats | 668,897 |
| Exposition totale | 353,755 années |
| Fréquence observée | 7.46% par an |
| Coût moyen observé | 1,977 € |
| Prime pure moyenne observée | 147.6 € / an |
| Déviance de Poisson test : modèle nul → GLM | 0.4850 → 0.4636 |
| Indice de Gini de la prime pure (test) | 0.248 |
| Charge prédite / charge observée (test) | 0.960 |

**Prime pure par décile de risque (test)**

| Décile | Exposition | Prime pure prédite (€) | Prime pure observée (€) |
|---|---|---|---|
| 1 | 7,442 | 61.7 | 119.9 |
| 2 | 7,620 | 78.7 | 86.3 |
| 3 | 7,674 | 89.9 | 91.3 |
| 4 | 7,638 | 100.2 | 89.7 |
| 5 | 7,534 | 111.2 | 123.4 |
| 6 | 7,317 | 124.2 | 112.1 |
| 7 | 7,173 | 142.1 | 183.0 |
| 8 | 6,856 | 169.0 | 148.7 |
| 9 | 6,167 | 222.4 | 224.4 |
| 10 | 5,489 | 456.4 | 426.3 |

**Principales relativités du modèle de fréquence**

| Variable | Relativité exp(β) |
|---|---|
| BonusMalus | 1.542 |
| DrivAgeBand_(25, 30] | 0.730 |
| VehAgeBand_(15, 200] | 0.741 |
| DrivAgeBand_(21, 25] | 0.775 |
| VehBrand_B12 | 0.798 |
| Region_R82 | 1.199 |
| VehBrand_B11 | 1.181 |
| Region_R73 | 0.825 |

![Courbe de Lorenz et lift](outputs/performance.png)

![Relativités par âge du conducteur](outputs/relativites_age.png)
