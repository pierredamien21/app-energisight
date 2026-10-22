# EnergiSight : simulateur de consommation énergétique et de CO₂

Application Streamlit qui estime la consommation d'énergie et les émissions de CO₂ d'un bâtiment non résidentiel à partir de quelques caractéristiques. Elle utilise les modèles entraînés dans le projet d'étude [energisight-lome](https://github.com/pierredamien21/energisight-lome).

## Fonctionnement

1. L'utilisateur choisit ce qu'il veut estimer : l'énergie (kBtu) ou les émissions de CO₂.
2. Il renseigne la surface, le type de bâtiment, l'année de construction, le score ENERGY STAR et le réseau électrique (Lomé, réseau thermique de la CEET, ou Seattle, réseau hydraulique).
3. L'application charge le modèle correspondant (XGBoost pour l'énergie, LightGBM pour le CO₂) et affiche le résultat, comparé à une valeur de référence pour ce type de bâtiment.

Le facteur d'émission du réseau change fortement le résultat : 0,62 kg CO₂/kWh à Lomé contre 0,05 à Seattle, soit environ 12 fois plus.

## Installation

```bash
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

## Contenu

```
app.py                 application Streamlit
models/                modèles entraînés (.pkl), hyperparamètres et comparaison des modèles CO₂
requirements.txt
```

## Limites

- Si un modèle ne se charge pas, l'application passe sur un calcul simplifié et l'indique à l'écran.
- Les barres d'importance des variables sont des valeurs fixes affichées à titre indicatif : elles ne sont pas recalculées pour chaque prédiction.
- Le modèle adapté à Lomé a été calibré sur un jeu de données synthétique (voir `models/mvp_lome_metadata.json`) : il reste à valider sur des relevés réels.

## Stack

Python · Streamlit · XGBoost · LightGBM · scikit-learn · pandas
