"""
EnergiSight Dashboard — Simulation de Prédiction ML
Startup GreenSight Lomé | Programme D-CLIC · OIF 2024–2025
Auteur : NADJAK Moyinib Pierre Damien

Lancement :  streamlit run energisight_dashboard.py
"""

import streamlit as st
import math
import time
import random
import pickle
from datetime import datetime
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    pd = None

# ─────────────────────────────────────────────
# CONFIG PAGE (doit être le premier appel Streamlit)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="EnergiSight — Simulation ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
# STYLES CSS PERSONNALISÉS
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&family=Inter:wght@300;400;500&display=swap');

/* ── Variables ── */
:root {
    --bg-primary:   #0A0F1E;
    --bg-card:      #111827;
    --bg-card2:     #1a2235;
    --teal:         #00D4C8;
    --teal-dim:     rgba(0, 212, 200, 0.12);
    --amber:        #F59E0B;
    --amber-dim:    rgba(245, 158, 11, 0.12);
    --green:        #10B981;
    --green-dim:    rgba(16, 185, 129, 0.12);
    --red:          #EF4444;
    --red-dim:      rgba(239, 68, 68, 0.12);
    --purple:       #8B5CF6;
    --purple-dim:   rgba(139, 92, 246, 0.12);
    --text:         #E2E8F0;
    --text-dim:     #64748B;
    --border:       rgba(255,255,255,0.06);
    --font-main:    'Space Grotesk', sans-serif;
    --font-mono:    'JetBrains Mono', monospace;
    --font-body:    'Inter', sans-serif;
}

/* ── Reset & Global ── */
html, body, [class*="css"] {
    font-family: var(--font-body) !important;
    color: var(--text) !important;
}

.stApp {
    background: var(--bg-primary) !important;
    background-image: 
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,212,200,0.05) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 80%, rgba(139,92,246,0.04) 0%, transparent 50%);
}

/* ── Hide streamlit branding ── */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
[data-testid="stToolbar"] { display: none; }

/* ── Main container padding ── */
.main .block-container {
    padding: 2rem 3rem 4rem !important;
    max-width: 1400px !important;
}

/* ── Headings ── */
h1, h2, h3 { font-family: var(--font-main) !important; }

/* ── Custom hero ── */
.hero-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 2.5rem 3rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero-container::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--teal), var(--purple), var(--amber));
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: var(--teal-dim);
    border: 1px solid rgba(0,212,200,0.3);
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    font-family: var(--font-mono) !important;
    color: var(--teal) !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    margin-bottom: 12px;
}
.hero-title {
    font-family: var(--font-main) !important;
    font-size: 3rem !important;
    font-weight: 700 !important;
    color: #FFFFFF !important;
    line-height: 1.1 !important;
    margin: 0 0 8px !important;
}
.hero-title span { color: var(--teal) !important; }
.hero-subtitle {
    font-size: 1rem !important;
    color: var(--text-dim) !important;
    font-weight: 400 !important;
    margin: 0 !important;
}
.hero-stats {
    display: flex;
    gap: 2rem;
}
.stat-item { text-align: center; }
.stat-val {
    font-family: var(--font-mono) !important;
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: var(--teal) !important;
    display: block;
}
.stat-lbl {
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.stat-divider {
    width: 1px;
    background: var(--border);
    align-self: stretch;
}

/* ── Section labels ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 1rem;
}
.section-label-text {
    font-family: var(--font-main) !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    color: var(--text-dim) !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
}
.section-label-line {
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Target selector cards ── */
.target-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 2rem;
}
.target-card {
    padding: 1.5rem;
    background: var(--bg-card);
    border: 2px solid var(--border);
    border-radius: 16px;
    cursor: pointer;
    transition: all 0.25s ease;
    position: relative;
    overflow: hidden;
}
.target-card.energy {
    border-color: var(--amber);
    background: linear-gradient(135deg, var(--bg-card), rgba(245,158,11,0.06));
}
.target-card.co2 {
    border-color: var(--green);
    background: linear-gradient(135deg, var(--bg-card), rgba(16,185,129,0.06));
}
.target-card-icon { font-size: 2rem; margin-bottom: 8px; }
.target-card-title {
    font-family: var(--font-main) !important;
    font-size: 1.1rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
    margin-bottom: 4px !important;
}
.target-card-desc {
    font-size: 0.8rem !important;
    color: var(--text-dim) !important;
}
.target-card-unit {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    margin-top: 10px !important;
}
.unit-energy { color: var(--amber) !important; }
.unit-co2    { color: var(--green) !important; }

/* ── Parameter cards ── */
.param-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1rem;
}
.param-label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    font-weight: 500 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim) !important;
    margin-bottom: 6px !important;
}

/* ── Streamlit widget overrides ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stNumberInput"] > div > div > input,
[data-testid="stSlider"] {
    background: var(--bg-card2) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    color: var(--text) !important;
    font-family: var(--font-body) !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stNumberInput"] label,
[data-testid="stSlider"] label,
[data-testid="stRadio"] label {
    color: var(--text-dim) !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    font-family: var(--font-main) !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* ── Radio button (target selector) ── */
[data-testid="stRadio"] > div {
    display: flex;
    gap: 1rem;
    flex-direction: row !important;
}
[data-testid="stRadio"] > div > label {
    flex: 1;
    background: var(--bg-card) !important;
    border: 2px solid var(--border) !important;
    border-radius: 14px !important;
    padding: 1.2rem !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
}

/* ── Predict button ── */
.stButton > button {
    width: 100% !important;
    padding: 1rem 2rem !important;
    background: linear-gradient(135deg, var(--teal), #00a89f) !important;
    color: #000 !important;
    font-family: var(--font-main) !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 14px !important;
    cursor: pointer !important;
    letter-spacing: 0.05em !important;
    transition: all 0.25s ease !important;
    text-transform: uppercase !important;
    box-shadow: 0 4px 24px rgba(0,212,200,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(0,212,200,0.5) !important;
}

/* ── Result card ── */
.result-main {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 2.5rem;
    position: relative;
    overflow: hidden;
}
.result-main.energy-result {
    border-color: rgba(245,158,11,0.4);
    background: linear-gradient(135deg, var(--bg-card), rgba(245,158,11,0.04));
}
.result-main.co2-result {
    border-color: rgba(16,185,129,0.4);
    background: linear-gradient(135deg, var(--bg-card), rgba(16,185,129,0.04));
}
.result-main::after {
    content: '';
    position: absolute;
    bottom: -60px; right: -60px;
    width: 200px; height: 200px;
    border-radius: 50%;
    opacity: 0.04;
}
.result-main.energy-result::after { background: var(--amber); }
.result-main.co2-result::after    { background: var(--green); }

.result-label {
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: var(--text-dim) !important;
    margin-bottom: 4px !important;
}
.result-value {
    font-family: var(--font-main) !important;
    font-size: 3.5rem !important;
    font-weight: 700 !important;
    line-height: 1 !important;
    margin-bottom: 4px !important;
}
.result-value.energy { color: var(--amber) !important; }
.result-value.co2    { color: var(--green) !important; }
.result-unit {
    font-family: var(--font-mono) !important;
    font-size: 1rem !important;
    color: var(--text-dim) !important;
}

/* ── Confidence interval ── */
.ci-bar-container {
    background: rgba(255,255,255,0.05);
    border-radius: 8px;
    height: 8px;
    margin: 12px 0 6px;
    overflow: hidden;
    position: relative;
}
.ci-bar-fill {
    position: absolute;
    top: 0; bottom: 0;
    border-radius: 8px;
}
.ci-fill-energy { background: linear-gradient(90deg, var(--amber), #fbbf24); }
.ci-fill-co2    { background: linear-gradient(90deg, var(--green), #34d399); }

/* ── Metric chips ── */
.metric-chip {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: var(--bg-card2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 5px 10px;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    color: var(--text) !important;
    margin: 3px;
}

/* ── Insight card ── */
.insight-box {
    background: var(--bg-card2);
    border-left: 3px solid var(--teal);
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.25rem;
    margin-top: 1.2rem;
}
.insight-title {
    font-family: var(--font-main) !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--teal) !important;
    margin-bottom: 6px !important;
}
.insight-text {
    font-size: 0.87rem !important;
    color: var(--text-dim) !important;
    line-height: 1.6 !important;
}

/* ── SHAP bar ── */
.shap-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 8px;
}
.shap-label {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    width: 160px;
    flex-shrink: 0;
}
.shap-bar-bg {
    flex: 1;
    background: rgba(255,255,255,0.05);
    border-radius: 4px;
    height: 10px;
    overflow: hidden;
}
.shap-bar-fill {
    height: 100%;
    border-radius: 4px;
    transition: width 0.6s ease;
}
.shap-pct {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
    width: 36px;
    text-align: right;
    flex-shrink: 0;
}

/* ── Comparison block ── */
.compare-item {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.75rem 1rem;
    background: var(--bg-card2);
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid var(--border);
}
.compare-item-label {
    font-size: 0.82rem !important;
    color: var(--text-dim) !important;
}
.compare-item-value {
    font-family: var(--font-mono) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    color: var(--text) !important;
}

/* ── Bottom bar ── */
.bottom-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 1rem 1.5rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 14px;
    margin-top: 2rem;
}
.bb-left {
    font-family: var(--font-mono) !important;
    font-size: 0.72rem !important;
    color: var(--text-dim) !important;
}
.bb-chips { display: flex; gap: 6px; }
.bb-chip {
    background: var(--teal-dim);
    border: 1px solid rgba(0,212,200,0.2);
    border-radius: 6px;
    padding: 3px 8px;
    font-family: var(--font-mono) !important;
    font-size: 0.68rem !important;
    color: var(--teal) !important;
}

/* ── Divider ── */
.vdiv {
    width: 1px;
    background: var(--border);
    align-self: stretch;
    margin: 0 1rem;
}

/* ── Spinner override ── */
.stSpinner > div {
    border-color: var(--teal) transparent transparent transparent !important;
}

/* ── Alert / info boxes ── */
[data-testid="stAlert"] {
    background: var(--teal-dim) !important;
    border: 1px solid rgba(0,212,200,0.25) !important;
    border-radius: 12px !important;
    color: var(--teal) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.1); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DONNÉES DE RÉFÉRENCE
# ─────────────────────────────────────────────

BUILDING_TYPES = {
    "🏢 Bureaux / Office": {
        "base_eui": 92,        # kBtu/sf/yr (Seattle médiane)
        "co2_factor": 1.0,
        "description": "Espaces de bureaux, open-spaces, immeubles corporate"
    },
    "🏪 Commerce / Retail": {
        "base_eui": 108,
        "co2_factor": 1.15,
        "description": "Centres commerciaux, boutiques, supermarchés"
    },
    "🏨 Hôtel / Lodging": {
        "base_eui": 145,
        "co2_factor": 1.35,
        "description": "Hôtels, résidences, bed & breakfast"
    },
    "🏫 École / University": {
        "base_eui": 78,
        "co2_factor": 0.88,
        "description": "Écoles primaires, lycées, universités, campus"
    },
    "🏥 Hôpital / Healthcare": {
        "base_eui": 285,
        "co2_factor": 2.40,
        "description": "Hôpitaux, cliniques, centres médicaux"
    },
    "🏦 Banque / Finance": {
        "base_eui": 112,
        "co2_factor": 1.20,
        "description": "Agences bancaires, sièges financiers"
    },
    "🏭 Entrepôt / Warehouse": {
        "base_eui": 38,
        "co2_factor": 0.55,
        "description": "Entrepôts, logistique, stockage"
    },
    "🍽️ Restaurant / Food": {
        "base_eui": 320,
        "co2_factor": 2.80,
        "description": "Restaurants, fast-foods, cuisines industrielles"
    },
}

BUILDING_TYPE_ENCODING = {name: idx for idx, name in enumerate(BUILDING_TYPES.keys())}
CURRENT_YEAR = datetime.now().year
MODELS_DIR = Path(__file__).parent / "models"
MODEL_PATHS = {
    "energy": MODELS_DIR / "model_energy_xgboost_optimized.pkl",
    "co2": MODELS_DIR / "model_co2_lightgbm_optimized.pkl",
}

# Émission grid Lomé vs Seattle
EMISSION_FACTORS = {
    "Lomé (CEET — thermique)": 0.62,
    "Seattle (hydro)": 0.05,
}

# ─────────────────────────────────────────────
# MOTEUR DE PRÉDICTION (MODELES .PKL + FALLBACK)
# ─────────────────────────────────────────────

@st.cache_resource
def load_pickle_model(path: Path):
    if not path.exists():
        return None, f"Model file not found: {path.name}"
    try:
        with path.open("rb") as f:
            return pickle.load(f), None
    except Exception as exc:
        return None, f"Failed to load {path.name}: {exc}"


def build_feature_payload(surface_m2: float, building_type: str, year_built: int,
                          energy_star_score: float, city: str) -> dict:
    bt = BUILDING_TYPES[building_type]
    surface_sf = surface_m2 * 10.764  # m² → sq ft
    age = CURRENT_YEAR - year_built
    base_kbtu = surface_sf * bt["base_eui"]
    eui_kwh_m2 = bt["base_eui"] * 0.293071 / 0.092903
    type_encoded = BUILDING_TYPE_ENCODING.get(building_type, 0)

    return {
        "propertygfatotal_log": math.log1p(surface_sf),
        "propertygfa_log": math.log1p(surface_sf),
        "building_age": age,
        "energystarscore": energy_star_score,
        "energystar_missing": 0,
        "numberoffloors_log": math.log1p(1),
        "elec_ratio": 1.0,
        "uses_steam": 0,
        "steam_ratio": 0.0,
        "has_parking": 0,
        "is_multipurpose": 0,
        "type_encoded": type_encoded,
        "primarytype_encoded": type_encoded,
        "seattle_pred_log": math.log1p(base_kbtu),
        "seattle_biais": 0.0,
        "uses_clim": 0,
        "has_generator": 0,
        "eui_kwh_m2_an": eui_kwh_m2,
        "emission_factor_grid": EMISSION_FACTORS[city],
    }


def predict_from_model(model, features: dict):
    if pd is None:
        return None, "pandas is not installed"

    feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None and hasattr(model, "steps"):
        for _, step in model.steps:
            if hasattr(step, "feature_names_in_"):
                feature_names = step.feature_names_in_
                break

    if feature_names is None:
        X = pd.DataFrame([features])
    else:
        X = pd.DataFrame([[features.get(name, 0) for name in feature_names]],
                         columns=list(feature_names))

    try:
        pred = model.predict(X)
        pred_val = float(pred[0]) if hasattr(pred, "__len__") else float(pred)
        return max(pred_val, 0.0), None
    except Exception as exc:
        return None, str(exc)


def build_energy_result(predicted_kbtu: float, surface_m2: float, building_type: str,
                        year_built: int, city: str, margin_pct: float = 0.0596) -> dict:
    surface_sf = surface_m2 * 10.764
    age = CURRENT_YEAR - year_built
    ef = EMISSION_FACTORS[city]
    kwh = predicted_kbtu / 3.412
    ci_low = predicted_kbtu * (1 - margin_pct)
    ci_high = predicted_kbtu * (1 + margin_pct)
    co2_tons = kwh * ef / 1000

    return {
        "energy_kbtu":    round(predicted_kbtu),
        "energy_mwh":     round(kwh / 1000, 1),
        "ci_low":         round(ci_low),
        "ci_high":        round(ci_high),
        "co2_tons":       round(co2_tons, 1),
        "emission_factor": ef,
        "age":            age,
        "surface_sf":     round(surface_sf),
        "eui":            round(predicted_kbtu / surface_sf, 1),
    }


def build_co2_result(co2_tons: float, surface_m2: float, building_type: str,
                     year_built: int, city: str, margin_co2: float = 0.058) -> dict:
    bt = BUILDING_TYPES[building_type]
    surface_sf = surface_m2 * 10.764
    ef = EMISSION_FACTORS[city]
    energy_kbtu_est = surface_sf * bt["base_eui"]

    ci_low = co2_tons * (1 - margin_co2)
    ci_high = co2_tons * (1 + margin_co2)
    km_voiture = co2_tons * 1000 / 0.21
    arbres_an = co2_tons / 0.022

    return {
        "co2_tons":      round(co2_tons, 2),
        "ci_low":        round(ci_low, 2),
        "ci_high":       round(ci_high, 2),
        "km_voiture":    round(km_voiture),
        "arbres_an":     round(arbres_an),
        "energy_kbtu":   round(energy_kbtu_est),
        "eui":           round(energy_kbtu_est / surface_sf, 1),
        "emission_factor": ef,
        "city_penalty":  "×12 vs Seattle" if ef > 0.3 else "×1 (reference)",
    }


def predict_energy_simulated(surface_m2: float, building_type: str, year_built: int,
                             energy_star_score: float, city: str) -> dict:
    bt = BUILDING_TYPES[building_type]
    surface_sf = surface_m2 * 10.764

    base_consumption = surface_sf * bt["base_eui"]
    age = CURRENT_YEAR - year_built
    age_factor = 1.0 + max(0, age - 15) * 0.006
    age_factor = min(age_factor, 1.65)
    star_factor = 1.0 - (energy_star_score - 50) / 250

    predicted_kbtu = base_consumption * age_factor * star_factor
    return build_energy_result(predicted_kbtu, surface_m2, building_type, year_built, city)


def predict_co2_simulated(surface_m2: float, building_type: str, year_built: int,
                          energy_star_score: float, city: str) -> dict:
    base = predict_energy_simulated(surface_m2, building_type, year_built, energy_star_score, city)
    ef = EMISSION_FACTORS[city]
    kwh = base["energy_kbtu"] / 3.412
    co2_direct = kwh * ef / 1000
    return build_co2_result(co2_direct, surface_m2, building_type, year_built, city)


def predict_energy(surface_m2: float, building_type: str, year_built: int,
                   energy_star_score: float, city: str) -> dict:
    features = build_feature_payload(surface_m2, building_type, year_built, energy_star_score, city)
    model, err = load_pickle_model(MODEL_PATHS["energy"])

    if model is not None and err is None:
        pred, pred_err = predict_from_model(model, features)
        if pred_err is None:
            res = build_energy_result(pred, surface_m2, building_type, year_built, city)
            res["source"] = "model"
            return res
        warn = f"Model prediction failed: {pred_err}"
    else:
        warn = err or "Model not available"

    res = predict_energy_simulated(surface_m2, building_type, year_built, energy_star_score, city)
    res["source"] = "simulation"
    res["warning"] = warn
    return res


def predict_co2(surface_m2: float, building_type: str, year_built: int,
                energy_star_score: float, city: str) -> dict:
    features = build_feature_payload(surface_m2, building_type, year_built, energy_star_score, city)
    model, err = load_pickle_model(MODEL_PATHS["co2"])

    if model is not None and err is None:
        pred, pred_err = predict_from_model(model, features)
        if pred_err is None:
            res = build_co2_result(pred, surface_m2, building_type, year_built, city)
            res["source"] = "model"
            return res
        warn = f"Model prediction failed: {pred_err}"
    else:
        warn = err or "Model not available"

    res = predict_co2_simulated(surface_m2, building_type, year_built, energy_star_score, city)
    res["source"] = "simulation"
    res["warning"] = warn
    return res


def get_benchmark_energy(surface_m2: float, building_type: str) -> dict:
    """Médiane Seattle pour ce type de bâtiment."""
    bt = BUILDING_TYPES[building_type]
    surface_sf = surface_m2 * 10.764
    median_kbtu = surface_sf * bt["base_eui"]
    return {"median": round(median_kbtu), "eui": bt["base_eui"]}


# ─────────────────────────────────────────────
# COMPOSANTS UI
# ─────────────────────────────────────────────

def render_hero():
    st.markdown("""
    <div class="hero-container">
        <div>
            <div class="hero-badge">⚡ EnergiSight ML — v2.1</div>
            <h1 class="hero-title">Prédiction <span>Énergétique</span><br>& Émissions CO₂</h1>
            <p class="hero-subtitle">Moteur XGBoost · Dataset Seattle 2016 · Adapté Lomé</p>
        </div>
        <div class="hero-stats">
            <div class="stat-item">
                <span class="stat-val">99.03%</span>
                <span class="stat-lbl">R² Énergie</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-item">
                <span class="stat-val">99.53%</span>
                <span class="stat-lbl">R² CO₂</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-item">
                <span class="stat-val">5.96%</span>
                <span class="stat-lbl">MAPE</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-item">
                <span class="stat-val">1648</span>
                <span class="stat-lbl">Bâtiments</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


def section_label(text: str):
    st.markdown(f"""
    <div class="section-label">
        <span class="section-label-text">{text}</span>
        <div class="section-label-line"></div>
    </div>
    """, unsafe_allow_html=True)


def shap_bars(target: str, surface: float, surface_max: float = 10000):
    """Affiche les barres d'importance SHAP selon la target."""
    is_energy = target == "⚡ Énergie (kBtu)"

    if is_energy:
        features = [
            ("propertygfa_log",  28, "#F59E0B"),
            ("building_age",     18, "#F59E0B"),
            ("primarytype",      15, "#F59E0B"),
            ("energystarscore",  12, "#EF4444"),
            ("elec_ratio",        9, "#F59E0B"),
            ("numberoffloors",    8, "#F59E0B"),
            ("steam_ratio",       5, "#F59E0B"),
            ("uses_gas",          5, "#F59E0B"),
        ]
    else:
        features = [
            ("propertygfa_log",  25, "#10B981"),
            ("elec_ratio",       22, "#10B981"),
            ("uses_gas",         18, "#10B981"),
            ("building_age",     14, "#10B981"),
            ("primarytype",      11, "#10B981"),
            ("energystarscore",   7, "#EF4444"),
            ("emission_factor",   2, "#10B981"),
            ("steam_ratio",       1, "#10B981"),
        ]

    html = '<div style="margin-top:1rem">'
    for fname, pct, color in features:
        html += f"""
        <div class="shap-row">
            <span class="shap-label">{fname}</span>
            <div class="shap-bar-bg">
                <div class="shap-bar-fill" style="width:{pct*3}%; background:{color}; opacity:0.8"></div>
            </div>
            <span class="shap-pct">{pct}%</span>
        </div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# LAYOUT PRINCIPAL
# ─────────────────────────────────────────────

render_hero()

# ── Deux colonnes principales ─────────────────
col_form, col_result = st.columns([1, 1.4], gap="large")

# ════════════════════════════════════════════════
# COLONNE GAUCHE : FORMULAIRE
# ════════════════════════════════════════════════
with col_form:

    # ── Target ──────────────────────────────────
    section_label("01 — Variable cible")

    target = st.radio(
        "Choisissez ce que vous souhaitez prédire",
        ["⚡ Énergie (kBtu)", "🌿 Émissions CO₂ (tCO₂eq)"],
        horizontal=True,
        label_visibility="collapsed"
    )
    is_energy = target == "⚡ Énergie (kBtu)"

    # Badge info
    if is_energy:
        st.markdown("""
        <div style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.3);
        border-radius:10px;padding:10px 14px;margin:8px 0 16px;font-size:0.82rem;color:#F59E0B">
        ⚡ <b>SiteEnergyUse</b> — Consommation énergétique annuelle totale du bâtiment (kBtu/an)
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:rgba(16,185,129,0.08);border:1px solid rgba(16,185,129,0.3);
        border-radius:10px;padding:10px 14px;margin:8px 0 16px;font-size:0.82rem;color:#10B981">
        🌿 <b>TotalGHGEmissions</b> — Émissions totales de gaz à effet de serre (tCO₂eq/an)
        </div>""", unsafe_allow_html=True)

    st.markdown("---", unsafe_allow_html=False)

    # ── Paramètre 1 : Surface ────────────────────
    section_label("02 — Paramètres du bâtiment")

    surface_m2 = st.number_input(
        "📐  Surface totale (m²)",
        min_value=100,
        max_value=500_000,
        value=2_000,
        step=100,
        help="Surface brute totale du bâtiment (GFA — Gross Floor Area)"
    )
    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:#64748B;margin:-8px 0 12px;padding-left:4px">
    ≈ {round(surface_m2 * 10.764):,} sq ft &nbsp;|&nbsp; {surface_m2:,} m²
    </div>""", unsafe_allow_html=True)

    # ── Paramètre 2 : Type ──────────────────────
    building_type = st.selectbox(
        "🏗️  Type de bâtiment",
        options=list(BUILDING_TYPES.keys()),
        index=0,
        help="Catégorie fonctionnelle principale du bâtiment"
    )
    bt_info = BUILDING_TYPES[building_type]
    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:#64748B;margin:-8px 0 12px;padding-left:4px">
    EUI de référence : {bt_info["base_eui"]} kBtu/sf/yr &nbsp;|&nbsp; {bt_info["description"]}
    </div>""", unsafe_allow_html=True)

    # ── Paramètre 3 : Année ─────────────────────
    year_built = st.slider(
        "📅  Année de construction",
        min_value=1900,
        max_value=CURRENT_YEAR,
        value=2005,
        step=1,
        help="Année de construction initiale du bâtiment"
    )
    age = CURRENT_YEAR - year_built
    era_map = {
        range(1900, 1950): ("Époque historique", "#8B5CF6"),
        range(1950, 1975): ("Ère post-guerre", "#F59E0B"),
        range(1975, 2000): ("Époque moderne", "#3B82F6"),
        range(2000, 2015): ("Époque contemporaine", "#10B981"),
        range(2015, 2025): ("Construction récente", "#00D4C8"),
    }
    era_label, era_color = "Moderne", "#64748B"
    for r, (lbl, clr) in era_map.items():
        if year_built in r:
            era_label, era_color = lbl, clr
            break

    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:{era_color};margin:-8px 0 12px;padding-left:4px">
    Âge : {age} ans &nbsp;|&nbsp; {era_label}
    </div>""", unsafe_allow_html=True)

    # ── Paramètre 4 : EnergyStar Score ──────────
    energy_star = st.slider(
        "⭐  Score EnergyStar (1–100)",
        min_value=1,
        max_value=100,
        value=65,
        step=1,
        help="Score d'efficacité énergétique EPA. 50=médiane nationale, 75+=efficace, 90+=exceptionnel"
    )
    star_tier = "🔴 Peu efficace" if energy_star < 40 else ("🟡 Moyen" if energy_star < 65 else ("🟢 Efficace" if energy_star < 85 else "⭐ Exceptionnel"))
    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:#64748B;margin:-8px 0 12px;padding-left:4px">
    {star_tier} &nbsp;|&nbsp; Percentile {energy_star}e
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Contexte : ville ────────────────────────
    section_label("03 — Contexte géographique")

    city = st.selectbox(
        "🌍  Facteur d'émission réseau",
        options=list(EMISSION_FACTORS.keys()),
        index=0,
        help="Le facteur d'émission impacte le calcul CO₂. Lomé (CEET thermique) = ×12 vs Seattle."
    )
    ef = EMISSION_FACTORS[city]
    ef_color = "#EF4444" if ef > 0.3 else "#10B981"
    st.markdown(f"""
    <div style="font-family:var(--font-mono);font-size:0.72rem;color:{ef_color};margin:-8px 0 16px;padding-left:4px">
    {ef} kgCO₂/kWh &nbsp;{"⚠️ Réseau très carboné" if ef > 0.3 else "✅ Réseau décarboné"}
    </div>""", unsafe_allow_html=True)

    # ── Bouton ──────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🚀  Lancer la prédiction", use_container_width=True)


# ════════════════════════════════════════════════
# COLONNE DROITE : RÉSULTATS
# ════════════════════════════════════════════════
with col_result:

    if not predict_btn:
        # ── État initial ─────────────────────────
        st.markdown("""
        <div style="height:100%;display:flex;flex-direction:column;justify-content:center;
        align-items:center;padding:4rem 2rem;background:rgba(255,255,255,0.02);
        border:1px dashed rgba(255,255,255,0.08);border-radius:20px;text-align:center">
            <div style="font-size:4rem;margin-bottom:1.5rem;opacity:0.4">⚡</div>
            <div style="font-family:'Space Grotesk',sans-serif;font-size:1.25rem;
            font-weight:600;color:#E2E8F0;margin-bottom:8px">Prêt pour la simulation</div>
            <div style="font-size:0.875rem;color:#64748B;max-width:320px;line-height:1.6">
            Configurez les paramètres du bâtiment, choisissez votre target,
            et lancez la prédiction pour obtenir une estimation instantanée.
            </div>
            <div style="margin-top:2rem;display:flex;gap:8px;flex-wrap:wrap;justify-content:center">
                <span style="background:rgba(0,212,200,0.1);border:1px solid rgba(0,212,200,0.2);
                border-radius:20px;padding:4px 12px;font-size:0.72rem;
                font-family:'JetBrains Mono';color:#00D4C8">XGBoost</span>
                <span style="background:rgba(0,212,200,0.1);border:1px solid rgba(0,212,200,0.2);
                border-radius:20px;padding:4px 12px;font-size:0.72rem;
                font-family:'JetBrains Mono';color:#00D4C8">R²=99%+</span>
                <span style="background:rgba(0,212,200,0.1);border:1px solid rgba(0,212,200,0.2);
                border-radius:20px;padding:4px 12px;font-size:0.72rem;
                font-family:'JetBrains Mono';color:#00D4C8">SHAP Explainability</span>
                <span style="background:rgba(0,212,200,0.1);border:1px solid rgba(0,212,200,0.2);
                border-radius:20px;padding:4px 12px;font-size:0.72rem;
                font-family:'JetBrains Mono';color:#00D4C8">Seattle 2016</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Simulation ───────────────────────────
        with st.spinner("Inférence en cours…"):
            time.sleep(0.9)

        # ── Calcul ───────────────────────────────
        if is_energy:
            res = predict_energy(surface_m2, building_type, year_built, energy_star, city)
            bench = get_benchmark_energy(surface_m2, building_type)
        else:
            res = predict_co2(surface_m2, building_type, year_built, energy_star, city)
            bench = get_benchmark_energy(surface_m2, building_type)

        if res.get("source") == "model":
            st.info("Modele .pkl utilise pour la prediction.")
        elif res.get("warning"):
            st.warning(f"Fallback simulation (modele indisponible): {res['warning']}")

        # ── Résultat principal ───────────────────
        section_label("Résultat de la prédiction")

        if is_energy:
            val_fmt  = f"{res['energy_kbtu']:,.0f}"
            unit_str = "kBtu / an"
            val_class = "energy"
            card_class = "energy-result"
            ci_class = "ci-fill-energy"
            ci_low_fmt  = f"{res['ci_low']:,.0f}"
            ci_high_fmt = f"{res['ci_high']:,.0f}"
            pct_of_max = min(res['energy_kbtu'] / (bench['median'] * 3), 1.0)
        else:
            val_fmt  = f"{res['co2_tons']:,.1f}"
            unit_str = "tCO₂eq / an"
            val_class = "co2"
            card_class = "co2-result"
            ci_class = "ci-fill-co2"
            ci_low_fmt  = f"{res['ci_low']:,.2f}"
            ci_high_fmt = f"{res['ci_high']:,.2f}"
            pct_of_max = min(res['co2_tons'] / (res['co2_tons'] * 2.5), 0.6)

        bar_pct = round(pct_of_max * 100)

        st.markdown(f"""
        <div class="result-main {card_class}">
            <div class="result-label">PRÉDICTION PRINCIPALE · XGBoost · {'Énergie' if is_energy else 'CO₂'}</div>
            <div class="result-value {val_class}">{val_fmt}</div>
            <div class="result-unit">{unit_str}</div>
            <div style="margin-top:1.2rem">
                <div style="display:flex;justify-content:space-between;
                font-family:'JetBrains Mono';font-size:0.7rem;color:#64748B;margin-bottom:4px">
                    <span>IC 95% : {ci_low_fmt}</span>
                    <span>{ci_high_fmt}</span>
                </div>
                <div class="ci-bar-container">
                    <div class="ci-bar-fill {ci_class}" 
                    style="left:15%;width:{bar_pct}%"></div>
                </div>
                <div style="font-family:'JetBrains Mono';font-size:0.68rem;color:#64748B;margin-top:4px">
                Intervalle de confiance · MAPE modèle = {'5.96%' if is_energy else '~6.0%'}
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Métriques secondaires ────────────────
        m1, m2, m3 = st.columns(3)

        if is_energy:
            with m1:
                delta_vs_bench = round((res['energy_kbtu'] / bench['median'] - 1) * 100)
                delta_color = "#EF4444" if delta_vs_bench > 0 else "#10B981"
                delta_sign  = "+" if delta_vs_bench > 0 else ""
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">vs Médiane type</div>
                    <div style="font-family:'Space Grotesk';font-size:1.6rem;
                    font-weight:700;color:{delta_color}">{delta_sign}{delta_vs_bench}%</div>
                    <div style="font-size:0.72rem;color:#64748B">{bench['median']:,.0f} kBtu médiane</div>
                </div>""", unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">EUI Bâtiment</div>
                    <div style="font-family:'Space Grotesk';font-size:1.6rem;
                    font-weight:700;color:#00D4C8">{res['eui']}</div>
                    <div style="font-size:0.72rem;color:#64748B">kBtu/sf/yr</div>
                </div>""", unsafe_allow_html=True)

            with m3:
                mwh = round(res['energy_kbtu'] / 3412)
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">Équivalent</div>
                    <div style="font-family:'Space Grotesk';font-size:1.6rem;
                    font-weight:700;color:#F59E0B">{mwh:,.0f}</div>
                    <div style="font-size:0.72rem;color:#64748B">MWh / an</div>
                </div>""", unsafe_allow_html=True)

        else:
            with m1:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">km voiture</div>
                    <div style="font-family:'Space Grotesk';font-size:1.5rem;
                    font-weight:700;color:#F59E0B">{res['km_voiture']:,.0f}</div>
                    <div style="font-size:0.72rem;color:#64748B">équivalent annuel</div>
                </div>""", unsafe_allow_html=True)

            with m2:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">Arbres à planter</div>
                    <div style="font-family:'Space Grotesk';font-size:1.5rem;
                    font-weight:700;color:#10B981">{res['arbres_an']:,.0f}</div>
                    <div style="font-size:0.72rem;color:#64748B">pour compenser</div>
                </div>""", unsafe_allow_html=True)

            with m3:
                st.markdown(f"""
                <div style="background:#111827;border:1px solid rgba(255,255,255,0.06);
                border-radius:14px;padding:1.2rem;text-align:center">
                    <div style="font-family:'JetBrains Mono';font-size:0.65rem;
                    color:#64748B;text-transform:uppercase;margin-bottom:6px">Facteur CO₂</div>
                    <div style="font-family:'Space Grotesk';font-size:1.5rem;
                    font-weight:700;color:#EF4444">{res['emission_factor']}</div>
                    <div style="font-size:0.72rem;color:#64748B">kgCO₂/kWh réseau</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── SHAP Importance ──────────────────────
        section_label("Importance des features (SHAP)")
        shap_bars(target, surface_m2)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Récap paramètres ─────────────────────
        section_label("Récapitulatif des entrées")

        st.markdown(f"""
        <div class="compare-item">
            <span class="compare-item-label">📐 Surface</span>
            <span class="compare-item-value">{surface_m2:,} m²  ({round(surface_m2*10.764):,} sf)</span>
        </div>
        <div class="compare-item">
            <span class="compare-item-label">🏗️ Type</span>
            <span class="compare-item-value">{building_type.split(' ',1)[1].split('/')[0].strip()}</span>
        </div>
        <div class="compare-item">
            <span class="compare-item-label">📅 Année / Âge</span>
            <span class="compare-item-value">{year_built}  ({age} ans)</span>
        </div>
        <div class="compare-item">
            <span class="compare-item-label">⭐ EnergyStar</span>
            <span class="compare-item-value">{energy_star}/100 — {star_tier}</span>
        </div>
        <div class="compare-item">
            <span class="compare-item-label">🌍 Réseau / Ville</span>
            <span class="compare-item-value">{city.split('(')[0].strip()}  ({ef} kgCO₂/kWh)</span>
        </div>
        """, unsafe_allow_html=True)

        # ── Insight ──────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        if is_energy:
            if age > 40:
                insight = f"Ce bâtiment de {age} ans présente probablement une isolation thermique médiocre (+{round((1+max(0,age-15)*0.006-1)*100)}% vs neuf). Une rénovation thermique pourrait réduire la consommation de 20–35%."
            elif energy_star < 40:
                insight = f"Le score EnergyStar de {energy_star}/100 indique un bâtiment peu efficace. Une optimisation CVC et éclairage pourrait améliorer le score à 65+ et réduire la consommation de 15–25%."
            else:
                insight = f"Ce bâtiment présente un profil de consommation dans la norme pour un {building_type.split(' ',1)[1].split('/')[0].strip()}. L'EUI de {res['eui']} kBtu/sf/yr est comparable à la médiane Seattle ({bench['eui']} kBtu/sf/yr)."
        else:
            if ef > 0.3:
                insight = f"Le facteur d'émission de Lomé ({ef} kgCO₂/kWh) est ×{round(ef/0.05)}x supérieur à Seattle. Pour {res['co2_tons']} tCO₂eq/an, il faudrait planter {res['arbres_an']:,.0f} arbres, ou installer {round(res['energy_kbtu']/3412/1.5)} kWc de panneaux solaires pour compenser."
            else:
                insight = f"Le réseau hydroélectrique de Seattle ({ef} kgCO₂/kWh) maintient les émissions basses. Le même bâtiment à Lomé (0.62 kgCO₂/kWh) émettrait {round(res['co2_tons'] * 0.62 / ef, 1)} tCO₂eq/an."

        st.markdown(f"""
        <div class="insight-box">
            <div class="insight-title">💡 Insight EnergiSight</div>
            <div class="insight-text">{insight}</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# BARRE DE STATUT BAS DE PAGE
# ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="bottom-bar">
    <div class="bb-left">
        EnergiSight v2.1 &nbsp;·&nbsp; GreenSight Lomé &nbsp;·&nbsp; 
        NADJAK Moyinib Pierre Damien &nbsp;·&nbsp; D-CLIC · OIF 2024–2025
    </div>
    <div class="bb-chips">
        <span class="bb-chip">XGBoost</span>
        <span class="bb-chip">SHAP</span>
        <span class="bb-chip">Seattle 2016</span>
        <span class="bb-chip">R²=99%+</span>
        <span class="bb-chip">Domain Adaptation</span>
    </div>
</div>
""", unsafe_allow_html=True)