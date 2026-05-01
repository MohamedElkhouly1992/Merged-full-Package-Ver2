from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import streamlit as st

from hvac_v3_engine import (
    BuildingSpec,
    HVACConfig,
    HVAC_PRESETS,
    SCENARIOS,
    SEVERITY_LEVELS,
    CLIMATE_LEVELS,
    run_scenario_model,
    train_surrogate_models,
)
from report_addons import (
    read_weather_upload,
    build_detailed_tables,
    save_detailed_outputs,
    load_validation_file,
    build_validation_comparison,
    create_zip_from_folder,
    find_result_paths,
    setup_to_json_bytes,
    setup_from_upload,
)

st.set_page_config(page_title="HVAC ROM-Degradation Suite", layout="wide")

CUSTOM_CSS = """
<style>
.stApp {background: linear-gradient(180deg, #07101f 0%, #101729 55%, #151827 100%);} 
.block-container {padding-top: 1.15rem; padding-bottom: 2.2rem; max-width: 1360px;}
h1, h2, h3, h4, h5, h6, p, label, span, div {color: #eaf0fb;}
[data-testid="stHeader"] {background: rgba(0,0,0,0);} 
div[data-baseweb="tab-list"] {gap: 0.45rem; border-bottom: 1px solid rgba(255,255,255,0.10); padding-bottom: 0.25rem;}
button[data-baseweb="tab"] {background: rgba(255,255,255,0.035) !important; border-radius: 14px 14px 0 0 !important; padding: 0.72rem 0.92rem !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.07) !important;}
button[data-baseweb="tab"][aria-selected="true"] {color: #ff686b !important; border-bottom: 2px solid #ff686b !important; background: rgba(255,255,255,0.075) !important;}
div[data-testid="stExpander"] {border: 1px solid rgba(255,255,255,0.10); border-radius: 16px; background: rgba(255,255,255,0.035); margin-bottom: 0.9rem;}
div[data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(255,255,255,0.08); border-radius: 16px; padding: 0.5rem 0.7rem;}
div[data-testid="stDataFrame"] {border: 1px solid rgba(255,255,255,0.08); border-radius: 14px; overflow: hidden;}
div.stButton > button {border-radius: 14px !important; font-weight: 700 !important; border: 1px solid rgba(255,255,255,0.18) !important;}
.small-muted {color:#aeb8ce; font-size:0.94rem;}
.section-card {background: rgba(255,255,255,0.035); border: 1px solid rgba(255,255,255,0.08); border-radius: 18px; padding: 1rem; margin-bottom: 0.8rem;}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
st.markdown(
    """
    <div style="padding: 0.55rem 0 1.0rem 0;">
      <div style="font-size: 2.75rem; font-weight: 850; letter-spacing: -0.035em; color: #f6f8fc;">
        HVAC ROM-Degradation Suite
      </div>
      <div class="small-muted" style="max-width: 1040px; margin-top:0.35rem;">
        Reduced-order HVAC energy, degradation, maintenance-strategy, climate-scenario, validation, reporting, parameter-switch, and CatBoost surrogate modelling platform.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# Defaults and helpers
# -----------------------------
SETUP_DEFAULTS: Dict[str, Any] = {
    "building_type": "Educational / University building",
    "location": "User-defined",
    "area_m2": 5000.0,
    "floors": 4,
    "n_spaces": 40,
    "occupancy_density": 0.08,
    "lighting_w_m2": 10.0,
    "equipment_w_m2": 8.0,
    "sensible_w_per_person": 75.0,
    "airflow_m3h_m2": 4.0,
    "infiltration_ach": 0.5,
    "cooling_w_m2": 100.0,
    "heating_w_m2": 55.0,
    "wall_u": 0.60,
    "roof_u": 0.35,
    "window_u": 2.70,
    "shgc": 0.35,
    "glazing_ratio": 0.30,
    "years": 20,
    "hvac_system_type": "Chiller_AHU",
    "use_hvac_preset": True,
    "cop_cool_nom": 4.5,
    "cop_heat_nom": 3.2,
    "fan_eff": 0.70,
    "dp_clean": 150.0,
    "dp_warn": 320.0,
    "dp_thresh": 420.0,
    "dp_max": 450.0,
    "t_set": 23.0,
    "t_sp_min": 21.0,
    "t_sp_max": 26.0,
    "af_min": 0.55,
    "af_max": 1.00,
    "energy_price": 0.12,
    "co2_factor": 0.536,
    "cost_filter": 50.0,
    "cost_hx": 300.0,
    "filter_interval": 90,
    "hx_interval": 180,
    "w_energy": 0.35,
    "w_degrad": 0.25,
    "w_comfort": 0.25,
    "w_carbon": 0.15,
    "cop_aging_rate": 0.005,
    "rf_star": 2e-4,
    "b_foul": 0.015,
    "dust_rate": 1.2,
    "k_clog": 6.0,
    "deg_trigger": 0.55,
    "degradation_model": "physics",
    "linear_deg_per_day": 0.00012,
    "exp_deg_rate_per_day": 0.00018,
}

PRESET_SETUPS: Dict[str, Dict[str, Any]] = {
    "Default educational building": SETUP_DEFAULTS,
    "Compact university lab block": {**SETUP_DEFAULTS, "building_type": "University laboratory block", "area_m2": 2800.0, "floors": 3, "n_spaces": 28, "occupancy_density": 0.09, "equipment_w_m2": 14.0, "cooling_w_m2": 120.0, "hvac_system_type": "VRF", "cop_cool_nom": 3.8, "cop_heat_nom": 3.6, "fan_eff": 0.62},
    "Large lecture building": {**SETUP_DEFAULTS, "building_type": "Large lecture building", "area_m2": 9000.0, "floors": 5, "n_spaces": 70, "occupancy_density": 0.12, "lighting_w_m2": 12.0, "cooling_w_m2": 115.0},
    "Improved envelope case": {**SETUP_DEFAULTS, "building_type": "Educational building - improved envelope", "wall_u": 0.35, "roof_u": 0.22, "window_u": 1.8, "shgc": 0.28, "glazing_ratio": 0.24, "infiltration_ach": 0.30},
}

SWITCH_DEFAULTS = {
    "sw_use_envelope": True,
    "sw_use_walls": True,
    "sw_use_roof": True,
    "sw_use_windows": True,
    "sw_use_solar": True,
    "sw_use_infiltration": True,
    "sw_use_internal_gains": True,
    "sw_use_people_gains": True,
    "sw_use_lighting_gains": True,
    "sw_use_equipment_gains": True,
    "sw_use_hvac_fans": True,
    "sw_use_hvac_pumps": False,
    "sw_use_hvac_aux": False,
    "sw_use_cooling": True,
    "sw_use_heating": True,
    "sw_use_degradation": True,
    "sw_use_carbon": True,
    "sw_use_maintenance_cost": True,
    "sw_use_zone_analysis": True,
    "sw_use_validation": True,
    "sw_use_benchmark": True,
    "sw_use_surrogate": True,
}

for k, v in SETUP_DEFAULTS.items():
    st.session_state.setdefault(k, v)
for k, v in SWITCH_DEFAULTS.items():
    st.session_state.setdefault(k, v)
st.session_state.setdefault("custom_setup_library", {})


def apply_setup_dict(data: Dict[str, Any]):
    for k, v in data.items():
        if k in SETUP_DEFAULTS or k in SWITCH_DEFAULTS:
            st.session_state[k] = v


def current_setup_dict() -> Dict[str, Any]:
    data = {k: st.session_state.get(k, v) for k, v in SETUP_DEFAULTS.items()}
    data.update({k: st.session_state.get(k, v) for k, v in SWITCH_DEFAULTS.items()})
    return data


def default_zone_table() -> pd.DataFrame:
    return pd.DataFrame([
        {"zone_name": "Lecture_01", "zone_type": "Lecture", "area_m2": 200.0, "occ_density": 0.12, "term_factor": 0.95, "break_factor": 0.20, "summer_factor": 0.10},
        {"zone_name": "Office_01", "zone_type": "Office", "area_m2": 120.0, "occ_density": 0.06, "term_factor": 0.85, "break_factor": 0.55, "summer_factor": 0.35},
        {"zone_name": "Lab_01", "zone_type": "Lab", "area_m2": 180.0, "occ_density": 0.08, "term_factor": 0.90, "break_factor": 0.45, "summer_factor": 0.30},
        {"zone_name": "Corridor", "zone_type": "Corridor", "area_m2": 100.0, "occ_density": 0.01, "term_factor": 0.60, "break_factor": 0.45, "summer_factor": 0.35},
        {"zone_name": "Service_01", "zone_type": "Service", "area_m2": 80.0, "occ_density": 0.02, "term_factor": 0.70, "break_factor": 0.65, "summer_factor": 0.60},
    ])


def download_file_button(path: str | Path, label: str, key: str | None = None):
    path = Path(path)
    if path.exists() and path.is_file():
        with path.open("rb") as f:
            st.download_button(label, f.read(), file_name=path.name, key=key or f"dl_{path.name}")


def collect_switches() -> Dict[str, bool]:
    return {k: bool(st.session_state.get(k, v)) for k, v in SWITCH_DEFAULTS.items()}


def build_model_inputs() -> tuple[BuildingSpec, HVACConfig, Dict[str, bool]]:
    switches = collect_switches()
    bldg = BuildingSpec(
        building_type=str(st.session_state["building_type"]),
        location=str(st.session_state["location"]),
        conditioned_area_m2=float(st.session_state["area_m2"]),
        floors=int(st.session_state["floors"]),
        n_spaces=int(st.session_state["n_spaces"]),
        occupancy_density_p_m2=float(st.session_state["occupancy_density"]),
        lighting_w_m2=float(st.session_state["lighting_w_m2"]) if switches["sw_use_lighting_gains"] else 0.0,
        equipment_w_m2=float(st.session_state["equipment_w_m2"]) if switches["sw_use_equipment_gains"] else 0.0,
        airflow_m3h_m2=float(st.session_state["airflow_m3h_m2"]) if switches["sw_use_hvac_fans"] else 0.0,
        infiltration_ach=float(st.session_state["infiltration_ach"]) if switches["sw_use_infiltration"] else 0.0,
        sensible_w_per_person=float(st.session_state["sensible_w_per_person"]) if switches["sw_use_people_gains"] else 0.0,
        cooling_intensity_w_m2=float(st.session_state["cooling_w_m2"]) if switches["sw_use_cooling"] else 0.0,
        heating_intensity_w_m2=float(st.session_state["heating_w_m2"]) if switches["sw_use_heating"] else 0.0,
        wall_u=float(st.session_state["wall_u"]) if switches["sw_use_walls"] else 0.0,
        roof_u=float(st.session_state["roof_u"]) if switches["sw_use_roof"] else 0.0,
        window_u=float(st.session_state["window_u"]) if switches["sw_use_windows"] else 0.0,
        shgc=float(st.session_state["shgc"]) if switches["sw_use_solar"] else 0.0,
        glazing_ratio=float(st.session_state["glazing_ratio"]) if switches["sw_use_windows"] else 0.0,
    )
    cfg = HVACConfig(
        years=int(st.session_state["years"]),
        hvac_system_type=str(st.session_state["hvac_system_type"]),
        USE_HVAC_PRESET=bool(st.session_state["use_hvac_preset"]),
        COP_COOL_NOM=float(st.session_state["cop_cool_nom"]),
        COP_HEAT_NOM=float(st.session_state["cop_heat_nom"]),
        COP_AGING_RATE=float(st.session_state["cop_aging_rate"]),
        FAN_EFF=float(st.session_state["fan_eff"]),
        T_SET=float(st.session_state["t_set"]),
        T_SP_MIN=float(st.session_state["t_sp_min"]),
        T_SP_MAX=float(st.session_state["t_sp_max"]),
        AF_MIN=float(st.session_state["af_min"]),
        AF_MAX=float(st.session_state["af_max"]),
        RF_STAR=float(st.session_state["rf_star"]),
        B_FOUL=float(st.session_state["b_foul"]),
        DP_CLEAN=float(st.session_state["dp_clean"]),
        DP_WARN=float(st.session_state["dp_warn"]),
        DP_THRESH=float(st.session_state["dp_thresh"]),
        DP_MAX=float(st.session_state["dp_max"]),
        DUST_RATE=float(st.session_state["dust_rate"]),
        K_CLOG=float(st.session_state["k_clog"]),
        DEG_TRIGGER=float(st.session_state["deg_trigger"]),
        E_PRICE=float(st.session_state["energy_price"]),
        CO2_FACTOR=float(st.session_state["co2_factor"]) if switches["sw_use_carbon"] else 0.0,
        COST_FILTER=float(st.session_state["cost_filter"]) if switches["sw_use_maintenance_cost"] else 0.0,
        COST_HX=float(st.session_state["cost_hx"]) if switches["sw_use_maintenance_cost"] else 0.0,
        FILTER_INTERVAL=int(st.session_state["filter_interval"]),
        HX_INTERVAL=int(st.session_state["hx_interval"]),
        W_ENERGY=float(st.session_state["w_energy"]),
        W_DEGRAD=float(st.session_state["w_degrad"]),
        W_COMFORT=float(st.session_state["w_comfort"]),
        W_CARBON=float(st.session_state["w_carbon"]),
        degradation_model=str(st.session_state["degradation_model"]),
        LINEAR_DEG_PER_DAY=float(st.session_state["linear_deg_per_day"]),
        EXP_DEG_RATE_PER_DAY=float(st.session_state["exp_deg_rate_per_day"]),
        USE_ENVELOPE=switches["sw_use_envelope"],
        USE_SOLAR=switches["sw_use_solar"],
        USE_INFILTRATION=switches["sw_use_infiltration"],
        USE_INTERNAL_GAINS=switches["sw_use_internal_gains"],
        USE_PEOPLE_GAINS=switches["sw_use_people_gains"],
        USE_LIGHTING_GAINS=switches["sw_use_lighting_gains"],
        USE_EQUIPMENT_GAINS=switches["sw_use_equipment_gains"],
        USE_HVAC_FANS=switches["sw_use_hvac_fans"],
        USE_COOLING=switches["sw_use_cooling"],
        USE_HEATING=switches["sw_use_heating"],
        USE_DEGRADATION=switches["sw_use_degradation"],
    )
    return bldg, cfg, switches


def render_switch_group(title: str, items: list[tuple[str, str]], columns: int = 2):
    st.markdown(f"#### {title}")
    cols = st.columns(columns)
    for i, (key, label) in enumerate(items):
        with cols[i % columns]:
            st.checkbox(label, key=key)


# -----------------------------
# Main tabs: setup first, then scenario modelling
# -----------------------------
tabs = st.tabs([
    "Building Identity & Setup",
    "Parameter Switches",
    "Scenario Modeling",
    "Extra UI Tools",
    "KPI Charts",
    "Surrogate Train / Predict",
    "Exports",
    "Guide",
])

with tabs[0]:
    st.subheader("Building identity and configuration setup")
    st.markdown("The setup is placed before scenario modelling. Use the saved setup selector, then adjust each configuration group separately.")

    c1, c2, c3 = st.columns([2.0, 1.0, 1.2])
    setup_names = list(PRESET_SETUPS.keys()) + list(st.session_state["custom_setup_library"].keys())
    selected_setup = c1.selectbox("Saved building configuration selector", setup_names)
    setup_name_to_save = c2.text_input("New setup name", "My building setup")
    with c3:
        st.write("")
        st.write("")
        if st.button("Apply selected setup"):
            setup_data = PRESET_SETUPS.get(selected_setup) or st.session_state["custom_setup_library"].get(selected_setup, {})
            apply_setup_dict(setup_data)
            st.rerun()
    c1, c2, c3 = st.columns(3)
    if c1.button("Save current setup in session"):
        st.session_state["custom_setup_library"][setup_name_to_save] = current_setup_dict()
        st.success(f"Saved setup: {setup_name_to_save}")
    c2.download_button("Download current setup JSON", setup_to_json_bytes(current_setup_dict()), file_name="building_setup.json", mime="application/json")
    setup_upload = c3.file_uploader("Load setup JSON", type=["json"], key="setup_json_upload")
    if setup_upload is not None:
        try:
            apply_setup_dict(setup_from_upload(setup_upload))
            st.success("Setup loaded. The page will refresh with the uploaded values.")
            st.rerun()
        except Exception as e:
            st.error(f"Could not load setup JSON: {e}")

    st.markdown("---")
    with st.expander("1. Building identity", expanded=True):
        c1, c2 = st.columns(2)
        c1.text_input("Building type", key="building_type")
        c2.text_input("Location / weather source label", key="location")

    with st.expander("2. Geometry", expanded=True):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Conditioned area (m²)", min_value=1.0, step=100.0, key="area_m2")
        c2.number_input("Floors", min_value=1, step=1, key="floors")
        c3.number_input("Number of spaces", min_value=1, step=1, key="n_spaces")

    with st.expander("3. Envelope", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.number_input("Wall U-value (W/m²K)", min_value=0.0, step=0.05, key="wall_u")
        c2.number_input("Roof U-value (W/m²K)", min_value=0.0, step=0.05, key="roof_u")
        c3.number_input("Window U-value (W/m²K)", min_value=0.0, step=0.10, key="window_u")
        c1, c2, c3 = st.columns(3)
        c1.number_input("SHGC", min_value=0.0, max_value=1.0, step=0.01, key="shgc")
        c2.number_input("Glazing ratio", min_value=0.0, max_value=1.0, step=0.01, key="glazing_ratio")
        c3.number_input("Infiltration (ACH)", min_value=0.0, step=0.1, key="infiltration_ach")

    with st.expander("4. Internal loads", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("General occupancy density (person/m²)", min_value=0.0, step=0.01, key="occupancy_density")
        c2.number_input("Lighting power density (W/m²)", min_value=0.0, step=1.0, key="lighting_w_m2")
        c3.number_input("Equipment power density (W/m²)", min_value=0.0, step=1.0, key="equipment_w_m2")
        c4.number_input("Sensible heat per person (W)", min_value=0.0, step=5.0, key="sensible_w_per_person")

    with st.expander("5. HVAC sizing and component", expanded=True):
        st.markdown("This section is expanded to follow the older interface style while still passing values into `HVACConfig`.")
        c1, c2, c3, c4 = st.columns(4)
        hvac_types = list(HVAC_PRESETS.keys())
        current_hvac = st.session_state.get("hvac_system_type", "Chiller_AHU")
        c1.selectbox("HVAC system type", hvac_types, index=hvac_types.index(current_hvac) if current_hvac in hvac_types else 0, key="hvac_system_type")
        c2.checkbox("Apply selected HVAC preset", key="use_hvac_preset")
        c3.number_input("Simulation years", min_value=1, max_value=50, step=1, key="years")
        c4.number_input("Airflow intensity (m³/h·m²)", min_value=0.0, step=0.1, key="airflow_m3h_m2")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Cooling design intensity (W/m²)", min_value=0.0, step=5.0, key="cooling_w_m2")
        c2.number_input("Heating design intensity (W/m²)", min_value=0.0, step=5.0, key="heating_w_m2")
        c3.number_input("Nominal cooling COP", min_value=0.1, step=0.1, key="cop_cool_nom")
        c4.number_input("Nominal heating COP", min_value=0.1, step=0.1, key="cop_heat_nom")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Fan total efficiency", min_value=0.01, max_value=1.0, step=0.01, key="fan_eff")
        c2.number_input("Clean static pressure ΔP clean (Pa)", min_value=0.0, step=10.0, key="dp_clean")
        c3.number_input("Warning static pressure ΔP warn (Pa)", min_value=0.0, step=10.0, key="dp_warn")
        c4.number_input("Replacement threshold ΔP thresh (Pa)", min_value=0.0, step=10.0, key="dp_thresh")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Maximum static pressure ΔP max (Pa)", min_value=1.0, step=10.0, key="dp_max")
        c2.number_input("Electricity price", min_value=0.0, step=0.01, key="energy_price")
        c3.number_input("Electricity CO₂ factor (kg/kWh)", min_value=0.0, step=0.01, key="co2_factor")
        c4.number_input("Filter replacement cost", min_value=0.0, step=10.0, key="cost_filter")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("HX cleaning cost", min_value=0.0, step=10.0, key="cost_hx")
        c2.number_input("Filter interval (days)", min_value=1, step=1, key="filter_interval")
        c3.number_input("HX interval (days)", min_value=1, step=1, key="hx_interval")
        c4.info("Pump and auxiliary switches are retained for compatibility; the current engine primarily models compressor and fan energy.")

    with st.expander("6. Control and optimization weights", expanded=False):
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Main setpoint T_SET (°C)", min_value=10.0, max_value=35.0, step=0.5, key="t_set")
        c2.number_input("Minimum setpoint T_SP_MIN (°C)", min_value=10.0, max_value=35.0, step=0.5, key="t_sp_min")
        c3.number_input("Maximum setpoint T_SP_MAX (°C)", min_value=10.0, max_value=35.0, step=0.5, key="t_sp_max")
        c4.number_input("Minimum airflow factor AF_MIN", min_value=0.0, max_value=1.0, step=0.05, key="af_min")
        c1, c2, c3, c4 = st.columns(4)
        c1.number_input("Maximum airflow factor AF_MAX", min_value=0.0, max_value=1.5, step=0.05, key="af_max")
        c2.number_input("Energy objective weight", min_value=0.0, max_value=1.0, step=0.05, key="w_energy")
        c3.number_input("Degradation objective weight", min_value=0.0, max_value=1.0, step=0.05, key="w_degrad")
        c4.number_input("Comfort objective weight", min_value=0.0, max_value=1.0, step=0.05, key="w_comfort")
        st.number_input("Carbon objective weight", min_value=0.0, max_value=1.0, step=0.05, key="w_carbon")

    with st.expander("7. Degradation parameters", expanded=False):
        c1, c2, c3 = st.columns(3)
        models = ["physics", "linear_ts", "exponential_ts"]
        c1.selectbox("Degradation model", models, index=models.index(st.session_state.get("degradation_model", "physics")), key="degradation_model", format_func=lambda x: {"physics":"Physics-based fouling/clogging", "linear_ts":"Linear time-series", "exponential_ts":"Exponential time-series"}[x])
        c2.number_input("COP aging rate", min_value=0.0, step=0.001, format="%.4f", key="cop_aging_rate")
        c3.number_input("RF* fouling asymptote", min_value=0.0, format="%.6f", key="rf_star")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Fouling growth constant B", min_value=0.0, step=0.001, format="%.3f", key="b_foul")
        c2.number_input("Dust accumulation rate", min_value=0.0, step=0.1, key="dust_rate")
        c3.number_input("Clogging coefficient", min_value=0.0, step=0.1, key="k_clog")
        c1, c2, c3 = st.columns(3)
        c1.number_input("Degradation trigger", min_value=0.0, max_value=1.5, step=0.01, key="deg_trigger")
        c2.number_input("Linear degradation slope/day", min_value=0.0, step=0.00001, format="%.6f", key="linear_deg_per_day")
        c3.number_input("Exponential degradation rate/day", min_value=0.0, step=0.00001, format="%.6f", key="exp_deg_rate_per_day")

    bldg_preview, cfg_preview, switches_preview = build_model_inputs()
    st.markdown("### Current model setup preview")
    c1, c2 = st.columns(2)
    with c1:
        st.json(asdict(bldg_preview), expanded=False)
    with c2:
        cfg_json = asdict(cfg_preview)
        st.json({k: cfg_json[k] for k in cfg_json if k.startswith("USE_") or k in ["hvac_system_type", "COP_COOL_NOM", "COP_HEAT_NOM", "FAN_EFF", "T_SET", "degradation_model"]}, expanded=False)

with tabs[1]:
    st.subheader("Parameter switches / Quick Control")
    st.markdown("Switches neutralize selected model terms while keeping `hvac_v3_engine.py` as the only numerical model.")
    c1, c2, c3, c4 = st.columns(4)
    if c1.button("Enable all main terms"):
        for k in SWITCH_DEFAULTS:
            st.session_state[k] = True
        st.session_state["sw_use_hvac_pumps"] = False
        st.session_state["sw_use_hvac_aux"] = False
        st.rerun()
    if c2.button("Thermal-only: no degradation"):
        for k in SWITCH_DEFAULTS:
            st.session_state[k] = True
        st.session_state["sw_use_degradation"] = False
        st.rerun()
    if c3.button("Envelope + weather only"):
        for k in SWITCH_DEFAULTS:
            st.session_state[k] = False
        for k in ["sw_use_envelope", "sw_use_walls", "sw_use_roof", "sw_use_windows", "sw_use_solar", "sw_use_infiltration", "sw_use_cooling", "sw_use_heating", "sw_use_carbon"]:
            st.session_state[k] = True
        st.rerun()
    if c4.button("Disable all optional post-tools"):
        for k in ["sw_use_zone_analysis", "sw_use_validation", "sw_use_benchmark", "sw_use_surrogate"]:
            st.session_state[k] = False
        st.rerun()

    c1, c2 = st.columns(2)
    with c1:
        render_switch_group("Envelope and site", [
            ("sw_use_envelope", "Use envelope load component"),
            ("sw_use_walls", "Include wall parameter"),
            ("sw_use_roof", "Include roof parameter"),
            ("sw_use_windows", "Include window/glazing parameter"),
            ("sw_use_solar", "Include solar gains"),
            ("sw_use_infiltration", "Include infiltration"),
        ], columns=2)
        render_switch_group("Internal loads", [
            ("sw_use_internal_gains", "Use internal gains"),
            ("sw_use_people_gains", "Include people sensible gains"),
            ("sw_use_lighting_gains", "Include lighting gains"),
            ("sw_use_equipment_gains", "Include equipment gains"),
        ], columns=2)
    with c2:
        render_switch_group("HVAC, degradation, and outputs", [
            ("sw_use_hvac_fans", "Include HVAC fan energy"),
            ("sw_use_hvac_pumps", "Pump switch retained for reporting"),
            ("sw_use_hvac_aux", "Auxiliary switch retained for reporting"),
            ("sw_use_cooling", "Include cooling load"),
            ("sw_use_heating", "Include heating load"),
            ("sw_use_degradation", "Include degradation effect"),
            ("sw_use_carbon", "Calculate carbon emissions"),
            ("sw_use_maintenance_cost", "Include maintenance cost"),
        ], columns=2)
        render_switch_group("Post-processing tools", [
            ("sw_use_zone_analysis", "Enable zone analysis"),
            ("sw_use_validation", "Enable validation upload"),
            ("sw_use_benchmark", "Enable benchmark/sensitivity"),
            ("sw_use_surrogate", "Enable surrogate modelling tab"),
        ], columns=2)

    st.markdown("### Active switch state")
    st.dataframe(pd.DataFrame([collect_switches()]).T.rename(columns={0: "enabled"}), use_container_width=True)

with tabs[2]:
    st.subheader("Scenario modeling")
    bldg, cfg, parameter_switches = build_model_inputs()
    c1, c2, c3, c4 = st.columns(4)
    axis_options = ["baseline_scenario", "one_severity", "one_strategy", "two_axis", "three_axis"]
    axis_mode = c1.selectbox(
        "Analysis mode",
        axis_options,
        index=2,
        format_func=lambda x: {
            "baseline_scenario": "Baseline Scenario only",
            "one_severity": "One-axis severity",
            "one_strategy": "One-axis strategy S0–S3",
            "two_axis": "Strategy × severity",
            "three_axis": "Strategy × severity × climate",
        }[x],
    )
    fixed_strategy = c2.selectbox("Strategy selection S0–S3", list(SCENARIOS.keys()), index=3, format_func=lambda x: f"{x} — {SCENARIOS[x]}")
    fixed_severity = c3.selectbox("Fixed severity", list(SEVERITY_LEVELS.keys()), index=1)
    fixed_climate = c4.selectbox("Fixed climate", list(CLIMATE_LEVELS.keys()), index=0)

    c1, c2, c3 = st.columns([1.3, 1.0, 2.0])
    weather_mode_ui = c1.selectbox(
        "Weather source",
        ["synthetic", "upload_csv_epw", "epw_path", "csv_path"],
        format_func=lambda x: {"synthetic":"Synthetic daily weather", "upload_csv_epw":"Upload CSV/EPW directly", "epw_path":"EPW path", "csv_path":"CSV path"}[x],
    )
    random_state = int(c2.number_input("Random state", min_value=1, value=42, step=1))
    out_dir = c3.text_input("Output folder", "v3_run")

    uploaded_weather = None
    weather_df = None
    epw_path = None
    csv_path = None
    if weather_mode_ui == "upload_csv_epw":
        uploaded_weather = st.file_uploader("Upload weather file (.csv or .epw)", type=["csv", "epw", "txt"], key="weather_upload")
        if uploaded_weather is not None:
            try:
                weather_df = read_weather_upload(uploaded_weather)
                st.session_state["uploaded_weather_df"] = weather_df
                st.success(f"Weather upload parsed successfully: {len(weather_df)} daily records")
                st.dataframe(weather_df.head(), use_container_width=True)
            except Exception as e:
                st.error(f"Weather upload error: {e}")
    elif weather_mode_ui == "epw_path":
        epw_path = st.text_input("EPW file path", "")
    elif weather_mode_ui == "csv_path":
        csv_path = st.text_input("CSV weather file path", "")

    c1, c2, c3 = st.columns(3)
    include_baseline_layer = c1.checkbox("Export baseline no-degradation files", value=True)
    include_baseline_as_scenario = c2.checkbox("Add Baseline Scenario to main output calculation", value=True)
    use_zone_occ = c3.checkbox("Use zone-specific occupancy input", value=False)
    zone_df = None
    if use_zone_occ:
        zone_df = st.data_editor(default_zone_table(), num_rows="dynamic", use_container_width=True, key="zone_editor")

    if not parameter_switches["sw_use_degradation"]:
        st.info("Degradation is switched off. Scenario runs will use the same engine with degradation terms neutralized; the Baseline Scenario option remains available as a separate output case.")

    if st.button("Run selected model", type="primary"):
        try:
            if weather_mode_ui == "upload_csv_epw":
                weather_df = st.session_state.get("uploaded_weather_df")
                if weather_df is None:
                    st.warning("Upload a CSV/EPW weather file first.")
                    st.stop()
                engine_weather_mode = "uploaded"
            elif weather_mode_ui == "epw_path":
                engine_weather_mode = "epw"
            elif weather_mode_ui == "csv_path":
                engine_weather_mode = "csv"
            else:
                engine_weather_mode = "synthetic"

            result = run_scenario_model(
                output_dir=out_dir,
                axis_mode=axis_mode,
                bldg=bldg,
                cfg=cfg,
                weather_mode=engine_weather_mode,
                epw_path=epw_path if epw_path else None,
                csv_path=csv_path if csv_path else None,
                weather_df=weather_df,
                fixed_strategy=fixed_strategy,
                fixed_severity=fixed_severity,
                fixed_climate=fixed_climate,
                zone_df=zone_df,
                random_state=random_state,
                include_baseline_layer=include_baseline_layer,
                degradation_model=st.session_state["degradation_model"],
                include_baseline_as_scenario=include_baseline_as_scenario,
                parameter_switches=parameter_switches,
            )
            st.session_state["last_result"] = result
            st.session_state["last_result_dir"] = out_dir
            st.session_state["last_zone_df"] = zone_df

            tables = build_detailed_tables(out_dir, bldg=bldg, cfg=cfg, zone_df=zone_df if parameter_switches["sw_use_zone_analysis"] else None)
            detailed_paths = save_detailed_outputs(out_dir, tables)
            st.session_state["last_detailed_paths"] = detailed_paths
            st.success("Model run and detailed outputs finished.")
            st.json({**result, "extra_detailed_outputs": detailed_paths})
            summary_path = Path(result["summary_csv"])
            if summary_path.exists():
                st.dataframe(pd.read_csv(summary_path), use_container_width=True)
        except Exception as e:
            st.exception(e)

with tabs[3]:
    st.subheader("Extra UI tools: validation, benchmark sensitivity, zone tables, upload handling")
    target_folder = st.text_input("Result folder for extra tools", st.session_state.get("last_result_dir", "v3_run"), key="extra_folder")
    bldg, cfg, parameter_switches = build_model_inputs()
    paths = find_result_paths(target_folder)
    if paths["summary"].exists():
        summary_df = pd.read_csv(paths["summary"])
        if parameter_switches["sw_use_validation"]:
            st.markdown("### Validation upload")
            vfile = st.file_uploader("Upload validation CSV from DesignBuilder, EnergyPlus, measured data, or published reference", type=["csv"], key="validation_file")
            if vfile is not None:
                validation_df = load_validation_file(vfile)
                comparison = build_validation_comparison(summary_df, validation_df, source_name=Path(vfile.name).stem)
                comparison_path = Path(target_folder) / "validation_comparison.csv"
                comparison.to_csv(comparison_path, index=False)
                st.dataframe(comparison, use_container_width=True)
                download_file_button(comparison_path, "Download validation_comparison.csv")
        else:
            st.info("Validation upload is disabled in Parameter Switches.")

        if parameter_switches["sw_use_benchmark"]:
            st.markdown("### Benchmark / sensitivity summary")
            if (Path(target_folder) / "benchmark_summary.csv").exists():
                bench = pd.read_csv(Path(target_folder) / "benchmark_summary.csv")
            else:
                tables = build_detailed_tables(target_folder, bldg=bldg, cfg=cfg, zone_df=st.session_state.get("last_zone_df"))
                save_detailed_outputs(target_folder, tables)
                bench = tables["benchmark_summary"]
            st.dataframe(bench, use_container_width=True)
            if len(bench) and "energy_delta_pct" in bench.columns and "scenario_combo_3axis" in bench.columns:
                st.bar_chart(bench.set_index("scenario_combo_3axis")["energy_delta_pct"])
        else:
            st.info("Benchmark/sensitivity summary is disabled in Parameter Switches.")

        st.markdown("### Zone analysis")
        zone_path = Path(target_folder) / "zone_analysis.csv"
        if parameter_switches["sw_use_zone_analysis"] and zone_path.exists():
            zdf = pd.read_csv(zone_path)
            st.dataframe(zdf.head(500), use_container_width=True)
            download_file_button(zone_path, "Download zone_analysis.csv")
        elif not parameter_switches["sw_use_zone_analysis"]:
            st.info("Zone analysis is disabled in Parameter Switches.")
        else:
            st.info("Run the model with zone-specific occupancy enabled to generate zone analysis.")
    else:
        st.info("Run a model first, or type an existing result folder.")

with tabs[4]:
    st.subheader("KPI charts")
    folder = Path(st.text_input("Result folder", st.session_state.get("last_result_dir", "v3_run"), key="kpi_folder"))
    if folder.exists():
        paths = find_result_paths(folder)
        if paths["summary"].exists():
            kpi = pd.read_csv(paths["summary"])
            st.dataframe(kpi, use_container_width=True)
            for metric in ["Total Energy MWh", "Mean Degradation Index", "Mean Comfort Deviation C", "Total CO2 tonne"]:
                if metric in kpi.columns and "scenario_combo_3axis" in kpi.columns:
                    st.line_chart(kpi.set_index("scenario_combo_3axis")[metric])
        figs = folder / "figures"
        if figs.exists():
            img_files = sorted(figs.glob("*.png"))[:24]
            cols = st.columns(2)
            for i, img in enumerate(img_files):
                with cols[i % 2]:
                    st.image(str(img), caption=img.name, use_container_width=True)
    else:
        st.info("No result folder found yet.")

with tabs[5]:
    st.subheader("Train CatBoost surrogate")
    _, _, parameter_switches = build_model_inputs()
    if not parameter_switches["sw_use_surrogate"]:
        st.info("Surrogate modelling is disabled in Parameter Switches.")
    else:
        dataset_path = st.text_input("Input dataset CSV", str(Path(st.session_state.get("last_result_dir", "v3_run")) / "matrix_ml_dataset.csv"))
        surrogate_out = st.text_input("Surrogate output folder", "v3_surrogate")
        n_iter_search = int(st.number_input("CatBoost search iterations", min_value=2, value=6, step=1))
        shap_sample = int(st.number_input("SHAP sample size", min_value=100, value=1000, step=100))
        if st.button("Train CatBoost surrogate"):
            try:
                result = train_surrogate_models(dataset_path, surrogate_out, n_iter_search, shap_sample, int(42))
                st.success("Surrogate training finished.")
                st.json(result)
                p = Path(result["metrics_csv"])
                if p.exists():
                    st.dataframe(pd.read_csv(p), use_container_width=True)
            except Exception as e:
                st.exception(e)

with tabs[6]:
    st.subheader("Exports and results")
    folder = Path(st.text_input("Folder to inspect/export", st.session_state.get("last_result_dir", "v3_run"), key="export_folder"))
    if folder.exists():
        csvs = sorted(folder.glob("*.csv"))
        st.write(f"CSV files found: {len(csvs)}")
        for csvf in csvs[:24]:
            with st.expander(csvf.name):
                try:
                    st.dataframe(pd.read_csv(csvf).head(100), use_container_width=True)
                except Exception as e:
                    st.warning(str(e))
                download_file_button(csvf, f"Download {csvf.name}", key=f"download_{csvf.name}")
        for special in ["results_export.xlsx", "detailed_outputs.xlsx", "results_report.pdf", "surrogate_export.xlsx", "surrogate_report.pdf"]:
            download_file_button(folder / special, f"Download {special}", key=f"download_{special}")
        if st.button("Create ZIP bundle for this run"):
            zip_path = create_zip_from_folder(folder)
            st.success(f"ZIP created: {zip_path}")
            download_file_button(zip_path, "Download ZIP bundle")
    else:
        st.info("No folder found yet.")

with tabs[7]:
    st.subheader("Deployment guide")
    st.markdown(
        """
        **Run locally**

        ```bash
        pip install -r requirements.txt
        streamlit run streamlit_app.py
        ```

        **Recommended workflow**

        1. Open **Building Identity & Setup** first and choose or save a building setup.  
        2. Open **Parameter Switches** and include/exclude model terms.  
        3. Open **Scenario Modeling** and select Baseline Scenario, S0, S1, S2, S3, or scenario matrices.  
        4. Upload EPW/CSV weather directly or use the synthetic weather generator.  
        5. Export CSV, Excel, PDF, figures, and detailed post-processing sheets.  
        6. Use validation, benchmark, zone analysis, and surrogate training as needed.

        **Note**: The UI never duplicates the numerical model. It builds `BuildingSpec` and `HVACConfig`, then calls `run_scenario_model()` from `hvac_v3_engine.py`.
        """
    )
