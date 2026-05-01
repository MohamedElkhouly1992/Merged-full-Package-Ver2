# HVAC ROM-Degradation Suite

A Streamlit software package for reduced-order HVAC energy, degradation, climate scenario, maintenance strategy, validation, benchmark, zone-analysis, and CatBoost surrogate modelling.

## Important design rule

`hvac_v3_engine.py` is the main scientific framework and the single source of numerical calculations. The Streamlit UI does **not** duplicate the main model. It builds `BuildingSpec` and `HVACConfig`, applies parameter switches, then calls:

- `run_scenario_model()`
- `train_surrogate_models()`

## Main updates in this version

1. **Building Identity & Setup is now the first tab** before Scenario Modeling.
2. Setup is divided into separate configuration sections:
   - Building identity
   - Geometry
   - Envelope
   - Internal loads
   - HVAC sizing and component
   - Control and optimization weights
   - Degradation parameters
3. A **saved building setup selector** was added:
   - Apply built-in setup presets
   - Save current setup in the session
   - Download setup as JSON
   - Upload setup JSON
4. A **Parameter Switches / Quick Control** tab was added.
5. Each major model component can be included or excluded:
   - Envelope
   - Walls, roof, windows
   - Solar
   - Infiltration
   - Internal gains
   - People, lighting, equipment gains
   - HVAC fans
   - Cooling
   - Heating
   - Degradation
   - Carbon
   - Maintenance cost
   - Validation, benchmark, zone analysis, surrogate tools
6. HVAC sizing and component inputs were expanded to include:
   - HVAC type
   - Custom/preset COP handling
   - Cooling/heating design intensity
   - Fan efficiency
   - Static pressure limits
   - Setpoints and airflow bounds
   - Energy price and CO2 factor
   - Maintenance intervals and costs
7. **Baseline Scenario** was added as a separate analysis mode and can also be appended to the main output calculation.
8. Direct weather upload supports CSV and EPW through Streamlit.
9. Detailed post-processing outputs include:
   - `fuel_breakdown.csv`
   - `comfort.csv`
   - `site_data.csv`
   - `internal_gains.csv`
   - `validation_template.csv`
   - `validation_comparison.csv`
   - `benchmark_summary.csv`
   - `zone_analysis.csv`
   - `kpi_summary.csv`
   - `detailed_outputs.xlsx`

## Local installation

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

If `streamlit` is not recognized:

```bash
python -m streamlit run streamlit_app.py
```

## Quick engine test

```bash
python run_example.py
```

This creates an example result folder and verifies that the engine, baseline scenario, and detailed-output sheets work.

## Recommended Streamlit Community Cloud deployment

1. Upload the extracted files to a public GitHub repository.
2. Make sure these files are in the repository root:
   - `streamlit_app.py`
   - `hvac_v3_engine.py`
   - `report_addons.py`
   - `requirements.txt`
3. Open Streamlit Community Cloud.
4. Choose the GitHub repository.
5. Set main file path:

```text
streamlit_app.py
```

6. Deploy.

## Recommended research workflow

1. Open **Building Identity & Setup**.
2. Select or save a building configuration.
3. Open **Parameter Switches** and include/exclude required calculation components.
4. Open **Scenario Modeling**.
5. Select:
   - Baseline Scenario only
   - One-axis severity
   - One-axis strategy S0-S3
   - Strategy × severity
   - Strategy × severity × climate
6. Select the weather source:
   - Synthetic daily weather
   - Uploaded CSV/EPW
   - CSV path
   - EPW path
7. Run the model.
8. Use Extra UI Tools for validation, benchmark, and zone analysis.
9. Export CSV, Excel, PDF, figures, and ZIP bundle.
10. Train surrogate models using `matrix_ml_dataset.csv` or `three_axis_ml_dataset.csv`.

## Notes on switches

The switches are implemented through `HVACConfig` and neutralized inputs. This preserves the same calculation framework while allowing research experiments such as:

- degradation on/off
- internal gains on/off
- solar on/off
- infiltration on/off
- fan energy on/off
- cooling/heating on/off

Pump and auxiliary switches are retained for interface compatibility with the older app, but the current reduced-order engine primarily models compressor/HVAC load and fan energy.
