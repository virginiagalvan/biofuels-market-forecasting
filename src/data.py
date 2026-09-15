"""
Loading and cleaning of the base dataset.

Source: Our World in Data - Global Energy dataset
(https://github.com/owid/energy-data), which in turn consolidates:
- Energy Institute - Statistical Review of World Energy (2025)
- Ember - Yearly Electricity Data (2026)
- Maddison Project Database 2023 (GDP)
- Population based on various sources (2024)

"Biofuels" is reported as a single consolidated metric covering
biodiesel and ethanol together.
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_PATH = REPO_ROOT / "data" / "raw" / "owid_energy_raw.csv"

RAW_URL = "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv"

COLUMNS = [
    "country",
    "year",
    "iso_code",
    "population",
    "gdp",
    "biofuel_consumption",
    "biofuel_cons_per_capita",
    "biofuel_share_energy",
    "biofuel_electricity",
    "oil_consumption",
    "energy_per_capita",
    "renewables_share_energy",
    "electricity_generation",
]

YEAR_MIN = 2000
YEAR_MAX = 2024


def load_raw(path=None):
    """Load the already-downloaded raw CSV (see data/raw/README.md)."""
    return pd.read_csv(path or DEFAULT_RAW_PATH, usecols=COLUMNS)


def filter_real_countries(df):
    """
    OWID includes regional aggregates (World, Africa, High-income
    countries...) alongside real countries. Aggregates don't have a
    3-letter iso_code.
    """
    return df[df["iso_code"].notna() & (df["iso_code"].str.len() == 3)].copy()


def build_panel(df):
    """
    Country-year panel 2000-2024, keeping only rows where
    biofuel_consumption is reported (not null). A value of 0 is real
    information (the country has no relevant production/consumption),
    it is not dropped.
    """
    panel = df[(df["year"] >= YEAR_MIN) & (df["year"] <= YEAR_MAX)].copy()
    panel = panel[panel["biofuel_consumption"].notna()].copy()
    return panel.sort_values(["country", "year"]).reset_index(drop=True)


def load_clean_panel(raw_path=None):
    df = load_raw(raw_path)
    df = filter_real_countries(df)
    df = build_panel(df)
    return df
