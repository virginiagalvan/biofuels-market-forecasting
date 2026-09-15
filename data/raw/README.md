# Data source

`owid_energy_raw.csv` is a subset (real countries, 2000-2024, relevant
columns) of the public **Our World in Data - Global Energy** dataset
(https://github.com/owid/energy-data), downloaded on 2026-09-04.

OWID consolidates, for the columns used in this project:

- `biofuel_consumption`, `biofuel_share_energy`, `biofuel_electricity`,
  `biofuel_cons_per_capita`: **Energy Institute - Statistical Review of
  World Energy (2025)**.
- `population`: Population based on various sources (2024).
- `gdp`: Maddison Project Database 2023.
- `oil_consumption`, `energy_per_capita`, `renewables_share_energy`,
  `electricity_generation`: Energy Institute / Ember Yearly Electricity
  Data, depending on the column.

`biofuel_consumption` (and the related `biofuel_*` columns, except
`biofuel_share_energy` which is a percentage and `biofuel_cons_per_capita`
which is kWh per person) report biofuels as a single consolidated metric
covering biodiesel and ethanol together, not split by fuel type, and are
measured in terawatt-hours (TWh), OWID's standard unit for energy
consumption columns.

## Variables and units

Raw columns (as downloaded) and the features built from them
(`src/features.py`), with units. Verified against the raw values for a
known country-year (United States, 2019) where the unit wasn't already
documented by OWID.

| column | unit | notes |
|---|---|---|
| `country`, `iso_code` | — | 3-letter ISO code |
| `year` | — | 2000-2024 |
| `population` | people | raw headcount |
| `gdp` | international-$ (PPP, Maddison Project convention) | total, not per capita |
| `gdp_per_capita` | international-$ per person | engineered: `gdp / population` |
| `energy_per_capita` | kWh per person | primary energy |
| `oil_consumption` | TWh | |
| `renewables_share_energy` | % | share of primary energy (not of electricity) |
| `electricity_generation` | TWh | |
| `biofuel_consumption` (target) | TWh | biodiesel + ethanol combined |
| `biofuel_share_energy` | % | share of primary energy |
| `biofuel_cons_per_capita` | kWh per person | |
| `continent` | — | engineered from `iso_code`, `src/continent_map.json` |
| `*_lag1` (`biofuel_consumption_lag1`, `biofuel_share_energy_lag1`, `gdp_per_capita_lag1`) | same as base column | engineered: previous year's value, per country |
| `biofuel_cagr_5y` | dimensionless rate (e.g. 0.05 = 5%/year) | engineered: 5-year compound annual growth rate, per country |
| `top80_consumer` | 0/1 flag | engineered: used to weight the business metric (WMAE), not a model input |

## How to regenerate this file

```python
import pandas as pd

df = pd.read_csv(
    "https://raw.githubusercontent.com/owid/energy-data/master/owid-energy-data.csv",
    usecols=[
        "country", "year", "iso_code", "population", "gdp",
        "biofuel_consumption", "biofuel_cons_per_capita", "biofuel_share_energy",
        "biofuel_electricity", "oil_consumption", "energy_per_capita",
        "renewables_share_energy", "electricity_generation",
    ],
)
real = df[df["iso_code"].notna() & (df["iso_code"].str.len() == 3)]
panel = real[(real["year"] >= 2000) & (real["year"] <= 2024)]
panel = panel[panel["biofuel_consumption"].notna()]
panel.to_csv("owid_energy_raw.csv", index=False)
```
