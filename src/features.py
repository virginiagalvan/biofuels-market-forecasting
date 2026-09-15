"""
Feature engineering for the country-year biofuels consumption panel.

General principle: every feature for year Y is built using only
information available up to Y (or Y-1 for autoregressive features), so
the setup is consistent with real-world use: "given what we know about
the country today, how much will it consume next year."
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

CONTINENT_MAP_PATH = Path(__file__).parent / "continent_map.json"


def add_continent(df):
    with open(CONTINENT_MAP_PATH) as f:
        mapping = json.load(f)
    df = df[df["iso_code"] != "ANT"].copy()  # entity dissolved in 2010, no current ISO continent
    df["continent"] = df["iso_code"].map(mapping)
    return df


def add_macro_features(df):
    df = df.copy()
    df["gdp_per_capita"] = df["gdp"] / df["population"]
    return df


def add_lag_features(df, lag_cols=("biofuel_consumption", "biofuel_share_energy", "gdp_per_capita")):
    """
    Adds the previous year's value of each column, per country. These
    are the features that keep the model from "seeing" the actual
    target of the year it's predicting.
    """
    df = df.sort_values(["country", "year"]).copy()
    for col in lag_cols:
        df[f"{col}_lag1"] = df.groupby("country")[col].shift(1)
    return df


def add_cagr(df, col="biofuel_consumption", window=5, new_col="biofuel_cagr_5y"):
    """
    Compound annual growth rate over the last `window` years, per
    country, using only data up to the previous year (does not include
    the target year itself).
    """
    df = df.sort_values(["country", "year"]).copy()

    def _cagr(series):
        shifted = series.shift(1).astype(float)
        past = shifted.shift(window - 1).replace(0, np.nan)
        growth = (shifted / past) ** (1 / window) - 1
        return growth.replace([np.inf, -np.inf], np.nan)

    df[new_col] = df.groupby("country")[col].transform(_cagr)
    return df


def flag_top80_consumers(df, col="biofuel_consumption", train_years=None):
    """
    Flags the countries that account for 80% of cumulative world
    consumption (computed only over the training years, so no future
    information leaks in). Used for the business metric (WMAE).
    """
    base = df if train_years is None else df[df["year"].isin(train_years)]
    totals = base.groupby("country")[col].sum().sort_values(ascending=False)
    cum_share = totals.cumsum() / totals.sum()
    top80_countries = set(cum_share[cum_share <= 0.80].index) | {cum_share.index[0]}
    df = df.copy()
    df["top80_consumer"] = df["country"].isin(top80_countries).astype(int)
    return df, top80_countries


def build_features(df):
    df = add_continent(df)
    df = add_macro_features(df)
    df = add_lag_features(df)
    df = add_cagr(df)
    return df
