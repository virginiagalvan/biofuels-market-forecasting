"""
Plots: exploratory (matplotlib/seaborn) and interactive (Plotly).

Plotly is only imported inside the functions that need it, so that
importing this module for the matplotlib-only plots (used in the EDA
and modeling notebooks) doesn't require plotly to be installed.
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_target_distribution(df, target="biofuel_consumption", ax=None):
    ax = ax or plt.gca()
    df[target].plot(kind="hist", bins=40, ax=ax, log=True)
    ax.set_xlabel(f"{target} (TWh)")
    ax.set_ylabel("frequency (log scale)")
    ax.set_title("Distribution of biofuels consumption")
    return ax


def plot_model_comparison(results_df, metric="MAE"):
    import plotly.express as px

    fig = px.bar(
        results_df, x="model", y=metric, text=metric,
        title=f"Model comparison - {metric}", template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
    return fig


def plot_choropleth_latest_year(df, value_col="biofuel_consumption", year=None, title=None):
    import plotly.express as px

    year = year or df["year"].max()
    latest = df[df["year"] == year]
    fig = px.choropleth(
        latest, locations="iso_code", color=value_col, hover_name="country",
        color_continuous_scale="Greens",
        title=title or f"Biofuels consumption by country ({year})",
        template="plotly_white",
    )
    return fig


def plot_country_ranking(df, value_col="biofuel_consumption", year=None, top_n=15):
    import plotly.express as px

    year = year or df["year"].max()
    latest = df[df["year"] == year].sort_values(value_col, ascending=False).head(top_n)
    fig = px.bar(
        latest, x=value_col, y="country", orientation="h",
        title=f"Top {top_n} countries by biofuels consumption ({year})", template="plotly_white",
    )
    fig.update_layout(yaxis=dict(autorange="reversed"), xaxis_title="biofuels consumption (TWh)")
    return fig


def plot_cumulative_concentration(totals, cum_share, n_top80, top_n=15, ax=None,
                                   title="Consumption concentration by country"):
    """
    Pareto-style chart: bars for each country's individual share of
    world consumption (left axis), line for the running cumulative
    share (right axis), with a reference line at 80%. Makes the
    "N countries account for 80% of consumption" finding visible at a
    glance, instead of only readable from a table of numbers.
    """
    ax = ax or plt.gca()
    top = totals.head(top_n)
    individual_share = top / totals.sum()
    cum = cum_share.head(top_n)

    ax.bar(range(len(top)), individual_share.values, color="#4C72B0")
    ax.set_xticks(range(len(top)))
    ax.set_xticklabels(top.index, rotation=45, ha="right")
    ax.set_ylabel("individual share of world consumption")
    ax.set_title(title)

    ax2 = ax.twinx()
    ax2.plot(range(len(cum)), cum.values, color="#C44E52", marker="o", linewidth=2)
    ax2.axhline(0.80, color="#C44E52", linestyle="--", linewidth=1, alpha=0.6)
    ax2.text(len(cum) - 1, 0.80, " 80%", color="#C44E52", va="bottom", ha="right")
    ax2.axvline(n_top80 - 1, color="gray", linestyle=":", linewidth=1)
    ax2.set_ylabel("cumulative share", color="#C44E52")
    ax2.set_ylim(0, 1.05)
    return ax


def plot_correlation_heatmap(corr_df, ax=None, title="Correlation between numeric variables"):
    """
    Heatmap of the full pairwise correlation matrix (every numeric
    variable against every other one, not just against the target),
    with the coefficient annotated in each cell.
    """
    ax = ax or plt.gca()
    im = ax.imshow(corr_df.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr_df.columns)))
    ax.set_xticklabels(corr_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr_df.index)))
    ax.set_yticklabels(corr_df.index)
    for i in range(len(corr_df.index)):
        for j in range(len(corr_df.columns)):
            value = corr_df.values[i, j]
            ax.text(j, i, f"{value:.2f}", ha="center", va="center",
                     color="white" if abs(value) > 0.5 else "black", fontsize=8)
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return ax


def plot_top_n_ranking(df, value_col, label_col="country", top_n=10, ax=None,
                        xlabel=None, title=None, value_fmt="{:.1f}"):
    """
    Horizontal bar chart of the top N rows by `value_col`, labeled by
    `label_col`, with the value annotated at the end of each bar.
    Generic ranking chart, reused for both KPI tables (CAGR,
    penetration) instead of a plain table.
    """
    ax = ax or plt.gca()
    top = df.sort_values(value_col, ascending=False).head(top_n).iloc[::-1]
    bars = ax.barh(top[label_col], top[value_col], color="#4C72B0")
    for bar, value in zip(bars, top[value_col]):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2,
                 f" {value_fmt.format(value)}", va="center", fontsize=9)
    if "%" in value_fmt and ":.1%" in value_fmt:
        # value_fmt formats the fraction as a percentage (e.g. 0.05 ->
        # "5.0%"); match the axis ticks to that, not the raw fraction.
        import matplotlib.ticker as mtick
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlabel(xlabel or value_col)
    ax.set_title(title or f"Top {top_n} by {value_col}")
    return ax


def plot_feature_importance(feature_names, importances, top_n=15, title="Feature importance"):
    imp_df = pd.DataFrame({"feature": feature_names, "importance": importances})
    imp_df = imp_df[imp_df["importance"] > 0].sort_values("importance", ascending=False).head(top_n)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(imp_df["feature"][::-1], imp_df["importance"][::-1])
    ax.set_xlabel("importance")
    ax.set_title(title)
    return fig, imp_df


def plot_model_comparison_static(comparison_df, metric="MAE", ax=None,
                                  title="Model comparison"):
    """
    Static (matplotlib) bar chart of `metric` by model, sorted
    ascending, with the value annotated on top of each bar.
    """
    df_sorted = comparison_df.sort_values(metric)
    ax = ax or plt.gca()
    bars = ax.bar(df_sorted["model"], df_sorted[metric], color="#4C72B0")
    for bar, value in zip(bars, df_sorted[metric]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel(f"{metric} (TWh)" if metric in ("MAE", "RMSE", "WMAE") else metric)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=20)
    return ax


def plot_actual_vs_predicted(y_true, y_pred, ax=None,
                              title="Actual vs. predicted biofuels consumption"):
    """
    Scatter of actual vs. predicted values with a y=x reference line.
    A model that predicts perfectly would have every point on that
    line; distance from it is the error.
    """
    ax = ax or plt.gca()
    ax.scatter(y_true, y_pred, alpha=0.4, s=18, color="#4C72B0")
    lims = [0, max(max(y_true), max(y_pred)) * 1.05]
    ax.plot(lims, lims, color="#C44E52", linestyle="--", linewidth=1, label="y = x")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("actual consumption (TWh)")
    ax.set_ylabel("predicted consumption (TWh)")
    ax.set_title(title)
    ax.legend()
    return ax


def plot_business_error_summary(business_summary_df, ax=None,
                                 title="Model error as % of actual consumption, by group"):
    """
    Bar chart translating the model's error into the business terms
    used in notebooks/03_business_kpis.ipynb: error as a percentage of
    actual consumption, for the top-80% countries vs. the rest.
    """
    ax = ax or plt.gca()
    bars = ax.bar(business_summary_df.index, business_summary_df["error_pct_of_consumption"],
                   color=["#DD8452", "#4C72B0"])
    for bar, value in zip(bars, business_summary_df["error_pct_of_consumption"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                 f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("error (% of actual consumption)")
    ax.set_title(title)
    return ax
