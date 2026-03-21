import sys
sys.path.append('..')

import pandas as pd  
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as mpatches

# loading data
CHARTS_DIR    = "../charts/"
PROCESSED_DIR = "../data/processed/"


def plot_country_metrics(df: pd.DataFrame, countries: list,  metric: str, figsize = (12, 6)):
    '''Plot the specified metric for the given countries and seasons.
    
    Each country has a unique line marker. Where data is NaN (indicating seasons 
    where no teams from that country participated), a special 'X' marker is displayed
    to indicate missing data.
    '''
    
    # Define different markers for each country
    markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h']
    
    # Metric descriptions for meaningful titles
    metric_descriptions = {
        'total_matches': 'Total Matches per Season',
        'win_rate': 'Win Rate (%)',
        'loss_rate': 'Loss Rate (%)',
        'draw_rate': 'Draw Rate (%)',
        'ppg_3': 'Points Per Game (3pts for Win)',
        'ppg_2': 'Points Per Game (2pts for Win)',
        'goal_diff': 'Goal Difference',
        'gf_pg': 'Goals For per Game',
        'ga_pg': 'Goals Against per Game',
        'gdpg': 'Goal Difference per Game'
    }
    
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=figsize)
    
    dataset = df[df['country'].isin(countries)].copy()
    
    # Get color palette for consistent coloring
    palette = sns.color_palette("husl", len(countries))
    color_map = {country: palette[i] for i, country in enumerate(countries)}
    
    # Track if we've added the NaN label (add only once to legend)
    nan_label_added = False
    
    for i, country in enumerate(countries):
        country_data = dataset[dataset['country'] == country].sort_values('season')
        marker = markers[i % len(markers)]
        color = color_map[country]
        
        # Plot line with markers for non-NaN values (gaps appear at NaN)
        ax.plot(country_data['season'], country_data[metric], 
                marker=marker, label=country, linewidth=1.5, 
                markersize=4, color=color)
        
        # Add special 'X' markers for NaN values to indicate "no data"
        nan_mask = country_data[metric].isna()
        if nan_mask.any():
            nan_seasons = country_data.loc[nan_mask, 'season'].values
            # Position NaN markers slightly below the minimum y value
            y_min = ax.get_ylim()[0]
            nan_y = [y_min] * len(nan_seasons)
            
            ax.scatter(nan_seasons, nan_y, marker='x', s=200, 
                      color=color, linewidths=1.5, alpha=0.6, zorder=5,
                      label='No matches' if not nan_label_added else '')
            
            if not nan_label_added:
                nan_label_added = True
    
    # Adjust y-axis to accommodate NaN markers at the bottom
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min - (y_max - y_min) * 0.12, y_max)
    
    # Use descriptive title if metric is in the descriptions dictionary
    metric_title = metric_descriptions.get(metric, metric)
    
    plt.xticks(rotation=70)
    plt.title(f"{metric_title} by Country over Seasons")
    plt.xlabel("Season")
    plt.ylabel(metric_title)
    plt.legend(title='Country', fontsize=9)
    plt.tight_layout()
    plt.show()


# ── Era labels (single source of truth) ──────────────────────────────────────
ERA_PRE    = "Pre-Golden Era"
ERA_GOLDEN = "Golden Era"
ERA_POST   = "Post-Golden Era"

# Ordered list – useful for sorting / CategoricalDtype later
ERA_ORDER = [ERA_PRE, ERA_GOLDEN, ERA_POST]

# -- Palette ------------------------------------------------------------------
DARK_BG      = "#1C1C1C"
PANEL_BG     = "#242424"
ITALY_BLUE   = "#4C9BE8"
GOLD_COLOR   = "#D4AF37"
GRID_COLOR   = "#333333"
TEXT_COLOR   = "#E0E0E0"
SUBTEXT      = "#999999"

ERA_BAND_COLORS = {
    "Pre-Golden Era" : "#555555",
    "Golden Era"     : "#D4AF37",
    "Post-Golden Era": "#C0392B",
}

METRICS = [
    ("win_rate", "Win rate"),
    ("gdpg_pt",  "Goal difference per game per team"),
]


def plot_italy_season_trend(
    italy_cs:   pd.DataFrame,
    charts_dir: str  = "../charts/",
    country_name: str = "Ita",
    filename:   str  = "4a_italy_season_trend.png",
    dpi:        int  = 150,
) -> None:
    """
    Plot Italy's win_rate and gdpg_pt across all seasons with era bands.

    Parameters
    ----------
    italy_cs     : Italy country_stats with 'era', 'win_rate', 'gdpg_pt'
    charts_dir   : output folder  (default '../charts/')
    country_name : label used in chart title
    filename     : output filename
    dpi          : image resolution
    """
    # -- Validate columns -----------------------------------------------------
    for col in ["season", "era", "win_rate", "gdpg_pt"]:
        if col not in italy_cs.columns:
            raise KeyError(f"Column '{col}' missing from italy_cs. "
                           f"Run filter_country + add_derived_metrics first.")

    # -- Sort seasons chronologically -----------------------------------------
    df = italy_cs.copy()
    df["_yr"] = df["season"].apply(
        lambda s: int(str(s).split("/")[0]) if "/" in str(s) else int(str(s)))
    df = df.sort_values("_yr").reset_index(drop=True)
    seasons = df["season"].tolist()
    x       = np.arange(len(seasons))

    # -- Era boundary indices -------------------------------------------------
    era_spans = _era_spans(df)

    # -- Figure ---------------------------------------------------------------
    fig, axes = plt.subplots(
        len(METRICS), 1,
        figsize=(16, 5 * len(METRICS)),
        facecolor=DARK_BG,
        sharex=True,
    )
    fig.suptitle(
        f"{country_name}  —  European competition performance across all seasons",
        fontsize=15, fontweight="bold",
        color=TEXT_COLOR, y=1.01,
    )

    for ax, (col, label) in zip(axes, METRICS):
        ax.set_facecolor(PANEL_BG)

        # Era bands
        for era, (x0, x1) in era_spans.items():
            ax.axvspan(x0 - 0.5, x1 + 0.5,
                       color=ERA_BAND_COLORS[era], alpha=0.12, zorder=1)

        # Era mean lines
        for era in ERA_ORDER:
            sub  = df[df["era"] == era]
            if sub.empty:
                continue
            idxs = sub.index.tolist()
            mean = sub[col].mean()
            ax.hlines(mean, idxs[0] - 0.4, idxs[-1] + 0.4,
                      colors=ERA_BAND_COLORS[era],
                      linewidth=2, linestyle="--",
                      alpha=0.85, zorder=3,
                      label=f"{era} mean ({mean:.3f})")

        # Main line
        vals = df[col].values
        ax.plot(x, vals, color=ITALY_BLUE,
                linewidth=2.2, zorder=4)
        ax.scatter(x, vals, color=ITALY_BLUE,
                   s=30, zorder=5)

        # Zero line for gdpg_pt
        if col == "gdpg_pt":
            ax.axhline(0, color=SUBTEXT, linewidth=0.8,
                       linestyle=":", zorder=2)

        # Labels
        ax.set_ylabel(label, color=TEXT_COLOR, fontsize=11)
        ax.tick_params(colors=TEXT_COLOR, labelsize=8)
        ax.spines[["top","right","left","bottom"]].set_visible(False)
        ax.grid(axis="y", color=GRID_COLOR, linewidth=0.6, zorder=0)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_COLOR)
        ax.legend(fontsize=8, loc="upper left",
                  facecolor=PANEL_BG, edgecolor=GRID_COLOR,
                  labelcolor=TEXT_COLOR)

    # x-axis labels on bottom panel only
    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(seasons, rotation=60,
                              fontsize=7.5, color=TEXT_COLOR)

    # Era legend (patches)
    patches = [
        mpatches.Patch(color=ERA_BAND_COLORS[e], alpha=0.5, label=e)
        for e in ERA_ORDER
    ]
    fig.legend(handles=patches,
               loc="lower center", ncol=3,
               fontsize=9, framealpha=0,
               labelcolor=TEXT_COLOR,
               bbox_to_anchor=(0.5, -0.03))

    plt.tight_layout()
    _save(fig, charts_dir, filename, dpi)
    plt.show()


# -- Helpers ------------------------------------------------------------------

def _era_spans(df: pd.DataFrame) -> dict[str, tuple[int, int]]:
    """Return {era: (first_index, last_index)} for background shading."""
    spans = {}
    for era in ERA_ORDER:
        idxs = df.index[df["era"] == era].tolist()
        if idxs:
            spans[era] = (idxs[0], idxs[-1])
    return spans


def _save(fig, charts_dir: str, filename: str, dpi: int) -> None:
    import os
    os.makedirs(charts_dir, exist_ok=True)
    path = os.path.join(charts_dir, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=DARK_BG)
    print(f"  Saved -> {path}")