import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


def filter_golden_era(
        cs:   pd.DataFrame,
        csh:  pd.DataFrame,
        csch: pd.DataFrame,
        rd:   pd.DataFrame | None = None,
        era_col: str = "era",
        ) -> dict[str, pd.DataFrame]:
    """
    Filter country_stats, country_season_highest,
    country_season_competition_highest (and optionally raw_data)
    to the Golden Era rows.

    Parameters
    ----------
    cs   : country_stats              (must have 'era' column)
    csh  : country_season_highest     (must have 'era' column)
    csch : country_season_competition_highest (must have 'era' column)
    rd   : raw_data                   (optional; pass None to skip)
    era_col : name of the era column  (default 'era')

    Returns
    -------
    dict with keys 'cs', 'csh', 'csch', and 'rd' (None if not passed)
    Each value is a reset-index copy filtered to Golden Era only.
    """
    
    # Era labels (single source of truth)
    ERA_PRE    = "Pre-Golden Era"
    ERA_GOLDEN = "Golden Era"
    ERA_POST   = "Post-Golden Era"

    
    inputs = {"cs": cs, "csh": csh, "csch": csch}
    if rd is not None:
        inputs["rd"] = rd

    result = {}
    print(f"\n  Filtering to: {ERA_GOLDEN}")
    print(f"  {'DataFrame':<40}  {'Rows in':>8}  {'Rows out':>9}")
    print("  " + "-" * 62)

    for key, df in inputs.items():
        if era_col not in df.columns:
            raise KeyError(
                f"'{era_col}' column missing from '{key}'. "
                "Run add_era_to_all() (step0_era.py) first."
            )
        filtered = (df[df[era_col] == ERA_GOLDEN]
                    .copy()
                    .reset_index(drop=True))
        result[key] = filtered
        print(f"  {key:<40}  {len(df):>8,}  {len(filtered):>9,}")

    if rd is None:
        result["rd"] = None

    # ── Coverage: seasons and countries present in Golden Era ────────────────
    ge_cs = result["cs"]
    seasons   = sorted(ge_cs["season"].unique(),
                        key=lambda s: int(s.split("/")[0]))
    countries = sorted(ge_cs["country"].unique())

    print(f"\n  Seasons  : {len(seasons)}  ({seasons[0]} → {seasons[-1]})")
    print(f"  Countries: {len(countries)}")

    return result

_REQUIRED = ["country", "season", "num_teams",
             "win_rate", "ppg_3", "ppg_2", "gdpg"]


def country_match_aggregates(ge_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate Golden Era country_stats to one row per country.

    Parameters
    ----------
    ge_cs : Golden Era country_stats  (output of step2a filter_golden_era['cs'])

    Returns
    -------
    agg : DataFrame, one row per country, sorted by avg_win_rate descending
    """
    missing = [c for c in _REQUIRED if c not in ge_cs.columns]
    if missing:
        raise KeyError(f"Missing columns in ge_cs: {missing}")

    df = ge_cs.copy()

    # Per-team metrics computed season-by-season before aggregating
    teams = df["num_teams"].replace(0, np.nan)
    df["ppg3_pt"] = df["ppg_3"] / teams
    df["ppg2_pt"] = df["ppg_2"] / teams
    df["gdpg_pt"] = df["gdpg"]  / teams

    agg = (df.groupby("country", sort=False)
             .agg(
                 n_seasons      = ("season",    "nunique"),
                 avg_num_teams  = ("num_teams", "mean"),
                 avg_win_rate   = ("win_rate",  "mean"),
                 avg_ppg3_pt    = ("ppg3_pt",   "mean"),
                 avg_ppg2_pt    = ("ppg2_pt",   "mean"),
                 avg_gdpg_pt    = ("gdpg_pt",   "mean"),
                 # Totals (useful context, not used in composite)
                 total_wins     = ("wins",          "sum"),
                 total_draws    = ("draws",         "sum"),
                 total_losses   = ("losses",        "sum"),
                 total_matches  = ("total_matches", "sum"),
             )
             .reset_index()
             .sort_values("avg_win_rate", ascending=False)
             .reset_index(drop=True))

    # NaN-safe rank: countries with no win_rate data get NaN rank, not a crash
    agg["rank_win_rate"] = (agg["avg_win_rate"]
                            .rank(ascending=False, method="min", na_option="bottom")
                            .astype("Int64"))   # nullable integer — holds NaN safely

    # ── Warn if any countries have incomplete data ────────────────────────────
    null_rows = agg[agg["avg_win_rate"].isna()]
    if not null_rows.empty:
        print(f"\n  WARNING: {len(null_rows)} country/countries have NaN avg_win_rate "
              f"(no match data in Golden Era) — ranked last:")
        print("  " + ", ".join(null_rows["country"].tolist()))

    # ── Print top-10 preview ──────────────────────────────────────────────────
    def _fmt(val, fmt=".3f"):
        """Format a value that might be NaN."""
        return f"{val:{fmt}}" if pd.notna(val) else "  —"

    print("\n  Per-country match aggregates (Golden Era) — top 10 by win rate")
    print(f"  {'#':>3}  {'Country':<20}  {'Seasons':>7}  "
          f"{'Win rate':>9}  {'PPG/team':>9}  {'GDpG/team':>10}")
    print("  " + "-" * 65)
    for _, r in agg.head(10).iterrows():
        print(f"  {r['rank_win_rate']:>3}  {r['country']:<20}  "
              f"{int(r['n_seasons']):>7}  "
              f"{_fmt(r['avg_win_rate']):>9}  "
              f"{_fmt(r['avg_ppg3_pt']):>9}  "
              f"{_fmt(r['avg_gdpg_pt']):>10}")

    # ── Highlight Italy ───────────────────────────────────────────────────────
    italy = agg[agg["country"].str.upper() == "ITA"]
    if not italy.empty:
        r = italy.iloc[0]
        print(f"\n  Italy  →  rank #{r['rank_win_rate']} of "
              f"{len(agg)} countries  "
              f"(win rate {_fmt(r['avg_win_rate'])})")

    return agg

def _ordinal(series: pd.Series) -> pd.Series:
    """
    Convert highest_round strings to integer ordinal (0-4). Unknown/null -> NaN.

    Values not in ROUND_GROUPS are replaced with NaN *before* building the
    Categorical, which avoids the Pandas4Warning about non-null entries not
    in the dtype's categories.
    """
    # Replace any value not in ROUND_GROUPS with NaN before categorising
    ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]

    known = set(ROUND_GROUPS)
    clean = series.where(series.isin(known), other=np.nan)

    # Warn once if unknown labels were found
    unknown = series[series.notna() & ~series.isin(known)]
    if not unknown.empty:
        counts = unknown.value_counts()
        print(f"  [depth] WARNING - {len(unknown)} row(s) have unrecognised "
              f"highest round values -> treated as NaN:")
        for val, n in counts.items():
            print(f"    '{val}'  ({n} row(s))")

    cat   = pd.Categorical(clean, categories=ROUND_GROUPS, ordered=True)
    codes = pd.Series(cat.codes, index=series.index, dtype="float")
    return codes.replace(-1, np.nan)


def country_depth_aggregates(ge_csh: pd.DataFrame,
                              round_col: str = "highest round") -> pd.DataFrame:
    """
    Aggregate Golden Era country_season_highest to one row per country.

    Parameters
    ----------
    ge_csh    : Golden Era country_season_highest
    round_col : column holding the grouped round label

    Returns
    -------
    agg : DataFrame, one row per country, sorted by avg_round_ord descending
    """
    ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]

    if round_col not in ge_csh.columns:
        raise KeyError(f"Column '{round_col}' not found in ge_csh.")

    df = ge_csh.copy()
    df["round_ord"] = _ordinal(df[round_col])

    agg = (df.groupby("country", sort=False)
             .agg(
                 n_seasons     = ("season",    "nunique"),
                 avg_round_ord = ("round_ord", "mean"),
                 max_round_ord = ("round_ord", "max"),
                 n_finals      = (round_col, lambda x: (x == "Final").sum()),
                 n_semis       = (round_col, lambda x: (x == "Semi Finals").sum()),
                 n_qf          = (round_col, lambda x: (x == "Quarter Finals").sum()),
             )
             .reset_index())

    agg["finals_rate"] = agg["n_finals"] / agg["n_seasons"]
    agg["sf_rate"]     = agg["n_semis"]  / agg["n_seasons"]

    # Best round label for display — NaN-safe
    agg["best_round"] = agg["max_round_ord"].apply(
        lambda x: ROUND_GROUPS[int(x)] if pd.notna(x) else "--"
    )

    agg = (agg.sort_values("avg_round_ord", ascending=False)
              .reset_index(drop=True))

    # NaN-safe rank: nullable Int64 so countries with no depth data don't crash
    agg["rank_depth"] = (agg["avg_round_ord"]
                         .rank(ascending=False, method="min", na_option="bottom")
                         .astype("Int64"))

    # Warn if any countries have no depth data at all
    null_rows = agg[agg["avg_round_ord"].isna()]
    if not null_rows.empty:
        print(f"\n  WARNING: {len(null_rows)} country/countries have NaN "
              f"avg_round_ord (no depth data in Golden Era) -- ranked last:")
        print("  " + ", ".join(null_rows["country"].tolist()))

    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if pd.notna(val) else "  --"

    # ── Print top-10 preview ──────────────────────────────────────────────────
    print("\n  Per-country tournament depth (Golden Era) -- top 10 by avg round")
    print(f"  {'#':>3}  {'Country':<20}  {'N':>4}  "
          f"{'Avg round':>10}  {'Finals':>7}  {'Semis':>6}  "
          f"{'F-rate':>7}  {'Best':>15}")
    print("  " + "-" * 78)
    for _, r in agg.head(10).iterrows():
        print(f"  {r['rank_depth']:>3}  {r['country']:<20}  "
              f"{int(r['n_seasons']):>4}  "
              f"{_fmt(r['avg_round_ord']):>10}  "
              f"{int(r['n_finals']):>7}  "
              f"{int(r['n_semis']):>6}  "
              f"{_fmt(r['finals_rate']):>7}  "
              f"{r['best_round']:>15}")

    italy = agg[agg["country"].str.upper() == "ITA"]
    if not italy.empty:
        r = italy.iloc[0]
        print(f"\n  Italy  ->  rank #{r['rank_depth']} of {len(agg)} countries  "
              f"(avg round ord {_fmt(r['avg_round_ord'])}, "
              f"{int(r['n_finals'])} finals)")

    return agg

def _ordinal(series: pd.Series) -> pd.Series:
    """
    Convert highest_round strings to integer ordinal (0-4). Unknown/null -> NaN.

    Unknown values are replaced with NaN *before* building the Categorical,
    avoiding the Pandas4Warning about non-null entries not in dtype categories.
    """
    ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]
    known = set(ROUND_GROUPS)
    clean = series.where(series.isin(known), other=np.nan)

    unknown = series[series.notna() & ~series.isin(known)]
    if not unknown.empty:
        counts = unknown.value_counts()
        print(f"  [comp] WARNING - {len(unknown)} row(s) have unrecognised "
              f"highest_round values -> treated as NaN:")
        for val, n in counts.items():
            print(f"    '{val}'  ({n} row(s))")

    cat = pd.Categorical(clean, categories=ROUND_GROUPS, ordered=True)
    return pd.Series(cat.codes, index=series.index, dtype="float").replace(-1, np.nan)


def _fmt(val, fmt=".3f") -> str:
    """Format a value that might be NaN."""
    return f"{val:{fmt}}" if pd.notna(val) else "  --"


def country_competition_breakdown(
    ge_csch: pd.DataFrame,
    round_col:   str = "highest_round",
    comp_col:    str = "competition",
    country_col: str = "country",
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Per-competition country rankings for the Golden Era.

    Parameters
    ----------
    ge_csch      : Golden Era country_season_competition_highest
    round_col    : column with grouped round label
    comp_col     : column with competition code
    country_col  : column with country name

    Returns
    -------
    by_comp : dict  {competition_code -> per-country DataFrame}
    pivot   : wide DataFrame  (country x competition, values = avg_round_ord)
              ready for heatmap plotting
    """
    ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]
    if round_col not in ge_csch.columns:
        raise KeyError(f"'{round_col}' not found in ge_csch.")

    df = ge_csch.copy()
    df["round_ord"] = _ordinal(df[round_col])

    competitions = sorted(df[comp_col].dropna().unique())
    by_comp      = {}

    print("\n  Per-competition breakdown (Golden Era)")
    print("  " + "=" * 72)

    for comp in competitions:
        sub = df[df[comp_col] == comp].copy()

        agg = (sub.groupby(country_col, sort=False)
                  .agg(
                      n_seasons     = ("season",    "nunique"),
                      avg_round_ord = ("round_ord", "mean"),
                      max_round_ord = ("round_ord", "max"),
                      n_finals      = (round_col, lambda x: (x == "Final").sum()),
                      n_semis       = (round_col, lambda x: (x == "Semi Finals").sum()),
                  )
                  .reset_index()
                  .sort_values("avg_round_ord", ascending=False)
                  .reset_index(drop=True))

        # NaN-safe rank
        agg["rank"] = (agg["avg_round_ord"]
                       .rank(ascending=False, method="min", na_option="bottom")
                       .astype("Int64"))

        # NaN-safe best round label
        agg["best_round"] = agg["max_round_ord"].apply(
            lambda x: ROUND_GROUPS[int(x)] if pd.notna(x) else "--"
        )

        by_comp[comp] = agg

        # ── Print per-competition top-8 ───────────────────────────────────────
        italy = agg[agg[country_col].str.upper() == "ITA"]
        if not italy.empty:
            italy_rank = f"#{italy['rank'].values[0]}"
            italy_avg  = _fmt(italy["avg_round_ord"].values[0], ".2f")
        else:
            italy_rank, italy_avg = "N/A", "--"

        print(f"\n  [{comp}]  Italy: rank {italy_rank} of {len(agg)} "
              f"| avg round ord {italy_avg}")
        print(f"  {'#':>3}  {'Country':<20}  {'N':>4}  "
              f"{'Avg round':>10}  {'Finals':>7}  {'Semis':>6}")
        print("  " + "-" * 55)
        for _, r in agg.head(8).iterrows():
            print(f"  {r['rank']:>3}  {r[country_col]:<20}  "
                  f"{int(r['n_seasons']):>4}  "
                  f"{_fmt(r['avg_round_ord']):>10}  "
                  f"{int(r['n_finals']):>7}  "
                  f"{int(r['n_semis']):>6}")

    # ── Pivot table (country x competition) ──────────────────────────────────
    pivot_rows = []
    for comp, agg in by_comp.items():
        for _, r in agg.iterrows():
            pivot_rows.append({
                country_col:    r[country_col],
                "competition":  comp,
                "avg_round_ord": r["avg_round_ord"],
            })

    pivot = (pd.DataFrame(pivot_rows)
               .pivot_table(index=country_col,
                            columns="competition",
                            values="avg_round_ord",
                            aggfunc="mean"))
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]

    print(f"\n  Pivot table shape: {pivot.shape}  "
          f"(countries x competitions)")

    return by_comp, pivot

DEFAULT_WEIGHTS = {
    "avg_round_ord" : 0.35,
    "avg_win_rate"  : 0.25,
    "avg_ppg3_pt"   : 0.25,
    "avg_gdpg_pt"   : 0.15,
}

_REQUIRED_MATCH = ["country", "avg_win_rate", "avg_ppg3_pt", "avg_gdpg_pt"]
_REQUIRED_DEPTH = ["country", "avg_round_ord", "n_finals", "n_semis",
                   "finals_rate", "n_seasons"]


def _minmax_norm(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.0, index=series.index)
    return (series - mn) / (mx - mn)


def build_composite(
    match_agg: pd.DataFrame,
    depth_agg: pd.DataFrame,
    weights:   dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Merge, normalise and rank all countries by composite score.

    Parameters
    ----------
    match_agg : output of step2b country_match_aggregates()
    depth_agg : output of step2c country_depth_aggregates()
    weights   : optional override dict  {column_name: weight}
                Must sum to 1.0.

    Returns
    -------
    rankings : DataFrame, one row per country, sorted by composite_score desc
               Includes all raw metrics, normalised pillars, and final rank.
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_w = sum(weights.values())
    if abs(total_w - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1.0, got {total_w:.4f}.")

    missing_m = [c for c in _REQUIRED_MATCH if c not in match_agg.columns]
    missing_d = [c for c in _REQUIRED_DEPTH if c not in depth_agg.columns]
    if missing_m or missing_d:
        raise KeyError(f"Missing cols — match: {missing_m}, depth: {missing_d}. "
                       "Run step2b and step2c first.")

    # ── Merge on country ──────────────────────────────────────────────────────
    merged = match_agg.merge(
        depth_agg[["country", "avg_round_ord", "max_round_ord",
                   "n_finals", "n_semis", "n_qf",
                   "finals_rate", "sf_rate", "best_round",
                   "n_seasons"]],
        on="country", how="outer", suffixes=("_match", "_depth"),
    )

    # If a country appears in one source but not the other, flag it
    match_only = merged["avg_win_rate"].isna().sum()
    depth_only = merged["avg_round_ord"].isna().sum()
    if match_only:
        print(f"  WARNING: {match_only} countries have depth data but no match data.")
    if depth_only:
        print(f"  WARNING: {depth_only} countries have match data but no depth data.")

    # ── Min-max normalise each pillar ────────────────────────────────────────
    norm_cols = {}
    for col in weights:
        if col not in merged.columns:
            raise KeyError(f"Weight column '{col}' not found after merge.")
        norm_col = f"{col}_norm"
        merged[norm_col] = _minmax_norm(merged[col].fillna(merged[col].median()))
        norm_cols[col] = norm_col

    # ── Composite score ───────────────────────────────────────────────────────
    merged["composite_score"] = sum(
        weights[col] * merged[norm_cols[col]]
        for col in weights
    )

    # ── Final ranking ────────────────────────────────────────────────────────
    merged = (merged.sort_values("composite_score", ascending=False)
                    .reset_index(drop=True))
    merged["rank"] = range(1, len(merged) + 1)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n  Final country rankings (Golden Era) — top 15 by composite score")
    print(f"  Weights: " +
          "  ".join(f"{k.replace('avg_','').replace('_ord','').replace('_pt','')}"
                    f"={v:.0%}" for k, v in weights.items()))
    print(f"\n  {'Rk':>3}  {'Country':<20}  {'Score':>7}  "
          f"{'WinRate':>8}  {'PPG/tm':>7}  {'GDpG/tm':>8}  "
          f"{'AvgRnd':>7}  {'Finals':>7}")
    print("  " + "-" * 75)

    for _, r in merged.head(15).iterrows():
        print(f"  {int(r['rank']):>3}  {r['country']:<20}  "
              f"{r['composite_score']:>7.4f}  "
              f"{r['avg_win_rate']:>8.3f}  "
              f"{r['avg_ppg3_pt']:>7.3f}  "
              f"{r['avg_gdpg_pt']:>8.3f}  "
              f"{r['avg_round_ord']:>7.3f}  "
              f"{int(r['n_finals'] if not pd.isna(r['n_finals']) else 0):>7}")

    italy = merged[merged["country"].str.upper() == "ITA"]
    if not italy.empty:
        r = italy.iloc[0]
        print(f"\n  ► Italy  →  rank #{int(r['rank'])} of {len(merged)}  "
              f"| composite score {r['composite_score']:.4f}")
        print(f"    win rate {r['avg_win_rate']:.3f}  "
              f"ppg/team {r['avg_ppg3_pt']:.3f}  "
              f"gdpg/team {r['avg_gdpg_pt']:.3f}  "
              f"avg round {r['avg_round_ord']:.3f}  "
              f"finals {int(r['n_finals'])}")

    return merged


ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]


# ── Shared helpers ────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0

def _sig_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return "ns"

def _effect_label(d: float) -> str:
    a = abs(d)
    if a < 0.2: return "negligible"
    if a < 0.5: return "small"
    if a < 0.8: return "medium"
    return "large"


MIN_N = 3


# Test 1: Mann-Whitney U — Italy vs pooled rest

def test_italy_vs_rest_match(
    ge_cs: pd.DataFrame,
    country_col: str = "country",
    country_name: str = "Ita",
) -> pd.DataFrame:
    """
    For each match-level metric, test whether Italy's season-by-season
    values are significantly higher than those of all other countries.

    Parameters
    ----------
    ge_cs        : Golden Era country_stats with derived per-team metrics
                   (must have ppg3_pt and gdpg_pt — run step2b logic first,
                    or add_derived_metrics if using the full cs slice)
    country_col  : country column name
    country_name : country to isolate

    Returns
    -------
    results DataFrame with one row per metric
    """
    # Add per-team cols if not already present
    df = ge_cs.copy()
    if "ppg3_pt" not in df.columns:
        teams = df["num_teams"].replace(0, np.nan)
        df["ppg3_pt"] = df["ppg_3"] / teams
        df["gdpg_pt"] = df["gdpg"]  / teams

    name_up  = country_name.strip().upper()
    italy    = df[df[country_col].str.upper() == name_up]
    rest     = df[df[country_col].str.upper() != name_up]

    metrics = {
        "win_rate" : "Win rate",
        "ppg3_pt"  : "PPG (3pt) per team",
        "gdpg_pt"  : "GDpG per team",
    }

    rows = []
    print(f"\n --- Mann-Whitney U: {country_name} vs rest --- ")
    print(f"  {'Metric':<22}  {'n_italy':>7}  {'n_rest':>7}  "
          f"{'U':>8}  {'p':>8}  {'sig':>4}  {'d':>6}  effect")
    print("  " + "-" * 80)

    for col, label in metrics.items():
        italy_vals = italy[col].dropna().values
        rest_vals  = rest [col].dropna().values
        row = {"metric": label, "n_italy": len(italy_vals),
               "n_rest": len(rest_vals)}

        if len(italy_vals) < MIN_N or len(rest_vals) < MIN_N:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan, effect="too few")
            rows.append(row)
            continue

        stat, p = mannwhitneyu(italy_vals, rest_vals, alternative="greater")
        d       = _cohen_d(italy_vals, rest_vals)
        row.update(U=stat, p=p, sig=_sig_stars(p), d=d, effect=_effect_label(d))
        rows.append(row)
        print(f"  {label:<22}  {len(italy_vals):>7}  {len(rest_vals):>7}  "
              f"{stat:>8.0f}  {p:>8.4f}  {_sig_stars(p):>4}  "
              f"{d:>+6.2f}  {_effect_label(d)}")

    print()
    return pd.DataFrame(rows)


def test_italy_vs_rest_depth(
    ge_csh: pd.DataFrame,
    # round_col:    str = "highest_round",
    round_col:    str = "highest round",
    country_col:  str = "country",
    country_name: str = "Ita",
) -> pd.DataFrame:
    """
    Test whether Italy's season-by-season round depth is significantly
    higher than all other countries' depths.

    Parameters
    ----------
    ge_csh       : Golden Era country_season_highest

    Returns
    -------
    results DataFrame with one row
    """
    df = ge_csh.copy()
    cat = pd.Categorical(df[round_col], categories=ROUND_GROUPS, ordered=True)
    df["round_ord"] = pd.Series(cat.codes, dtype="float").replace(-1, np.nan)

    name_up    = country_name.strip().upper()
    italy_vals = df[df[country_col].str.upper() == name_up]["round_ord"].dropna().values
    rest_vals  = df[df[country_col].str.upper() != name_up]["round_ord"].dropna().values

    row = {"metric": "Round depth (ord)",
           "n_italy": len(italy_vals), "n_rest": len(rest_vals)}

    print(f"  --- Mann-Whitney U: round depth --- ")
    print(f"  {'Metric':<22}  {'n_italy':>7}  {'n_rest':>7}  "
          f"{'U':>8}  {'p':>8}  {'sig':>4}  {'d':>6}  effect")
    print("  " + "-" * 80)

    if len(italy_vals) < MIN_N or len(rest_vals) < MIN_N:
        row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan, effect="too few")
        print("  SKIP — too few seasons")
    else:
        stat, p = mannwhitneyu(italy_vals, rest_vals, alternative="greater")
        d       = _cohen_d(italy_vals, rest_vals)
        row.update(U=stat, p=p, sig=_sig_stars(p), d=d, effect=_effect_label(d))
        print(f"  {'Round depth (ord)':<22}  {len(italy_vals):>7}  {len(rest_vals):>7}  "
              f"{stat:>8.0f}  {p:>8.4f}  {_sig_stars(p):>4}  "
              f"{d:>+6.2f}  {_effect_label(d)}")

    print()
    return pd.DataFrame([row])


# Test 2: Season-rank consistency 

def rank_consistency(
    ge_cs: pd.DataFrame,
    country_col:  str = "country",
    country_name: str = "Ita",
) -> pd.DataFrame:
    """
    Each Golden Era season: rank all countries by win_rate.
    Report Italy's mean rank, median rank, and how often it
    was in the top 3 / top 5.

    Returns
    -------
    season_ranks : DataFrame with one row per season showing Italy's rank
    """
    df = ge_cs.copy()
    seasons = sorted(df["season"].unique(),
                     key=lambda s: int(s.split("/")[0]))
    name_up = country_name.strip().upper()

    records = []
    for s in seasons:
        sub = df[df["season"] == s].copy()
        sub["season_rank"] = sub["win_rate"].rank(ascending=False, method="min")
        n_countries = len(sub)
        italy_row = sub[sub[country_col].str.upper() == name_up]
        if italy_row.empty:
            continue
        records.append({
            "season"     : s,
            "rank"       : int(italy_row["season_rank"].values[0]),
            "n_countries": n_countries,
            "win_rate"   : italy_row["win_rate"].values[0],
            "pct_rank"   : italy_row["season_rank"].values[0] / n_countries,
        })

    season_ranks = pd.DataFrame(records)
    if season_ranks.empty:
        print(f"  No seasons found for {country_name} — check country name.")
        return season_ranks

    mean_rank    = season_ranks["rank"].mean()
    median_rank  = season_ranks["rank"].median()
    top3_count   = (season_ranks["rank"] <= 3).sum()
    top5_count   = (season_ranks["rank"] <= 5).sum()
    n_seasons    = len(season_ranks)
    n_countries  = season_ranks["n_countries"].median()

    print(f" --- Season rank consistency: {country_name} --- ")
    print(f"  Seasons analysed : {n_seasons}")
    print(f"  Avg countries/season : {n_countries:.0f}")
    print(f"  Mean rank   : {mean_rank:.2f}  "
          f"(expected if random: {(n_countries+1)/2:.1f})")
    print(f"  Median rank : {median_rank:.0f}")
    print(f"  Top-3 finishes : {top3_count} / {n_seasons}  "
          f"({100*top3_count/n_seasons:.0f}%)")
    print(f"  Top-5 finishes : {top5_count} / {n_seasons}  "
          f"({100*top5_count/n_seasons:.0f}%)")
    print(f"\n  {'Season':<10}  {'Rank':>5}  {'/ N':>5}  {'Win rate':>9}")
    print("  " + "-" * 38)
    for _, r in season_ranks.iterrows():
        flag = " ◄ top 3" if r["rank"] <= 3 else ""
        print(f"  {r['season']:<10}  {int(r['rank']):>5}  "
              f"{int(r['n_countries']):>5}  {r['win_rate']:>9.3f}{flag}")
    print()

    return season_ranks


