import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu


def filter_country(
        cs:  pd.DataFrame,
        csh: pd.DataFrame,
        country_col: str = "country",
        country_name: str = "Ita",
        ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Filter country_stats and country_season_highest to a single country.
 
    Parameters
    ----------
    cs           : country_stats DataFrame  (must already have 'era' column)
    csh          : country_season_highest DataFrame  (must already have 'era')
    country_col  : name of the country column  (default 'country')
    country_name : country to filter to         (default 'Ita')
 
    Returns
    -------
    italy_cs, italy_csh  — filtered copies, index reset
    """
    name_upper = country_name.strip().upper()
 
    italy_cs  = (cs [cs [country_col].str.upper() == name_upper]
                 .copy().reset_index(drop=True))
    italy_csh = (csh[csh[country_col].str.upper() == name_upper]
                 .copy().reset_index(drop=True))
 
    # ── Basic coverage check ──────────────────────────────────────────────────
    print(f"\n  Filtering to: {country_name}")
    print(f"  country_stats          rows : {len(italy_cs)}")
    print(f"  country_season_highest rows : {len(italy_csh)}")
 
    for label, df in [("country_stats", italy_cs),
                      ("country_season_highest", italy_csh)]:
        if "era" not in df.columns:
            print(f"  WARNING: 'era' column missing from {label} — "
                  "run add_era_column() first (step0_era.py)")
        else:
            era_counts = df["era"].value_counts().sort_index()
            print(f"  {label} era breakdown:")
            for era, n in era_counts.items():
                print(f"      {era:<22}: {n} seasons")
 
    return italy_cs, italy_csh

def add_derived_metrics(italy_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Add normalised per-team metrics to italy_cs.
 
    Parameters
    ----------
    italy_cs : Italy-filtered country_stats (output of step1a filter_italy)
 
    Returns
    -------
    Copy of italy_cs with three new columns:
        ppg3_pt, ppg2_pt, gdpg_pt
    """
    _REQUIRED = ["num_teams", "ppg_3", "ppg_2", "gdpg", "win_rate"]
    
    missing = [c for c in _REQUIRED if c not in italy_cs.columns]
    if missing:
        raise KeyError(f"Missing columns in italy_cs: {missing}")
 
    out = italy_cs.copy()
 
    # Guard: avoid division by zero if num_teams is 0 or NaN
    teams = out["num_teams"].replace(0, np.nan)
 
    out["ppg3_pt"] = out["ppg_3"] / teams
    out["ppg2_pt"] = out["ppg_2"] / teams
    out["gdpg_pt"] = out["gdpg"]  / teams
 
    # ── Sanity check: print a small preview ───────────────────────────────────
    preview_cols = ["season", "era", "num_teams",
                    "win_rate", "ppg3_pt", "gdpg_pt"]
    print("\n  Derived metrics preview (first 5 rows):")
    print(out[preview_cols].head().to_string(index=False))
 
    null_counts = out[["ppg3_pt", "ppg2_pt", "gdpg_pt"]].isnull().sum()
    if null_counts.any():
        print(f"\n  WARNING – NaNs introduced (likely num_teams == 0):")
        print(null_counts[null_counts > 0].to_string())
 
    return out

def era_summary_table(italy_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Return a tidy summary DataFrame: one row per era, stats for each metric.

    Parameters
    ----------
    italy_cs : Italy country_stats with 'era', 'win_rate', 'ppg3_pt', 'gdpg_pt'
               (output of step1b add_derived_metrics)

    Returns
    -------
    summary : DataFrame with MultiIndex columns  (metric, stat)
              Rows are ordered  Pre → Golden → Post
    """

    # ── Era labels (single source of truth) ──────────────────────────────────────
    ERA_PRE    = "Pre-Golden Era"
    ERA_GOLDEN = "Golden Era"
    ERA_POST   = "Post-Golden Era"

    # Ordered list – useful for sorting / CategoricalDtype later
    ERA_ORDER = [ERA_PRE, ERA_GOLDEN, ERA_POST]

    # Metrics to summarise and their display labels
    METRICS = {
        "win_rate" : "Win rate",
        "ppg3_pt"  : "PPG (3pt) per team",
        "gdpg_pt"  : "GDpG per team",
    }

    missing = [c for c in list(METRICS) + ["era"]
               if c not in italy_cs.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}. "
                       "Run step1a and step1b first.")

    records = []
    for era in ERA_ORDER:
        sub = italy_cs[italy_cs["era"] == era]
        row = {"era": era, "n_seasons": len(sub)}
        for col in METRICS:
            vals = sub[col].dropna()
            row[f"{col}_mean"]   = vals.mean()
            row[f"{col}_std"]    = vals.std(ddof=1)
            row[f"{col}_median"] = vals.median()
            row[f"{col}_min"]    = vals.min()
            row[f"{col}_max"]    = vals.max()
        records.append(row)

    summary = pd.DataFrame(records).set_index("era")

    # ── Pretty-print ──────────────────────────────────────────────────────────
    print("\n  Per-era summary  (Italy · European competitions)")
    print(f"  {'Era':<22}  {'N':>2}  "
          + "  ".join(f"{'  ' + lbl + '  ':^28}" for lbl in METRICS.values()))
    print(f"  {'':22}  {'':2}  "
          + "  ".join(f"{'mean':>7} {'± std':>7} {'median':>8}"
                      for _ in METRICS))
    print("  " + "-" * 90)

    for era in ERA_ORDER:
        if era not in summary.index:
            continue
        r   = summary.loc[era]
        n   = int(r["n_seasons"])
        row = f"  {era:<22}  {n:>2}  "
        for col in METRICS:
            m  = r[f"{col}_mean"]
            s  = r[f"{col}_std"]
            md = r[f"{col}_median"]
            row += f"  {m:>7.3f} {s:>7.3f} {md:>8.3f}  "
        print(row)

    print()
    return summary


def era_delta_table(summary: pd.DataFrame) -> pd.DataFrame:
    """
    Show how much Golden Era differs from Pre and Post (absolute and %).

    Parameters
    ----------
    summary : output of era_summary_table()

    Returns
    -------
    delta : DataFrame with rows  (vs Pre-Golden Era, vs Post-Golden Era)
            and columns for each metric's absolute and % difference
    """
    # Metrics to summarise and their display labels
    METRICS = {
        "win_rate" : "Win rate",
        "ppg3_pt"  : "PPG (3pt) per team",
        "gdpg_pt"  : "GDpG per team",
    }

    if "Golden Era" not in summary.index:
        print("  No Golden Era rows found — skipping delta table.")
        return pd.DataFrame()

    golden = summary.loc["Golden Era"]
    rows   = []

    for compare_era in ["Pre-Golden Era", "Post-Golden Era"]:
        if compare_era not in summary.index:
            continue
        other = summary.loc[compare_era]
        row   = {"comparison": f"Golden vs {compare_era}"}
        for col in METRICS:
            g = golden[f"{col}_mean"]
            o = other [f"{col}_mean"]
            row[f"{col}_diff"] = g - o
            row[f"{col}_pct"]  = 100 * (g - o) / o if o != 0 else np.nan
        rows.append(row)

    delta = pd.DataFrame(rows).set_index("comparison")

    print("  Golden Era delta vs other eras:")
    print(f"  {'Comparison':<34}  "
          + "  ".join(f"{lbl:^22}" for lbl in METRICS.values()))
    print(f"  {'':34}  "
          + "  ".join(f"{'diff':>9} {'%':>9}" for _ in METRICS))
    print("  " + "-" * 80)

    for idx, r in delta.iterrows():
        row = f"  {idx:<34}  "
        for col in METRICS:
            d   = r[f"{col}_diff"]
            pct = r[f"{col}_pct"]
            row += f"  {d:>+9.3f} {pct:>+8.1f}%  "
        print(row)

    print()
    return delta


# ── Era labels (single source of truth) ──────────────────────────────────────
ERA_PRE    = "Pre-Golden Era"
ERA_GOLDEN = "Golden Era"
ERA_POST   = "Post-Golden Era"

# Ordered list – useful for sorting / CategoricalDtype later
ERA_ORDER = [ERA_PRE, ERA_GOLDEN, ERA_POST]

# ── Ordered group labels (single source of truth) ─────────────────────────────
ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]


def _apply_round_order(italy_csh: pd.DataFrame,
                        round_col: str = "highest_round") -> pd.DataFrame:
    """
    Cast highest_round to an ordered Categorical on a copy of the DataFrame.
    Returns the enriched copy.
    """
    out = italy_csh.copy()
    out[round_col] = pd.Categorical(
        out[round_col],
        categories=ROUND_GROUPS,
        ordered=True,
    )
    # Add numeric code for mean / std calculations (stage=0 … Final=4)
    out["round_ord"] = out[round_col].cat.codes.replace(-1, pd.NA)
    return out


def depth_summary_table(italy_csh: pd.DataFrame,
                        round_col: str = "highest_round") -> pd.DataFrame:
    """
    Per-era average, median and maximum deepest round reached by Italy.

    Parameters
    ----------
    italy_csh : Italy-filtered country_season_highest with 'era' column
    round_col : column holding the grouped round label

    Returns
    -------
    summary DataFrame indexed by era
    """
    df = _apply_round_order(italy_csh, round_col)

    records = []
    for era in ERA_ORDER:
        sub  = df[df["era"] == era]
        ords = sub["round_ord"].dropna()
        records.append({
            "era"        : era,
            "n_seasons"  : len(sub),
            "avg_ord"    : ords.mean(),
            "median_ord" : ords.median(),
            "max_ord"    : ords.max(),
            # Human-readable labels for the most common and best round
            "avg_round"  : (ROUND_GROUPS[round(ords.mean())]
                            if not ords.empty and not pd.isna(ords.mean())
                            else "—"),
            "best_round" : (ROUND_GROUPS[int(ords.max())]
                            if not ords.empty and not pd.isna(ords.max())
                            else "—"),
        })

    summary = pd.DataFrame(records).set_index("era")

    print("\n  Tournament depth by era  (Italy)")
    print(f"  {'Era':<22}  {'N':>2}  {'Avg ord':>8}  "
          f"{'Avg round':<16}  {'Best round':<16}")
    print("  " + "-" * 72)
    for era in ERA_ORDER:
        if era not in summary.index:
            continue
        r = summary.loc[era]
        print(f"  {era:<22}  {int(r['n_seasons']):>2}  "
              f"{r['avg_ord']:>8.2f}  "
              f"{r['avg_round']:<16}  {r['best_round']:<16}")
    print()
    return summary


def milestone_counts(italy_csh: pd.DataFrame,
                     round_col: str = "highest_round") -> pd.DataFrame:
    """
    Count how many seasons Italy reached each milestone per era.

    Milestones: Final, Semi Finals, Quarter Finals, Round of 16
    (each season contributes to the deepest milestone reached only)

    Returns
    -------
    pivot DataFrame: rows = era, columns = milestone rounds, values = count
    """
    df  = _apply_round_order(italy_csh, round_col)
    milestones = ["Final", "Semi Finals", "Quarter Finals", "Round of 16"]

    records = []
    for era in ERA_ORDER:
        sub = df[df["era"] == era]
        row = {"era": era, "n_seasons": len(sub)}
        for m in milestones:
            row[m] = (sub[round_col] == m).sum()
        row["stage_only"] = (sub[round_col] == "stage").sum()
        records.append(row)

    counts = pd.DataFrame(records).set_index("era")

    print("  Milestone counts by era  (seasons Italy reached each round):")
    cols = ["n_seasons"] + milestones + ["stage_only"]
    header = f"  {'Era':<22}  {'N':>9}  " + \
             "  ".join(f"{m:>14}" for m in milestones) + \
             f"  {'Stage only':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for era in ERA_ORDER:
        if era not in counts.index:
            continue
        r   = counts.loc[era]
        row = f"  {era:<22}  {int(r['n_seasons']):>9}  "
        row += "  ".join(f"{int(r[m]):>14}" for m in milestones)
        row += f"  {int(r['stage_only']):>12}"
        print(row)
    print()
    return counts

# ── Helpers ───────────────────────────────────────────────────────────────────

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    """Pooled-SD Cohen's d (positive = a > b)."""
    var_a = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b = np.var(b, ddof=1) if len(b) > 1 else 0.0
    pooled = np.sqrt((var_a + var_b) / 2)
    return (np.mean(a) - np.mean(b)) / pooled if pooled > 0 else 0.0


def _sig_stars(p: float) -> str:
    if p < 0.01:  return "***"
    if p < 0.05:  return "**"
    if p < 0.10:  return "*"
    return "ns"


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


MIN_N = 3   # minimum seasons per era to run a test


# ── Main test functions ───────────────────────────────────────────────────────

def test_match_metrics(italy_cs: pd.DataFrame) -> pd.DataFrame:
    """
    Run Mann-Whitney U + Cohen's d for win_rate, ppg3_pt, gdpg_pt:
      Golden Era  vs  Pre-Golden Era
      Golden Era  vs  Post-Golden Era

    Parameters
    ----------
    italy_cs : Italy country_stats with 'era' and derived metric columns
               (output of step1b add_derived_metrics)

    Returns
    -------
    results DataFrame with one row per (metric × comparison)
    """
    
    METRICS = {
    "win_rate" : "Win rate",
    "ppg3_pt"  : "PPG (3pt) per team",
    "gdpg_pt"  : "GDpG per team",
    }
    
    golden = italy_cs[italy_cs["era"] == "Golden Era"]
    rows   = []

    print("\n  ── Match-level metric tests ──────────────────────────────────")
    print(f"  {'Metric':<22}  {'vs':<22}  "
          f"{'U':>7}  {'p':>7}  {'sig':>4}  "
          f"{'d':>6}  {'effect':<12}")
    print("  " + "-" * 80)

    for col, label in METRICS.items():
        g_vals = golden[col].dropna().values
        for other_era in ["Pre-Golden Era", "Post-Golden Era"]:
            o_vals = italy_cs[italy_cs["era"] == other_era][col].dropna().values
            row    = {"metric": label, "comparison": f"vs {other_era}",
                      "n_golden": len(g_vals), "n_other": len(o_vals)}

            if len(g_vals) < MIN_N or len(o_vals) < MIN_N:
                row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                           effect="too few seasons")
                print(f"  {label:<22}  {other_era:<22}  "
                      f"  SKIP (n_golden={len(g_vals)}, n_other={len(o_vals)})")
                rows.append(row)
                continue

            stat, p = mannwhitneyu(g_vals, o_vals, alternative="greater")
            d       = _cohen_d(g_vals, o_vals)
            sig     = _sig_stars(p)
            eff     = _effect_label(d)

            row.update(U=stat, p=p, sig=sig, d=d, effect=eff)
            rows.append(row)
            print(f"  {label:<22}  {other_era:<22}  "
                  f"{stat:>7.0f}  {p:>7.4f}  {sig:>4}  "
                  f"{d:>+6.2f}  {eff:<12}")

    print()
    return pd.DataFrame(rows)


def test_depth(italy_csh: pd.DataFrame,
               round_col: str = "highest_round") -> pd.DataFrame:
    """
    Run Mann-Whitney U + Cohen's d on the ordinal round depth:
      Golden Era  vs  Pre-Golden Era
      Golden Era  vs  Post-Golden Era

    Parameters
    ----------
    italy_csh : Italy country_season_highest with 'era' and 'highest_round'

    Returns
    -------
    results DataFrame with one row per comparison
    """
    # Apply numeric encoding (same as step1d)
    ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]
    
    df = italy_csh.copy()
    df[round_col] = pd.Categorical(df[round_col],
                                   categories=ROUND_GROUPS, ordered=True)
    df["round_ord"] = df[round_col].cat.codes.replace(-1, pd.NA)

    golden = df[df["era"] == "Golden Era"]
    rows   = []

    print("  ── Tournament depth tests ────────────────────────────────────")
    print(f"  {'Metric':<22}  {'vs':<22}  "
          f"{'U':>7}  {'p':>7}  {'sig':>4}  "
          f"{'d':>6}  {'effect':<12}")
    print("  " + "-" * 80)

    g_vals = golden["round_ord"].dropna().values
    for other_era in ["Pre-Golden Era", "Post-Golden Era"]:
        o_vals = df[df["era"] == other_era]["round_ord"].dropna().values
        row    = {"metric": "Round depth (ord)", "comparison": f"vs {other_era}",
                  "n_golden": len(g_vals), "n_other": len(o_vals)}

        if len(g_vals) < MIN_N or len(o_vals) < MIN_N:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                       effect="too few seasons")
            print(f"  {'Round depth':<22}  {other_era:<22}  "
                  f"  SKIP (n={len(g_vals)} / {len(o_vals)})")
            rows.append(row)
            continue

        stat, p = mannwhitneyu(g_vals, o_vals, alternative="greater")
        d       = _cohen_d(g_vals, o_vals)
        sig     = _sig_stars(p)
        eff     = _effect_label(d)

        row.update(U=stat, p=p, sig=sig, d=d, effect=eff)
        rows.append(row)
        print(f"  {'Round depth (ord)':<22}  {other_era:<22}  "
              f"{stat:>7.0f}  {p:>7.4f}  {sig:>4}  "
              f"{d:>+6.2f}  {eff:<12}")

    print()
    return pd.DataFrame(rows)


def full_stat_report(match_results: pd.DataFrame,
                     depth_results: pd.DataFrame) -> pd.DataFrame:
    """
    Combine match-level and depth test results into one summary table
    and print a plain-language interpretation.
    """
    combined = pd.concat([match_results, depth_results], ignore_index=True)

    print("  ── Interpretation guide ──────────────────────────────────────")
    print("  sig: *** p<0.01  ** p<0.05  * p<0.10  ns = not significant")
    print("  d  : + = Golden Era higher | effect: small/medium/large\n")

    sig_count = (combined["sig"].isin(["***","**","*"])).sum()
    total     = combined["sig"].notna().sum()
    print(f"  {sig_count} of {total} tests show a significant result "
          f"(α ≤ 0.10) in favour of the Golden Era.\n")

    return combined