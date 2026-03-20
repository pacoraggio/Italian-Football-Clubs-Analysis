import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


def italy_by_competition(
    by_comp:      dict[str, pd.DataFrame],
    country_col:  str = "country",
    country_name: str = "Ita",
    drop_empty:   bool = True,
) -> pd.DataFrame:
    """
    Pull one country's row from each competition table in by_comp.

    Parameters
    ----------
    by_comp      : dict output of step2d country_competition_breakdown
    country_col  : country column name
    country_name : country code to extract  (default 'Ita')
    drop_empty   : if True (default), exclude competitions where
                   avg_round_ord is NaN (no valid depth data)

    Returns
    -------
    summary : DataFrame, one row per competition,
              sorted by avg_round_ord descending
    """
    name_up = country_name.strip().upper()
    records = []

    for comp, agg in by_comp.items():
        country_row = agg[agg[country_col].str.upper() == name_up]
        n_countries = len(agg)

        if country_row.empty:
            # Country absent from this competition entirely
            records.append({
                "competition"  : comp,
                "n_seasons"    : 0,
                "avg_round_ord": None,
                "n_finals"     : 0,
                "n_semis"      : 0,
                "finals_rate"  : None,
                "rank"         : None,
                "n_countries"  : n_countries,
                "participated" : False,
            })
        else:
            r           = country_row.iloc[0]
            avg_ord     = r["avg_round_ord"]
            # participated = country was present AND has valid round data
            participated = pd.notna(avg_ord)
            records.append({
                "competition"  : comp,
                "n_seasons"    : int(r["n_seasons"]),
                "avg_round_ord": avg_ord if participated else None,
                "n_finals"     : int(r["n_finals"]),
                "n_semis"      : int(r["n_semis"]),
                "finals_rate"  : (r["n_finals"] / r["n_seasons"]
                                  if r["n_seasons"] > 0 else None),
                "rank"         : r["rank"] if participated else None,
                "n_countries"  : n_countries,
                "participated" : participated,
            })

    summary = (pd.DataFrame(records)
                 .sort_values("avg_round_ord", ascending=False,
                              na_position="last")
                 .reset_index(drop=True))

    # -- Filter out competitions with no valid depth data --------------------
    n_total   = len(summary)
    active    = summary[summary["participated"]]
    n_active  = len(active)
    n_dropped = n_total - n_active

    if drop_empty:
        if n_dropped:
            dropped_names = summary[~summary["participated"]]["competition"].tolist()
            print(f"  Note: {n_dropped} competition(s) excluded "
                  f"(no valid depth data): {dropped_names}")
        summary = active.reset_index(drop=True)

    # -- Print ---------------------------------------------------------------
    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if pd.notna(val) else "  --"

    print(f"\n  {country_name} -- per-competition summary (Golden Era)")
    print(f"  {'Comp':<28}  {'N':>4}  {'AvgRnd':>7}  "
          f"{'Finals':>7}  {'Semis':>6}  {'F-rate':>7}  "
          f"{'Rank':>5}  {'/ N':>5}")
    print("  " + "-" * 75)
    for _, r in summary.iterrows():
        rank_str = f"{r['rank']}" if pd.notna(r["rank"]) else " --"
        print(f"  {r['competition']:<28}  {r['n_seasons']:>4}  "
              f"{_fmt(r['avg_round_ord']):>7}  "
              f"{r['n_finals']:>7}  "
              f"{r['n_semis']:>6}  "
              f"{_fmt(r['finals_rate']):>7}  "
              f"{rank_str:>5}  "
              f"{r['n_countries']:>5}")

    # -- Participation summary -----------------------------------------------
    print(f"\n  Participated in {n_active} / {n_total} competitions "
          f"with valid depth data.")
    if not active.empty:
        best        = active.loc[active["avg_round_ord"].idxmax(), "competition"]
        most_finals = active.loc[active["n_finals"].idxmax(),      "competition"]
        print(f"  Deepest avg round : {best}")
        print(f"  Most Finals       : {most_finals}")

    return summary
 
 
def rivals_per_competition(
    by_comp:      dict[str, pd.DataFrame],
    comp_summary: pd.DataFrame,
    country_col:  str        = "country",
    country_name: str        = "Ita",
    rivals:       list[str] | None = None,
    top_n:        int        = 8,
) -> dict[str, pd.DataFrame]:
    """
    Compare focal country against the field and named rivals, per competition.
 
    Parameters
    ----------
    by_comp      : dict {comp_code -> per-country DataFrame} from step2d
    comp_summary : step3a output (one row per competition, already filtered)
    country_col  : country column name
    country_name : focal country code            (default 'Ita')
    rivals       : explicit list of rival codes  (None = auto top-5)
    top_n        : rows shown in the ranked field table
 
    Returns
    -------
    results : dict {comp_code -> DataFrame with rival comparison rows}
              Each DataFrame has columns:
                country, avg_round_ord, n_finals, finals_rate, rank,
                diff_avg_round, diff_n_finals, diff_finals_rate
              where diff_* = focal_country_value - rival_value
              (positive = focal country ahead)
    """
    name_up  = country_name.strip().upper()
    results  = {}
    valid_comps = comp_summary["competition"].tolist()
 
    print(f"\n  {country_name} vs rivals -- per-competition breakdown")
 
    for comp in valid_comps:
        if comp not in by_comp:
            continue
 
        agg = by_comp[comp].copy()
 
        # -- Focal country row -----------------------------------------------
        focal_rows = agg[agg[country_col].str.upper() == name_up]
        if focal_rows.empty or pd.isna(focal_rows.iloc[0]["avg_round_ord"]):
            print(f"\n  [{comp}]  {country_name} has no valid data -- skipped.")
            continue
        focal = focal_rows.iloc[0]
 
        # -- Auto-select rivals if not specified -----------------------------
        if rivals is None:
            # Top-N other countries by avg_round_ord in this competition
            other = (agg[agg[country_col].str.upper() != name_up]
                     .dropna(subset=["avg_round_ord"])
                     .sort_values("avg_round_ord", ascending=False)
                     .head(top_n))
            rival_codes = other[country_col].tolist()
        else:
            rival_codes = [r for r in rivals
                           if r.upper() != name_up]
 
        # -- Build comparison table ------------------------------------------
        records = []
        for _, r in agg[agg[country_col].isin(rival_codes)].iterrows():
            if pd.isna(r["avg_round_ord"]):
                continue
            records.append({
                country_col       : r[country_col],
                "avg_round_ord"   : r["avg_round_ord"],
                "n_finals"        : int(r["n_finals"]),
                "finals_rate"     : r.get("finals_rate",
                                          r["n_finals"] / r["n_seasons"]
                                          if r["n_seasons"] > 0 else np.nan),
                "rank"            : r["rank"],
                "diff_avg_round"  : focal["avg_round_ord"] - r["avg_round_ord"],
                "diff_n_finals"   : int(focal["n_finals"]) - int(r["n_finals"]),
                "diff_finals_rate": (focal.get("finals_rate",
                                     focal["n_finals"] / focal["n_seasons"])
                                     - r.get("finals_rate",
                                       r["n_finals"] / r["n_seasons"]
                                       if r["n_seasons"] > 0 else np.nan)),
            })
 
        if not records:
            print(f"\n  [{comp}]  No rival data available.")
            continue
 
        comp_df = (pd.DataFrame(records)
                     .sort_values("avg_round_ord", ascending=False)
                     .reset_index(drop=True))
        results[comp] = comp_df
 
        # -- Print -----------------------------------------------------------
        _print_comp_block(comp, focal, comp_df, agg, country_col,
                          country_name, top_n)
 
    return results
 
 
def _print_comp_block(comp, focal, comp_df, agg, country_col,
                      country_name, top_n):
    """Print the three sub-tables for one competition."""
 
    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if pd.notna(val) else "  --"
 
    # -- 1. Italy's headline numbers ----------------------------------------
    f_rate = (focal["n_finals"] / focal["n_seasons"]
              if focal["n_seasons"] > 0 else np.nan)
    print(f"\n  {'='*60}")
    print(f"  [{comp}]  {country_name}:  "
          f"rank #{focal['rank']}  |  "
          f"avg round {_fmt(focal['avg_round_ord'], '.3f')}  |  "
          f"finals {int(focal['n_finals'])}  |  "
          f"finals rate {_fmt(f_rate, '.3f')}")
    print(f"  {'='*60}")
 
    # -- 2. Gap to next-best country ----------------------------------------
    others = (agg[agg[country_col].str.upper() !=
                  str(focal[country_col]).upper()]
              .dropna(subset=["avg_round_ord"])
              .sort_values("avg_round_ord", ascending=False))
 
    if not others.empty:
        runner_up = others.iloc[0]
        gap_round  = focal["avg_round_ord"] - runner_up["avg_round_ord"]
        gap_finals = int(focal["n_finals"]) - int(runner_up["n_finals"])
        print(f"\n  Gap to runner-up ({runner_up[country_col]}):")
        print(f"    avg round  : {_fmt(focal['avg_round_ord'])} vs "
              f"{_fmt(runner_up['avg_round_ord'])}  "
              f"(+{gap_round:.3f} in favour of {country_name})")
        print(f"    finals     : {int(focal['n_finals'])} vs "
              f"{int(runner_up['n_finals'])}  "
              f"({gap_finals:+d})")
 
    # -- 3. Rivals comparison table -----------------------------------------
    print(f"\n  Rivals comparison ({country_name} values minus rival values):")
    print(f"  {'Country':<10}  {'AvgRnd':>8}  {'Finals':>7}  "
          f"{'F-rate':>8}  {'Rank':>5}  "
          f"{'d(AvgRnd)':>10}  {'d(Finals)':>10}  {'d(F-rate)':>10}")
    print("  " + "-" * 80)
 
    # Focal row first
    print(f"  {str(focal[country_col]):<10}  "
          f"{_fmt(focal['avg_round_ord']):>8}  "
          f"{int(focal['n_finals']):>7}  "
          f"{_fmt(f_rate):>8}  "
          f"{focal['rank']:>5}  "
          f"{'(focal)':>10}  {'(focal)':>10}  {'(focal)':>10}")
 
    for _, r in comp_df.iterrows():
        d_sign = lambda v: f"{v:>+10.3f}" if pd.notna(v) else "       --"
        print(f"  {str(r[country_col]):<10}  "
              f"{_fmt(r['avg_round_ord']):>8}  "
              f"{int(r['n_finals']):>7}  "
              f"{_fmt(r['finals_rate']):>8}  "
              f"{r['rank']:>5}  "
              f"{d_sign(r['diff_avg_round'])}  "
              f"{r['diff_n_finals']:>+10d}  "
              f"{d_sign(r['diff_finals_rate'])}")
 
    # -- 4. Full field top-N (for context) ----------------------------------
    top = (agg.dropna(subset=["avg_round_ord"])
              .sort_values("avg_round_ord", ascending=False)
              .head(top_n))
 
    print(f"\n  Full field -- top {top_n} by avg round:")
    print(f"  {'Rk':>3}  {'Country':<10}  {'AvgRnd':>8}  "
          f"{'Finals':>7}  {'Semis':>6}  {'F-rate':>8}")
    print("  " + "-" * 50)
    for _, r in top.iterrows():
        r_frate = (r["n_finals"] / r["n_seasons"]
                   if r["n_seasons"] > 0 else np.nan)
        marker = " <--" if str(r[country_col]).upper() == \
                           str(focal[country_col]).upper() else ""
        print(f"  {r['rank']:>3}  {str(r[country_col]):<10}  "
              f"{_fmt(r['avg_round_ord']):>8}  "
              f"{int(r['n_finals']):>7}  "
              f"{int(r['n_semis']):>6}  "
              f"{_fmt(r_frate):>8}{marker}")
 

def cross_competition_consistency(
    comp_summary: pd.DataFrame,
    results:      dict[str, pd.DataFrame],
    country_name: str = "Ita",
) -> pd.DataFrame:
    """
    Assess how consistently the focal country dominated across competitions.

    Parameters
    ----------
    comp_summary : step3a output (one row per competition)
    results      : step3b output dict {comp -> rivals DataFrame}
    country_name : focal country code  (default 'Ita')

    Returns
    -------
    profile : DataFrame with one row per competition and columns:
                competition, rank, n_countries, avg_round_ord,
                finals_rate, gap_to_runner_up, runner_up
    """
    valid_comps = [c for c in comp_summary["competition"] if c in results]

    if not valid_comps:
        print("  No competitions with both comp_summary and results data.")
        return pd.DataFrame()

    records = []
    for comp in valid_comps:
        row_s = comp_summary[comp_summary["competition"] == comp].iloc[0]
        rivals_df = results[comp]

        # Gap to runner-up = diff_avg_round of the closest rival
        # (rivals_df is already sorted by avg_round_ord desc,
        #  so the first row is the strongest rival)
        if not rivals_df.empty:
            runner_up     = rivals_df.iloc[0]["country"]
            gap_to_top    = rivals_df.iloc[0]["diff_avg_round"]
            gap_finals    = rivals_df.iloc[0]["diff_n_finals"]
        else:
            runner_up  = "--"
            gap_to_top = np.nan
            gap_finals = np.nan

        records.append({
            "competition"     : comp,
            "rank"            : row_s["rank"],
            "n_countries"     : row_s["n_countries"],
            "avg_round_ord"   : row_s["avg_round_ord"],
            "finals_rate"     : row_s["finals_rate"],
            "n_finals"        : row_s["n_finals"],
            "runner_up"       : runner_up,
            "gap_avg_round"   : gap_to_top,
            "gap_n_finals"    : gap_finals,
        })

    profile = pd.DataFrame(records)

    # -- 1. Rank consistency ------------------------------------------------
    ranks    = profile["rank"].dropna()
    all_top1 = (ranks == 1).all()
    rank_var = ranks.var(ddof=0)   # population variance across competitions

    # -- 2. Gap consistency -------------------------------------------------
    gaps     = profile["gap_avg_round"].dropna()
    gap_mean = gaps.mean()
    gap_std  = gaps.std(ddof=0)
    gap_cv   = gap_std / gap_mean if gap_mean > 0 else np.nan  # coeff of variation

    # -- 3. Metric profile --------------------------------------------------
    avg_depth_all  = profile["avg_round_ord"].mean()
    avg_frate_all  = profile["finals_rate"].mean()

    def _fmt(val, fmt=".3f"):
        return f"{val:{fmt}}" if pd.notna(val) else "  --"

    # -- Print --------------------------------------------------------------
    print(f"\n  {country_name} -- cross-competition consistency (Golden Era)")

    print(f"\n  1. Rank per competition")
    print(f"  {'Comp':<8}  {'Rank':>5}  {'/ N':>5}  "
          f"{'AvgRnd':>8}  {'Finals':>7}  {'F-rate':>8}  "
          f"{'Runner-up':<10}  {'Gap(Rnd)':>9}  {'Gap(Fin)':>9}")
    print("  " + "-" * 78)
    for _, r in profile.iterrows():
        print(f"  {r['competition']:<8}  "
              f"{_fmt(r['rank'], '.0f'):>5}  "
              f"{int(r['n_countries']):>5}  "
              f"{_fmt(r['avg_round_ord']):>8}  "
              f"{int(r['n_finals']):>7}  "
              f"{_fmt(r['finals_rate']):>8}  "
              f"{str(r['runner_up']):<10}  "
              f"{_fmt(r['gap_avg_round'], '+.3f'):>9}  "
              f"{_fmt(r['gap_n_finals'], '+.0f'):>9}")

    print(f"\n  2. Rank consistency")
    print(f"     All competitions ranked #1 : {'Yes' if all_top1 else 'No'}")
    print(f"     Rank variance              : {rank_var:.3f}  "
          f"(0 = identical rank in every competition)")

    print(f"\n  3. Gap to runner-up consistency")
    print(f"     Mean gap (avg_round_ord)   : {_fmt(gap_mean)}")
    print(f"     Std of gaps                : {_fmt(gap_std)}")
    print(f"     Coefficient of variation   : {_fmt(gap_cv)}  "
          f"(lower = more uniform dominance)")

    print(f"\n  4. Overall metric profile")
    print(f"     Avg depth across comps     : {_fmt(avg_depth_all)}")
    print(f"     Avg finals rate across comps: {_fmt(avg_frate_all)}")

    # -- Verdict ------------------------------------------------------------
    print(f"\n  5. Consistency verdict")
    if all_top1 and gap_cv < 0.25:
        verdict = "Uniform dominance: ranked #1 in all competitions with a consistent lead."
    elif all_top1 and gap_cv >= 0.25:
        verdict = ("Ranked #1 in all competitions but the margin varies — "
                   "dominance stronger in some than others.")
    elif not all_top1:
        best_comp = profile.loc[profile["avg_round_ord"].idxmax(), "competition"]
        verdict   = (f"Not #1 in all competitions — "
                     f"dominance most concentrated in {best_comp}.")
    else:
        verdict = "Mixed picture — inspect per-competition gaps above."
    print(f"     {verdict}")

    return profile



ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]

MIN_N = 3   # minimum seasons per group to run a test


# -- Helpers ------------------------------------------------------------------

def _ordinal(series: pd.Series) -> pd.Series:
    """Convert highest_round to ordinal (0-4). Unknown/null -> NaN."""
    known = set(ROUND_GROUPS)
    clean = series.where(series.isin(known), other=np.nan)
    cat   = pd.Categorical(clean, categories=ROUND_GROUPS, ordered=True)
    return pd.Series(cat.codes, index=series.index, dtype="float").replace(-1, np.nan)

def _cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    var_a  = np.var(a, ddof=1) if len(a) > 1 else 0.0
    var_b  = np.var(b, ddof=1) if len(b) > 1 else 0.0
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


# -- Main function ------------------------------------------------------------

def test_per_competition(
    ge_csch:      pd.DataFrame,
    valid_comps:  list[str] | None = None,
    country_col:  str = "country",
    comp_col:     str = "competition",
    round_col:    str = "highest_round",
    country_name: str = "Ita",
) -> pd.DataFrame:
    """
    Run Mann-Whitney U + Cohen's d for each competition separately.

    Parameters
    ----------
    ge_csch      : Golden Era country_season_competition_highest
    valid_comps  : competitions to test; None = all with sufficient data
    country_col  : country column name
    comp_col     : competition column name
    round_col    : highest_round column name
    country_name : focal country code  (default 'Ita')

    Returns
    -------
    results : DataFrame with one row per competition, columns:
                competition, n_focal, n_rest, U, p, sig, d, effect
    """
    df = ge_csch.copy()
    df["round_ord"] = _ordinal(df[round_col])

    name_up = country_name.strip().upper()

    # Determine which competitions to test
    all_comps = sorted(df[comp_col].dropna().unique())
    if valid_comps is not None:
        all_comps = [c for c in all_comps if c in valid_comps]

    rows = []
    print(f"\n  Mann-Whitney U per competition  ({country_name} vs rest)")
    print(f"  {'Comp':<28}  {'n_focal':>7}  {'n_rest':>7}  "
          f"{'U':>8}  {'p':>8}  {'sig':>4}  {'d':>6}  effect")
    print("  " + "-" * 82)

    for comp in all_comps:
        sub        = df[df[comp_col] == comp]
        focal_vals = (sub[sub[country_col].str.upper() == name_up]
                      ["round_ord"].dropna().values)
        rest_vals  = (sub[sub[country_col].str.upper() != name_up]
                      ["round_ord"].dropna().values)

        row = {"competition": comp,
               "n_focal": len(focal_vals),
               "n_rest":  len(rest_vals)}

        # Skip if either group has no valid round data at all
        if len(focal_vals) == 0:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                       effect="no focal data")
            print(f"  {comp:<28}  {'--':>7}  {len(rest_vals):>7}  "
                  f"  SKIP (no valid round data for {country_name})")
            rows.append(row)
            continue

        # Skip if too few observations
        if len(focal_vals) < MIN_N or len(rest_vals) < MIN_N:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                       effect=f"too few (n={len(focal_vals)}/{len(rest_vals)})")
            print(f"  {comp:<28}  {len(focal_vals):>7}  {len(rest_vals):>7}  "
                  f"  SKIP (n too small)")
            rows.append(row)
            continue

        stat, p = mannwhitneyu(focal_vals, rest_vals, alternative="greater")
        d        = _cohen_d(focal_vals, rest_vals)
        sig      = _sig_stars(p)
        effect   = _effect_label(d)

        row.update(U=stat, p=p, sig=sig, d=d, effect=effect)
        rows.append(row)

        print(f"  {comp:<28}  {len(focal_vals):>7}  {len(rest_vals):>7}  "
              f"{stat:>8.0f}  {p:>8.4f}  {sig:>4}  {d:>+6.2f}  {effect}")

    results = pd.DataFrame(rows)

    # -- Summary ----------------------------------------------------------
    tested   = results[results["sig"] != "skip"]
    sig_any  = tested[tested["sig"].isin(["***","**","*"])]
    n_tested = len(tested)
    n_sig    = len(sig_any)

    print(f"\n  {n_sig} / {n_tested} tested competitions show significant "
          f"superiority (alpha <= 0.10).")
    if not sig_any.empty:
        print(f"  Significant in: {', '.join(sig_any['competition'].tolist())}")
    not_sig = tested[~tested["sig"].isin(["***","**","*"])]
    if not not_sig.empty:
        print(f"  Not significant in: {', '.join(not_sig['competition'].tolist())}")

    return results

def test_per_competition_vs_top(
    ge_csch:      pd.DataFrame,
    by_comp:      dict[str, pd.DataFrame],
    valid_comps:  list[str] | None = None,
    country_col:  str  = "country",
    comp_col:     str  = "competition",
    round_col:    str  = "highest_round",
    country_name: str  = "Ita",
    top_n:        int  = 5,
) -> pd.DataFrame:
    """
    Same test as test_per_competition() but compares the focal country
    against only the top-N rivals per competition (by avg_round_ord),
    rather than the full pooled field.
 
    This is the more conservative version: passing here means Italy was
    significantly better even than the strongest competitors, not just
    the average country.
 
    Parameters
    ----------
    ge_csch      : Golden Era country_season_competition_highest
    by_comp      : dict from step2d (used to identify top-N per comp)
    valid_comps  : competitions to test; None = all with sufficient data
    country_col  : country column name
    comp_col     : competition column name
    round_col    : highest_round column name
    country_name : focal country code  (default 'Ita')
    top_n        : number of top rivals to test against  (default 5)
 
    Returns
    -------
    results : DataFrame with one row per competition, same columns as
              test_per_competition() plus 'rivals_used'
    """
    df = ge_csch.copy()
    df["round_ord"] = _ordinal(df[round_col])
 
    name_up     = country_name.strip().upper()
    all_comps   = sorted(df[comp_col].dropna().unique())
    if valid_comps is not None:
        all_comps = [c for c in all_comps if c in valid_comps]
 
    rows = []
    print(f"\n  Mann-Whitney U per competition  "
          f"({country_name} vs top-{top_n} rivals only)")
    print(f"  {'Comp':<28}  {'n_focal':>7}  {'n_rest':>7}  "
          f"{'U':>8}  {'p':>8}  {'sig':>4}  {'d':>6}  "
          f"{'effect':<12}  rivals")
    print("  " + "-" * 100)
 
    for comp in all_comps:
        if comp not in by_comp:
            continue
 
        sub = df[df[comp_col] == comp]
 
        # Identify top-N rivals for this competition from the aggregated table
        agg = by_comp[comp]
        top_rivals = (agg[agg[country_col].str.upper() != name_up]
                      .dropna(subset=["avg_round_ord"])
                      .sort_values("avg_round_ord", ascending=False)
                      .head(top_n)[country_col]
                      .tolist())
 
        focal_vals = (sub[sub[country_col].str.upper() == name_up]
                      ["round_ord"].dropna().values)
        rest_vals  = (sub[sub[country_col].isin(top_rivals)]
                      ["round_ord"].dropna().values)
 
        rivals_str = ", ".join(top_rivals)
        row = {"competition": comp,
               "n_focal":     len(focal_vals),
               "n_rest":      len(rest_vals),
               "rivals_used": rivals_str}
 
        if len(focal_vals) == 0:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                       effect="no focal data")
            print(f"  {comp:<28}  {'--':>7}  {len(rest_vals):>7}  "
                  f"  SKIP (no focal data)")
            rows.append(row)
            continue
 
        if len(focal_vals) < MIN_N or len(rest_vals) < MIN_N:
            row.update(U=np.nan, p=np.nan, sig="skip", d=np.nan,
                       effect=f"too few (n={len(focal_vals)}/{len(rest_vals)})")
            print(f"  {comp:<28}  {len(focal_vals):>7}  {len(rest_vals):>7}  "
                  f"  SKIP (n too small)")
            rows.append(row)
            continue
 
        stat, p = mannwhitneyu(focal_vals, rest_vals, alternative="greater")
        d        = _cohen_d(focal_vals, rest_vals)
        sig      = _sig_stars(p)
        effect   = _effect_label(d)
 
        row.update(U=stat, p=p, sig=sig, d=d, effect=effect)
        rows.append(row)
 
        print(f"  {comp:<28}  {len(focal_vals):>7}  {len(rest_vals):>7}  "
              f"{stat:>8.0f}  {p:>8.4f}  {sig:>4}  {d:>+6.2f}  "
              f"{effect:<12}  {rivals_str}")
 
    results = pd.DataFrame(rows)
 
    # -- Summary --------------------------------------------------------------
    tested  = results[results["sig"] != "skip"]
    sig_any = tested[tested["sig"].isin(["***","**","*"])]
    n_tested, n_sig = len(tested), len(sig_any)
 
    print(f"\n  {n_sig} / {n_tested} tested competitions show significant "
          f"superiority vs top-{top_n} rivals (alpha <= 0.10).")
    if not sig_any.empty:
        print(f"  Significant in: {', '.join(sig_any['competition'].tolist())}")
    not_sig = tested[~tested["sig"].isin(["***","**","*"])]
    if not not_sig.empty:
        print(f"  Not significant in: {', '.join(not_sig['competition'].tolist())}")
 
    return results
 