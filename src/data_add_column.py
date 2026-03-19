import pandas as pd

# ── Era labels (single source of truth) ──────────────────────────────────────
ERA_PRE    = "Pre-Golden Era"
ERA_GOLDEN = "Golden Era"
ERA_POST   = "Post-Golden Era"

# Ordered list – useful for sorting / CategoricalDtype later
ERA_ORDER = [ERA_PRE, ERA_GOLDEN, ERA_POST]


# ── Core helpers ─────────────────────────────────────────────────────────────

def _season_to_start_year(season: str) -> int | None:
    """
    Parse the start year from a season string.

    Handles the two common formats:
      '2002/03'  →  2002
      '2002-03'  →  2002

    Returns None if the season string cannot be parsed (allows the caller
    to decide how to handle bad data rather than raising silently).
    """
    if not isinstance(season, str):
        return None
    # Accept both '/' and '-' separators
    for sep in ("/", "-"):
        if sep in season:
            try:
                return int(season.split(sep)[0].strip())
            except ValueError:
                return None
    # Fallback: maybe it's just a plain year string e.g. '2002'
    try:
        return int(season.strip())
    except ValueError:
        return None


def _classify_era(season: str,
                  golden_start_year: int,
                  golden_end_year: int) -> str:
    """
    Return the era label for a single season string.
    Unparseable seasons get ERA_PRE as a safe default (will show up in the
    null/coverage report in later steps).
    """
    start_year = _season_to_start_year(season)
    if start_year is None:
        return ERA_PRE          # flag as early; easy to spot in QA

    if start_year < golden_start_year:
        return ERA_PRE
    elif start_year <= golden_end_year:
        return ERA_GOLDEN
    else:
        return ERA_POST


# ── Public API ────────────────────────────────────────────────────────────────

def add_era_column(
    df: pd.DataFrame,
    season_col: str = "season",
    golden_start: str = "1987/88",
    golden_end:   str = "2003/04",
    era_col:      str = "era",
) -> pd.DataFrame:
    """
    Add an 'era' column to *df* and return the enriched DataFrame.

    Parameters
    ----------
    df            : any of the four input DataFrames
    season_col    : name of the column that holds season strings (default 'season')
    golden_start  : first season of the Golden Era  (default '1987/88')
    golden_end    : last  season of the Golden Era  (default '2003/04')
    era_col       : name of the new column          (default 'era')

    Returns
    -------
    A copy of *df* with the new era column appended.
    The column is typed as an ordered Categorical so that
    groupby / sort operations respect the natural era sequence.

    Example
    -------
    >>> cs = add_era_column(country_stats, golden_start="1987/88", golden_end="2003/04")
    >>> cs["era"].value_counts()
    """
    # Parse boundary years once (fail fast if caller passes garbage)
    golden_start_year = _season_to_start_year(golden_start)
    golden_end_year   = _season_to_start_year(golden_end)

    if golden_start_year is None or golden_end_year is None:
        raise ValueError(
            f"Could not parse golden_start='{golden_start}' or "
            f"golden_end='{golden_end}'. "
            "Expected format: 'YYYY/YY'  e.g. '1987/88'."
        )
    if golden_start_year > golden_end_year:
        raise ValueError(
            f"golden_start ({golden_start}) must be before "
            f"golden_end ({golden_end})."
        )

    out = df.copy()

    if season_col not in out.columns:
        raise KeyError(
            f"Column '{season_col}' not found in DataFrame. "
            f"Available columns: {list(out.columns)}"
        )

    out[era_col] = out[season_col].apply(
        lambda s: _classify_era(s, golden_start_year, golden_end_year)
    )

    # Make it an ordered Categorical – preserves sort order in groupby outputs
    out[era_col] = pd.Categorical(out[era_col],
                                  categories=ERA_ORDER,
                                  ordered=True)

    return out


# ── Convenience wrapper: apply to all four DataFrames in one call ─────────────

def add_era_to_all(
    raw_data,
    country_stats,
    country_season_highest,
    country_season_competition_highest,
    golden_start: str = "1987/88",
    golden_end:   str = "2003/04",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Apply add_era_column to all four DataFrames and return enriched copies.

    Usage
    -----
    rd, cs, csh, csch = add_era_to_all(
        raw_data, country_stats,
        country_season_highest, country_season_competition_highest,
        golden_start="1987/88", golden_end="2003/04",
    )
    """
    kwargs = dict(golden_start=golden_start, golden_end=golden_end)

    rd   = add_era_column(raw_data,                          **kwargs)
    cs   = add_era_column(country_stats,                     **kwargs)
    csh  = add_era_column(country_season_highest,            **kwargs)
    csch = add_era_column(country_season_competition_highest,**kwargs)

    _print_era_summary(cs, golden_start, golden_end)

    return rd, cs, csh, csch


# ── QA helper ────────────────────────────────────────────────────────────────

def _print_era_summary(cs: pd.DataFrame,
                       golden_start: str,
                       golden_end: str) -> None:
    """Print a quick sanity-check table after labelling."""
    print(f"\n  Era definition : {golden_start}  →  {golden_end}")
    print(f"  {'Era':<22}  {'Seasons':>7}  {'First':>9}  {'Last':>9}")
    print("  " + "-" * 52)
    for era in ERA_ORDER:
        sub = cs[cs["era"] == era]
        if sub.empty:
            print(f"  {era:<22}  {'0':>7}  {'—':>9}  {'—':>9}")
        else:
            seasons = sorted(sub["season"].unique(),
                             key=lambda s: _season_to_start_year(s) or 0)
            print(f"  {era:<22}  {len(seasons):>7}  "
                  f"{seasons[0]:>9}  {seasons[-1]:>9}")
    print()

"""
Round grouping  (raw_data only)
─────────────────────────────────────────
Maps the granular round strings in raw_data['round'] to five ordered groups:

    'stage'  <  'Round of 16'  <  'Quarter Finals'  <  'Semi Finals'  <  'Final'

'stage' is the catch-all for every round that is not a knockout round
(all qualifying rounds, group stages, play-offs, league phases, etc.)

The result is stored in a new column  'round_group'  typed as an ordered
Categorical so that comparisons like

    df['round_group'] >= 'Quarter Finals'

and groupby / sort operations work correctly out of the box.

NOTE: country_season_highest and country_season_competition_highest already
carry the grouped label in 'highest_round' — no mapping or ordering is
applied to those DataFrames here.
"""

# ── Ordered group labels (single source of truth) ─────────────────────────────
ROUND_GROUPS = ["stage", "Round of 16", "Quarter Finals", "Semi Finals", "Final"]

# ── Exhaustive mapping: raw label → group ─────────────────────────────────────
# Every value found in raw_data['round'] that belongs to 'stage'.
# Knockout rounds map to themselves.
_ROUND_MAP: dict[str, str] = {
    # ── 'stage' bucket ────────────────────────────────────────────────────────
    "Qualifying Round"                  : "stage",
    "1st Qualifying Round"              : "stage",
    "2nd Qualifying Round"              : "stage",
    "3rd Qualifying Round"              : "stage",
    "4th Qualifying or Play-off Round"  : "stage",
    "Qualifying Play-off Round"         : "stage",
    "Preliminary Round"                 : "stage",
    "Round 1"                           : "stage",
    "Round 2"                           : "stage",
    "Round 3"                           : "stage",
    "Round 4"                           : "stage",
    "Group Stage"                       : "stage",
    "1st Group Stage"                   : "stage",
    "2nd Group Stage"                   : "stage",
    "Knockout round play-offs"          : "stage",
    "League Stage"                      : "stage",
    # ── Knockout rounds (map to themselves) ───────────────────────────────────
    "Round of 16"                       : "Round of 16",
    "Quarter Finals"                    : "Quarter Finals",
    "Semi Finals"                       : "Semi Finals",
    "Final"                             : "Final",
}


# ── Core function ─────────────────────────────────────────────────────────────

def add_round_group(
    df: pd.DataFrame,
    round_col:  str = "round",
    output_col: str = "round_group",
) -> pd.DataFrame:
    """
    Add a 'round_group' column to raw_data.

    Parameters
    ----------
    df         : raw_data DataFrame
    round_col  : name of the column that holds the raw round strings
    output_col : name of the new grouped column  (default 'round_group')

    Returns
    -------
    A copy of df with the new column appended as an ordered Categorical:
        'stage' < 'Round of 16' < 'Quarter Finals' < 'Semi Finals' < 'Final'

    Unmapped round strings are set to NaN and reported so you can extend
    _ROUND_MAP without silent data loss.
    """
    if round_col not in df.columns:
        raise KeyError(
            f"Column '{round_col}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    out = df.copy()
    out[output_col] = out[round_col].map(_ROUND_MAP)

    # ── QA: report any unmapped values ───────────────────────────────────────
    unmapped_mask = out[output_col].isna() & out[round_col].notna()
    if unmapped_mask.any():
        unmapped_vals = out.loc[unmapped_mask, round_col].value_counts()
        print(f"\n  [round_group] WARNING – {unmapped_mask.sum()} unmapped rows "
              f"({unmapped_vals.shape[0]} distinct label(s)):")
        for label, count in unmapped_vals.items():
            print(f"    '{label}'  →  {count} rows  "
                  f"(add to _ROUND_MAP in step0_round.py)")
    else:
        print(f"\n  [round_group] All round labels mapped successfully "
              f"({out[output_col].notna().sum()} / {len(out)} rows).")

    # ── Apply ordered Categorical ─────────────────────────────────────────────
    out[output_col] = pd.Categorical(
        out[output_col],
        categories=ROUND_GROUPS,
        ordered=True,
    )

    return out


# ── Quick-look helper ─────────────────────────────────────────────────────────

def round_group_summary(df: pd.DataFrame, round_group_col: str = "round_group") -> None:
    """
    Print a distribution table of the round_group column.
    Useful for a fast sanity check after calling add_round_group().
    """
    if round_group_col not in df.columns:
        print(f"  Column '{round_group_col}' not found — run add_round_group() first.")
        return

    counts = (df[round_group_col]
              .value_counts()
              .reindex(ROUND_GROUPS, fill_value=0))
    total  = counts.sum()

    print(f"\n  {'Round group':<20}  {'Count':>7}  {'%':>6}")
    print("  " + "-" * 38)
    for group, count in counts.items():
        pct = 100 * count / total if total else 0
        print(f"  {group:<20}  {count:>7,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':<20}  {total:>7,}  100.0%\n")
