"""
UEFA European Cup Match Scraper
================================
Scrapes match results from kassiesa.net for all three UEFA club competitions:
  - CL  : Champions League / European Cup (1955–present)
  - EL  : UEFA Cup / Europa League / Inter-Cities Fairs Cup
  - CW  : Cup Winners' Cup (1961–1999)

Output CSV columns:
  season, competition, round, date,
  team_a, country_a, team_b, country_b,
  score_90min, score_final, extra_time, penalties

Usage:
  pip install requests beautifulsoup4
  python uefa_scraper.py                        # 1980–present (default)
  python uefa_scraper.py --start 1990 --end 2005
  python uefa_scraper.py --start 1980 --output my_file.csv

Notes:
  - kassiesa.net is the data source. Please scrape politely (delay between requests).
  - The site uses different URL "method" prefixes for different time periods (see METHOD_MAP).
  - Scores are shown as "X-Y" for 90-min result.  When extra time was played
    the 90-min score may differ from the final; kassiesa encodes this inline
    (e.g. "1-1 aet 3-2p" or "1-1 (2-0)"). The parser extracts both where possible.
  - No match date is available on kassiesa.net (only season + round).
    A "date" column is included but will be empty; enrich later from another source
    if needed (e.g. EuroCupsHistory, Wikidata).
"""

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, fields
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# URL building
# ---------------------------------------------------------------------------

# kassiesa.net changed its URL scheme several times.
# Each entry is (first_season_year, last_season_year, method_prefix)
# where season year = the year the season ENDS (e.g. 1980 = 1979/80).
METHOD_MAP = [
    (1956, 1998, "method1"),   # old coefficient rules
    (1999, 2008, "method2"),   # 5-year, points/matches
    (2009, 2017, "method3"),   # 5-year, individual + 20 % country
    (2018, 2099, "method5"),   # current (individual only, min 20 %)
]


def get_method(season_end_year: int) -> str:
    for start, end, method in METHOD_MAP:
        if start <= season_end_year <= end:
            return method
    return "method5"


def build_match_url(season_end_year: int) -> str:
    method = get_method(season_end_year)
    return f"https://kassiesa.net/uefa/data/{method}/match{season_end_year}.html"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Match:
    season: str          # e.g. "1979/80"
    competition: str     # CL / EL / CW / ECL
    round: str           # F, SF, QF, R16, GS, R1, Q1 …
    date: str            # empty – not available on this source
    team_a: str
    country_a: str
    team_b: str
    country_b: str
    score_90min: str     # e.g. "2-1"
    score_final: str     # may differ if AET/penalties
    extra_time: bool     # True if match went to extra time
    penalties: bool      # True if decided on penalties


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

# Typical raw score strings from kassiesa.net:
#   "2-1"            → normal win in 90 min
#   "1-1 (3-1)"      → drew 1-1 in 90 min, 3-1 after extra time
#   "0-0 (3-2p)"     → 0-0 in 90 min / ET, 3-2 on penalties
#   "1-1 aet"        → went to extra time (no separate final score shown)
#   "w/o"            → walkover (no score)
SCORE_RE = re.compile(
    r"""
    (?P<s90>\d+-\d+)            # 90-min score (always present if played)
    (?:\s+                      # optional suffix
      (?:
        \((?P<sfinal>\d+-\d+)   # final score in parens  e.g. (3-1) or (3-2p)
        (?P<pen>p)?\)           # optional 'p' = penalties
      |
        (?P<aet>aet)            # just 'aet' with no separate final
      )
    )?
    """,
    re.VERBOSE,
)


def parse_score(raw: str):
    """
    Returns (score_90min, score_final, extra_time, penalties).
    """
    raw = raw.strip()
    if not raw or raw.lower() in ("w/o", "-", ""):
        return raw, raw, False, False

    m = SCORE_RE.search(raw)
    if not m:
        return raw, raw, False, False

    s90 = m.group("s90") or ""
    sfinal = m.group("sfinal")
    pen = bool(m.group("pen"))
    aet = bool(m.group("aet"))

    extra_time = bool(sfinal) or aet or pen
    penalties = pen

    if sfinal:
        score_final = sfinal
    else:
        score_final = s90  # best we have

    return s90, score_final, extra_time, penalties


# ---------------------------------------------------------------------------
# Page scraper
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; UEFA-research-scraper/1.0; "
        "+https://github.com/your-repo)"
    ),
    "Accept-Language": "en-GB,en;q=0.9",
}

# Competition labels the site uses
COMP_LABELS = {"CL", "EL", "CW", "ECL"}


def season_label(year: int) -> str:
    """2024 → '2023/24'"""
    return f"{year-1}/{str(year)[-2:]}"


def scrape_season(session: requests.Session, year: int, delay: float = 2.0):
    """
    Fetch and parse a single season's match page from kassiesa.net.
    Returns a list of Match objects.
    """
    url = build_match_url(year)
    season = season_label(year)
    matches = []

    try:
        resp = session.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"  [WARN] Could not fetch {url}: {e}", file=sys.stderr)
        return matches

    soup = BeautifulSoup(resp.text, "html.parser")

    # The match data is presented as a <pre> or <table> block depending on era.
    # Older pages (pre-2009) use a plain <pre> text block.
    # Newer pages use an HTML table.

    # --- Try HTML table first ---
    table = soup.find("table")
    if table:
        matches.extend(_parse_table(table, season))
    else:
        # --- Fall back to <pre> text block ---
        pre = soup.find("pre")
        if pre:
            matches.extend(_parse_pre(pre.get_text(), season))
        else:
            print(f"  [WARN] No parseable content found for {season} at {url}",
                  file=sys.stderr)

    time.sleep(delay)
    return matches


# --- Table parser (modern pages) ---

def _parse_table(table, season: str):
    """Parse a <table> element into Match objects."""
    matches = []
    rows = table.find_all("tr")

    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if not cells or cells[0] in ("cup", "Cup", "competition"):
            continue  # header row

        # Expected columns (may vary slightly by era):
        # [competition, round, home_team, home_country, score, away_team, away_country]
        # Some pages also have a date column at position 0.

        # Detect if first column looks like a date (dd/mm/yy or dd-mm-yy)
        date_re = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
        offset = 0
        match_date = ""
        if date_re.match(cells[0]):
            match_date = cells[0]
            offset = 1

        if len(cells) < offset + 6:
            continue

        try:
            competition = cells[offset + 0].upper()
            if competition not in COMP_LABELS:
                continue
            round_ = cells[offset + 1]
            team_a = cells[offset + 2]
            country_a = cells[offset + 3]
            raw_score = cells[offset + 4]
            team_b = cells[offset + 5]
            country_b = cells[offset + 6] if len(cells) > offset + 6 else ""

            s90, sfinal, et, pen = parse_score(raw_score)

            matches.append(Match(
                season=season,
                competition=competition,
                round=round_,
                date=match_date,
                team_a=team_a,
                country_a=country_a,
                team_b=team_b,
                country_b=country_b,
                score_90min=s90,
                score_final=sfinal,
                extra_time=et,
                penalties=pen,
            ))
        except (IndexError, ValueError):
            continue

    return matches


# --- Pre-text parser (legacy pages) ---

# Legacy line format (space-separated, fixed-width-ish):
# CL  F   Ajax          Ned  1-0   Juventus    Ita
LINE_RE = re.compile(
    r"""
    ^(?P<cup>CL|EL|CW|ECL)\s+         # competition
    (?P<round>\S+)\s+                  # round code
    (?P<team_a>.+?)\s{2,}             # home team (2+ spaces as delimiter)
    (?P<ctry_a>[A-Z][a-z]{2,3})\s+   # 3-letter country code
    (?P<score>\S+(?:\s+\S+)?)\s+      # score (may have suffix like 'aet')
    (?P<team_b>.+?)\s{2,}             # away team
    (?P<ctry_b>[A-Z][a-z]{2,3})\s*$  # away country
    """,
    re.VERBOSE,
)


def _parse_pre(text: str, season: str):
    matches = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = LINE_RE.match(line)
        if not m:
            continue
        s90, sfinal, et, pen = parse_score(m.group("score"))
        matches.append(Match(
            season=season,
            competition=m.group("cup").upper(),
            round=m.group("round"),
            date="",
            team_a=m.group("team_a").strip(),
            country_a=m.group("ctry_a"),
            team_b=m.group("team_b").strip(),
            country_b=m.group("ctry_b"),
            score_90min=s90,
            score_final=sfinal,
            extra_time=et,
            penalties=pen,
        ))
    return matches


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CSV_COLUMNS = [f.name for f in fields(Match)]


def run(start: int, end: int, output: str, delay: float):
    session = requests.Session()
    all_matches = []
    years = list(range(start, end + 1))

    print(f"Scraping {len(years)} seasons ({start-1}/{str(start)[-2:]} "
          f"→ {end-1}/{str(end)[-2:]}) …")

    for year in years:
        label = season_label(year)
        print(f"  Fetching {label} …", end=" ", flush=True)
        matches = scrape_season(session, year, delay=delay)
        print(f"{len(matches)} matches found.")
        all_matches.extend(matches)

    print(f"\nTotal matches collected: {len(all_matches)}")
    print(f"Writing to {output} …")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for m in all_matches:
            writer.writerow({
                "season": m.season,
                "competition": m.competition,
                "round": m.round,
                "date": m.date,
                "team_a": m.team_a,
                "country_a": m.country_a,
                "team_b": m.team_b,
                "country_b": m.country_b,
                "score_90min": m.score_90min,
                "score_final": m.score_final,
                "extra_time": m.extra_time,
                "penalties": m.penalties,
            })

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape UEFA club competition results from kassiesa.net"
    )
    parser.add_argument(
        "--start",
        type=int,
        default=1981,
        help="First season END year to scrape (default 1981 = season 1980/81). "
             "Use 1981 for the 1980/81 season.",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=2025,
        help="Last season END year to scrape (default 2025 = season 2024/25).",
    )
    parser.add_argument(
        "--output",
        default="uefa_results.csv",
        help="Output CSV file path (default: uefa_results.csv)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2.0,
        help="Seconds to wait between requests (default 2.0 – please be polite!)",
    )
    args = parser.parse_args()

    if args.start > args.end:
        print("Error: --start must be <= --end", file=sys.stderr)
        sys.exit(1)

    run(args.start, args.end, args.output, args.delay)