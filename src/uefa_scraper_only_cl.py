"""
UEFA European Cup Match Scraper — v3
=====================================
Scrapes match results from kassiesa.net for all UEFA club competitions:
  - CL : Champions League / European Cup (1955–present)
  - EL : UEFA Cup / Europa League
  - CW : Cup Winners' Cup (1961–1999)
  - CO : Conference League (2021–present)

Real table structure on kassiesa.net (discovered via --diagnose):
  - Competition name rows:  ['CHAMPIONS CUP'] or ['UEFA CUP'] etc.
  - Round name rows:        ['Qualifying Round'], ['Round 1'], ['Final'] etc.
  - Match data rows:        [team_a, country_a, team_b, country_b, score_leg1, score_leg2]
  - Empty rows:             [''] — separators, ignored

Output CSV columns:
  season, competition, round,
  team_a, country_a, team_b, country_b,
  score_leg1, score_leg2, score_aggregate,
  extra_time, penalties

Usage:
  pip install requests beautifulsoup4
  python uefa_scraper.py --start 1981 --end 2025
  python uefa_scraper.py --start 1981 --end 1983 --diagnose
"""

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, fields

import requests
from bs4 import BeautifulSoup

# ---------------------------------------------------------------------------
# URL resolution — try all method folders automatically
# ---------------------------------------------------------------------------

METHOD_PREFERENCE = {
    range(1956, 1999): "method1",   # based on diagnose output
    range(1999, 2009): "method1",
    range(2009, 2018): "method4",
    range(2018, 2100): "method5",
}

ALL_METHODS = ["method1", "method5", "method4", "method3", "method2", "method0"]


def build_urls(year: int):
    primary = "method1"
    for r, m in METHOD_PREFERENCE.items():
        if year in r:
            primary = m
            break
    others = [m for m in ALL_METHODS if m != primary]
    return [
        f"https://kassiesa.net/uefa/data/{m}/match{year}.html"
        for m in [primary] + others
    ]


def season_label(year: int) -> str:
    return f"{year - 1}/{str(year)[-2:]}"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Match:
    season: str
    competition: str    # e.g. "CHAMPIONS CUP", "UEFA CUP", "CUP WINNERS CUP"
    round: str          # e.g. "Qualifying Round", "Round 1", "Final"
    team_a: str
    country_a: str
    team_b: str
    country_b: str
    score_leg1: str     # first leg (or only score for single-leg games)
    score_leg2: str     # second leg (empty for finals / single-leg)
    score_aggregate: str  # computed sum, e.g. "3-2"
    extra_time: str     # "yes" / "no"
    penalties: str      # "yes" / "no"


CSV_COLUMNS = [f.name for f in fields(Match)]

# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

SCORE_RE = re.compile(r"(\d+)-(\d+)")

# Patterns like "1-0 aet", "1-1 (3-2p)", "0-0 (4-3p)"
AET_RE = re.compile(r"aet", re.IGNORECASE)
PEN_RE = re.compile(r"\((\d+)-(\d+)p\)")


def parse_score_cell(raw: str):
    """
    Returns (clean_score, extra_time, penalties).
    clean_score is the final result (after pens if applicable).
    """
    raw = raw.strip()
    extra_time = "no"
    penalties = "no"

    pen_match = PEN_RE.search(raw)
    if pen_match:
        extra_time = "yes"
        penalties = "yes"
        clean = f"{pen_match.group(1)}-{pen_match.group(2)}"
        return clean, extra_time, penalties

    if AET_RE.search(raw):
        extra_time = "yes"

    # Extract plain score
    m = SCORE_RE.search(raw)
    if m:
        clean = f"{m.group(1)}-{m.group(2)}"
    else:
        clean = raw  # w/o, ?, etc.

    return clean, extra_time, penalties


def compute_aggregate(leg1: str, leg2: str) -> str:
    """Add two scores like '3-0' + '1-2' → '4-2'."""
    if not leg2:
        return leg1
    m1 = SCORE_RE.match(leg1)
    m2 = SCORE_RE.match(leg2)
    if m1 and m2:
        a = int(m1.group(1)) + int(m2.group(1))
        b = int(m1.group(2)) + int(m2.group(2))
        return f"{a}-{b}"
    return ""


# ---------------------------------------------------------------------------
# Competition name normalisation
# ---------------------------------------------------------------------------

COMP_NAME_MAP = {
    "CHAMPIONS CUP":        "CL",
    "CHAMPION CLUBS CUP":   "CL",
    "CHAMPIONS LEAGUE":     "CL",
    "UEFA CUP":             "EL",
    "UEFA EUROPA LEAGUE":   "EL",
    "EUROPA LEAGUE":        "EL",
    "CUP WINNERS CUP":      "CW",
    "UEFA CUP WINNERS CUP": "CW",
    "CONFERENCE LEAGUE":    "CO",
    "UEFA CONFERENCE LEAGUE": "CO",
}

# Words that indicate a row is a competition header
COMP_KEYWORDS = ["CUP", "LEAGUE", "CHAMPIONS", "UEFA", "EUROPA", "CONFERENCE"]

# Words that indicate a row is a round header
ROUND_KEYWORDS = [
    "ROUND", "QUALIFYING", "FINAL", "SEMI", "QUARTER", "GROUP",
    "STAGE", "PHASE", "PLAY", "PRELIMINARY", "FIRST", "SECOND",
    "THIRD", "FOURTH",
]

COUNTRY_CODES = {
    # A sample — the parser uses length heuristic, not this lookup
    "Eng", "Fra", "Ger", "Esp", "Ita", "Ned", "Por", "Bel", "Sco",
    "Hun", "Mlt", "Nor", "Rom", "Gdr", "Isl", "Nir", "Wal", "Irl",
    "Tur", "Gre", "Sui", "Swe", "Den", "Pol", "Tch", "Bul", "Yug",
    "Aut", "Fin", "Lux", "Cyp", "Alb",
}


def normalise_comp(raw: str) -> str:
    key = raw.strip().upper()
    return COMP_NAME_MAP.get(key, raw.strip().title())


def is_comp_row(cells: list) -> bool:
    """Single-cell row whose text contains competition keywords."""
    if len(cells) != 1:
        return False
    text = cells[0].upper()
    return any(kw in text for kw in COMP_KEYWORDS)


def is_round_row(cells: list) -> bool:
    """Single-cell row whose text contains round keywords."""
    if len(cells) != 1:
        return False
    text = cells[0].upper()
    return any(kw in text for kw in ROUND_KEYWORDS)


def is_match_row(cells: list) -> bool:
    """
    A match row has 5 or 6 cells:
    [team_a, country_a, team_b, country_b, score_leg1(, score_leg2)]
    We detect it by checking that at least one score-like cell exists.
    """
    if len(cells) < 5:
        return False
    # At least one cell should look like a score
    for c in cells:
        if SCORE_RE.search(c) or c.strip().lower() in ("w/o", "?"):
            return True
    return False


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_season(soup: BeautifulSoup, season: str) -> list:
    matches = []
    table = soup.find("table")
    if not table:
        return matches

    current_comp = "UNKNOWN"
    current_round = "UNKNOWN"

    for row in table.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]

        # Skip empty rows
        if not cells or all(c == "" for c in cells):
            continue

        # Competition header row
        if is_comp_row(cells):
            current_comp = normalise_comp(cells[0])
            continue

        # Round header row
        if is_round_row(cells):
            current_round = cells[0].strip()
            continue

        # Match data row
        if is_match_row(cells):
            # Layout: team_a, ctry_a, team_b, ctry_b, score1[, score2]
            # Sometimes there may be an extra leading cell (e.g. match number)
            # Find the score cell(s) — they are at known positions from the end
            # Scores are always the last 1 or 2 cells
            # Countries are 2-4 char strings just before the scores

            # Detect how many score columns there are at the end
            score_cells = []
            non_score_cells = list(cells)
            while non_score_cells:
                last = non_score_cells[-1].strip()
                if SCORE_RE.search(last) or last.lower() in ("w/o", "?", ""):
                    score_cells.insert(0, non_score_cells.pop())
                else:
                    break

            if not score_cells:
                continue

            # non_score_cells should now be [maybe_extra, team_a, ctry_a, team_b, ctry_b]
            # We need exactly 4 at the end: team_a, ctry_a, team_b, ctry_b
            if len(non_score_cells) < 4:
                continue

            ctry_b  = non_score_cells[-1]
            team_b  = non_score_cells[-2]
            ctry_a  = non_score_cells[-3]
            team_a  = non_score_cells[-4]

            leg1_raw = score_cells[0] if len(score_cells) >= 1 else ""
            leg2_raw = score_cells[1] if len(score_cells) >= 2 else ""

            leg1, et1, pen1 = parse_score_cell(leg1_raw)
            leg2, et2, pen2 = parse_score_cell(leg2_raw) if leg2_raw else ("", "no", "no")

            extra_time = "yes" if et1 == "yes" or et2 == "yes" else "no"
            penalties  = "yes" if pen1 == "yes" or pen2 == "yes" else "no"
            aggregate  = compute_aggregate(leg1, leg2)

            matches.append(Match(
                season=season,
                competition=current_comp,
                round=current_round,
                team_a=team_a,
                country_a=ctry_a,
                team_b=team_b,
                country_b=ctry_b,
                score_leg1=leg1,
                score_leg2=leg2,
                score_aggregate=aggregate,
                extra_time=extra_time,
                penalties=penalties,
            ))

    return matches


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
}


def fetch(session: requests.Session, year: int, verbose=False):
    for url in build_urls(year):
        try:
            r = session.get(url, headers=HEADERS, timeout=20)
            if r.status_code == 200:
                if verbose:
                    print(f"    ✓ {url}")
                # Fix encoding issues (site serves latin-1 as utf-8 sometimes)
                r.encoding = r.apparent_encoding
                return url, r.text
            if verbose:
                print(f"    ✗ HTTP {r.status_code}: {url}")
        except requests.RequestException as e:
            if verbose:
                print(f"    ✗ Error ({url}): {e}")
    return None, None


# ---------------------------------------------------------------------------
# Diagnose mode
# ---------------------------------------------------------------------------

def diagnose(session: requests.Session, year: int, delay: float):
    label = season_label(year)
    print(f"\n{'='*65}")
    print(f"SEASON {label}  (year={year})")
    print("="*65)

    url, html = fetch(session, year, verbose=True)
    if html is None:
        print("  → FAILED: no URL worked.")
        return

    soup = BeautifulSoup(html, "html.parser")
    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        print(f"\n  <table> — first 10 rows:")
        for row in rows[:10]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            print(f"    {cells}")
    else:
        print("  No <table> found.")

    matches = parse_season(soup, label)
    print(f"\n  → Parsed {len(matches)} matches.")
    if matches:
        print(f"  First 3:")
        for m in matches[:3]:
            print(f"    {m}")
    else:
        print("  *** Still 0 matches — paste this output for further debugging ***")

    time.sleep(delay)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run(start: int, end: int, output: str, delay: float, diagnose_mode: bool):
    session = requests.Session()
    years = list(range(start, end + 1))

    if diagnose_mode:
        print("DIAGNOSE MODE — first 3 seasons only.\n")
        for year in years[:3]:
            diagnose(session, year, delay)
        return

    all_matches = []
    print(f"Scraping {len(years)} seasons "
          f"({season_label(start)} → {season_label(end)}) …\n")

    for year in years:
        label = season_label(year)
        print(f"  {label} … ", end="", flush=True)
        _, html = fetch(session, year)
        if html is None:
            print("FAILED (skipped)")
            continue
        soup = BeautifulSoup(html, "html.parser")
        matches = parse_season(soup, label)
        print(f"{len(matches)} matches")
        all_matches.extend(matches)
        time.sleep(delay)

    print(f"\nTotal: {len(all_matches)} matches")
    print(f"Writing → {output}")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for m in all_matches:
            writer.writerow({
                "season": m.season, "competition": m.competition,
                "round": m.round,
                "team_a": m.team_a, "country_a": m.country_a,
                "team_b": m.team_b, "country_b": m.country_b,
                "score_leg1": m.score_leg1, "score_leg2": m.score_leg2,
                "score_aggregate": m.score_aggregate,
                "extra_time": m.extra_time, "penalties": m.penalties,
            })

    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape UEFA club competition results from kassiesa.net"
    )
    parser.add_argument("--start",   type=int, default=1981)
    parser.add_argument("--end",     type=int, default=2025)
    parser.add_argument("--output",  default="uefa_results.csv")
    parser.add_argument("--delay",   type=float, default=2.0)
    parser.add_argument("--diagnose", action="store_true",
                        help="Print raw structure for first 3 seasons to verify parsing")
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("Error: --start must be <= --end")

    run(args.start, args.end, args.output, args.delay, args.diagnose)