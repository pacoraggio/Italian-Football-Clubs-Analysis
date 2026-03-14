"""
UEFA European Cup Match Scraper — v4
=====================================
Scrapes match results from kassiesa.net for all UEFA club competitions:
  - CL : Champions League / European Cup (1955–present)
  - EL : UEFA Cup / Europa League
  - CW : Cup Winners' Cup (1961–1999)
  - CO : Conference League (2021–present)

Page structure (confirmed from real HTML):
  - Each season page contains MULTIPLE <table> elements, one per competition.
  - Within each table, rows are either:
      - Competition header : ['CHAMPIONS CUP']  (single cell)
      - Round header       : ['Round 1']         (single cell)
      - Match row          : [team_a, ctry_a, team_b, ctry_b, score_leg1, score_leg2]
      - Empty separator    : ['']

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
# URL resolution
# ---------------------------------------------------------------------------

ALL_METHODS = ["method1", "method5", "method4", "method3", "method2", "method0"]

# Primary method per era (based on confirmed working URLs)
METHOD_MAP = [
    (1956, 2008, "method1"),
    (2009, 2017, "method4"),
    (2018, 2100, "method5"),
]


def build_urls(year: int):
    primary = "method1"
    for start, end, m in METHOD_MAP:
        if start <= year <= end:
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
    competition: str
    round: str
    team_a: str
    country_a: str
    team_b: str
    country_b: str
    score_leg1: str
    score_leg2: str
    score_aggregate: str
    extra_time: str     # "yes" / "no"
    penalties: str      # "yes" / "no"


CSV_COLUMNS = [f.name for f in fields(Match)]

# ---------------------------------------------------------------------------
# Score helpers
# ---------------------------------------------------------------------------

SCORE_RE = re.compile(r"(\d+)-(\d+)")
PEN_RE   = re.compile(r"\((\d+)-(\d+)p\)")
AET_RE   = re.compile(r"\baet\b", re.IGNORECASE)


def parse_score_cell(raw: str):
    """Returns (clean_score, extra_time, penalties) for one score cell."""
    raw = raw.strip()
    if not raw:
        return "", "no", "no"

    extra_time = "no"
    penalties  = "no"

    pen_m = PEN_RE.search(raw)
    if pen_m:
        return f"{pen_m.group(1)}-{pen_m.group(2)}", "yes", "yes"

    if AET_RE.search(raw):
        extra_time = "yes"

    score_m = SCORE_RE.search(raw)
    clean = f"{score_m.group(1)}-{score_m.group(2)}" if score_m else raw

    return clean, extra_time, penalties


def compute_aggregate(leg1: str, leg2: str) -> str:
    """'3-0' + '1-2' → '4-2'. Returns '' if either leg is missing/unparseable."""
    if not leg2:
        return leg1
    m1 = SCORE_RE.match(leg1)
    m2 = SCORE_RE.match(leg2)
    if m1 and m2:
        return f"{int(m1.group(1)) + int(m2.group(1))}-{int(m1.group(2)) + int(m2.group(2))}"
    return ""


# ---------------------------------------------------------------------------
# Competition name normalisation
# ---------------------------------------------------------------------------

COMP_NAME_MAP = {
    "CHAMPIONS CUP":          "CL",
    "CHAMPION CLUBS CUP":     "CL",
    "CHAMPIONS LEAGUE":       "CL",
    "UEFA CHAMPIONS LEAGUE":  "CL",
    "CUP WINNERS CUP":        "CW",
    "UEFA CUP WINNERS CUP":   "CW",
    "UEFA CUP":               "EL",
    "UEFA EUROPA LEAGUE":     "EL",
    "EUROPA LEAGUE":          "EL",
    "UEFA CONFERENCE LEAGUE": "CO",
    "CONFERENCE LEAGUE":      "CO",
}


def normalise_comp(raw: str) -> str:
    key = raw.strip().upper()
    return COMP_NAME_MAP.get(key, raw.strip())


# ---------------------------------------------------------------------------
# Row classification
# ---------------------------------------------------------------------------

ROUND_KEYWORDS = [
    "ROUND", "QUALIFYING", "FINAL", "SEMI", "QUARTER",
    "GROUP", "STAGE", "PHASE", "PLAY", "PRELIMINARY",
]


def is_section_row(cells: list) -> bool:
    """Single non-empty cell — either competition name or round name."""
    return len(cells) == 1 and cells[0].strip() != ""


def is_match_row(cells: list) -> bool:
    """5 or 6 cells where at least one looks like a score."""
    if len(cells) < 5:
        return False
    return any(SCORE_RE.search(c) or c.strip().lower() in ("w/o", "?") for c in cells)


def is_round_name(text: str) -> bool:
    return any(kw in text.upper() for kw in ROUND_KEYWORDS)


# ---------------------------------------------------------------------------
# Parser — iterates ALL tables on the page
# ---------------------------------------------------------------------------

def parse_season(soup: BeautifulSoup, season: str) -> list:
    matches = []

    for table in soup.find_all("table"):
        current_comp  = "UNKNOWN"
        current_round = "UNKNOWN"

        for row in table.find_all("tr"):
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]

            # Skip blank rows
            if not cells or all(c == "" for c in cells):
                continue

            # Single-cell row = competition or round header
            if is_section_row(cells):
                text = cells[0].strip()
                norm = normalise_comp(text)
                if norm != text or text.upper() in COMP_NAME_MAP:
                    # It mapped to a known competition code
                    current_comp = norm
                elif is_round_name(text):
                    current_round = text
                else:
                    # Could be either; prefer to treat as comp if we haven't set one
                    # or as round if comp is already set
                    if current_comp == "UNKNOWN":
                        current_comp = norm
                    else:
                        current_round = text
                continue

            # Match data row
            if is_match_row(cells):
                # Scores are the trailing cells (last 1 or 2)
                # Walk backwards to collect score cells
                # Strip trailing empty cells first (single-leg finals have a
                # blank 6th cell in the HTML — ignore it rather than treating
                # it as a second score).
                non_score = list(cells)
                while non_score and non_score[-1].strip() == "":
                    non_score.pop()

                # Now walk backwards collecting real score cells
                score_cells = []
                while non_score:
                    last = non_score[-1].strip()
                    if SCORE_RE.search(last) or last.lower() in ("w/o", "?"):
                        score_cells.insert(0, non_score.pop())
                    else:
                        break

                if not score_cells:
                    continue

                # Remaining cells: [..., team_a, ctry_a, team_b, ctry_b]
                if len(non_score) < 4:
                    continue

                ctry_b = non_score[-1]
                team_b = non_score[-2]
                ctry_a = non_score[-3]
                team_a = non_score[-4]

                leg1_raw = score_cells[0] if len(score_cells) >= 1 else ""
                leg2_raw = score_cells[1] if len(score_cells) >= 2 else ""

                leg1, et1, pen1 = parse_score_cell(leg1_raw)
                leg2, et2, pen2 = parse_score_cell(leg2_raw)

                extra_time = "yes" if "yes" in (et1, et2)  else "no"
                penalties  = "yes" if "yes" in (pen1, pen2) else "no"
                aggregate  = compute_aggregate(leg1, leg2)

                matches.append(Match(
                    season=season,
                    competition=current_comp,
                    round=current_round,
                    team_a=team_a,   country_a=ctry_a,
                    team_b=team_b,   country_b=ctry_b,
                    score_leg1=leg1, score_leg2=leg2,
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
    tables = soup.find_all("table")
    print(f"\n  Found {len(tables)} table(s) on page.")

    for i, table in enumerate(tables):
        rows = table.find_all("tr")
        non_empty = [r for r in rows if any(
            c.get_text(strip=True) for c in r.find_all(["td","th"]))]
        print(f"\n  Table {i+1} ({len(rows)} rows, {len(non_empty)} non-empty):")
        for row in rows[:6]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td","th"])]
            if cells:
                print(f"    {cells}")

    matches = parse_season(soup, label)
    print(f"\n  → Parsed {len(matches)} total matches across all competitions.")
    # Show breakdown by competition
    from collections import Counter
    comp_counts = Counter(m.competition for m in matches)
    for comp, count in sorted(comp_counts.items()):
        print(f"     {comp}: {count} matches")
    if matches:
        print(f"\n  First 3 matches:")
        for m in matches[:3]:
            print(f"    {m}")

    time.sleep(delay)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(start: int, end: int, output: str, delay: float, diagnose_mode: bool):
    session = requests.Session()
    years = list(range(start, end + 1))

    if diagnose_mode:
        print("DIAGNOSE MODE — checking first 3 seasons only.\n")
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
                "season":          m.season,
                "competition":     m.competition,
                "round":           m.round,
                "team_a":          m.team_a,
                "country_a":       m.country_a,
                "team_b":          m.team_b,
                "country_b":       m.country_b,
                "score_leg1":      m.score_leg1,
                "score_leg2":      m.score_leg2,
                "score_aggregate": m.score_aggregate,
                "extra_time":      m.extra_time,
                "penalties":       m.penalties,
            })

    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape UEFA club competition results from kassiesa.net"
    )
    parser.add_argument("--start",    type=int, default=1981)
    parser.add_argument("--end",      type=int, default=2025)
    parser.add_argument("--output",   default="uefa_results.csv")
    parser.add_argument("--delay",    type=float, default=2.0,
                        help="Seconds between requests (default 2.0 — please be polite)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print raw structure for first 3 seasons to verify parsing")
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("Error: --start must be <= --end")

    run(args.start, args.end, args.output, args.delay, args.diagnose)
