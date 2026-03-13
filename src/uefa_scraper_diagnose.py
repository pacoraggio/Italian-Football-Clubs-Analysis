"""
UEFA European Cup Match Scraper — v2
=====================================
Scrapes match results from kassiesa.net for all three UEFA club competitions:
  - CL  : Champions League / European Cup (1955–present)
  - EL  : UEFA Cup / Europa League
  - CW  : Cup Winners' Cup (1961–1999)
  - CO  : Conference League (2021–present)

Output CSV columns:
  season, competition, round, date,
  team_a, country_a, team_b, country_b,
  score_90min, score_final, extra_time, penalties

Usage:
  pip install requests beautifulsoup4
  python uefa_scraper.py --start 1981 --end 1983 --diagnose   ← run this first!
  python uefa_scraper.py --start 1981 --end 2025
  python uefa_scraper.py --start 1981 --end 2025 --output uefa_results.csv
"""

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, fields

import requests
from bs4 import BeautifulSoup, Tag

# ---------------------------------------------------------------------------
# URL / method map
# kassiesa.net stores data in different subfolders per coefficient era.
# If the primary URL 404s we automatically try all other methods.
# ---------------------------------------------------------------------------

METHOD_MAP = [
    (1956, 1998, "method0"),
    (1999, 2008, "method1"),
    (2009, 2017, "method4"),
    (2018, 2099, "method5"),
]

ALL_METHODS = ["method0", "method1", "method2", "method3", "method4", "method5"]


def primary_method(year: int) -> str:
    for start, end, method in METHOD_MAP:
        if start <= year <= end:
            return method
    return "method5"


def build_urls(year: int):
    """Return all candidate URLs for a season, most likely first."""
    primary = primary_method(year)
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
    date: str
    team_a: str
    country_a: str
    team_b: str
    country_b: str
    score_90min: str
    score_final: str
    extra_time: str    # "yes" / "no"
    penalties: str     # "yes" / "no"


CSV_COLUMNS = [f.name for f in fields(Match)]

# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------

SCORE_RE = re.compile(
    r"(?P<s90>\d+-\d+)"
    r"(?:\s+"
    r"(?:"
    r"\((?P<sfinal>\d+-\d+)(?P<pen>p)?\)"
    r"|(?P<aet>aet)"
    r"))?"
)


def parse_score(raw: str):
    raw = raw.strip()
    if not raw or raw.lower() in ("w/o", "-", "?", ""):
        return raw, raw, "no", "no"
    m = SCORE_RE.search(raw)
    if not m:
        return raw, raw, "no", "no"
    s90 = m.group("s90")
    sfinal_raw = m.group("sfinal")
    pen = bool(m.group("pen"))
    aet = bool(m.group("aet"))
    extra_time = "yes" if (sfinal_raw or aet or pen) else "no"
    penalties = "yes" if pen else "no"
    score_final = sfinal_raw if sfinal_raw else s90
    return s90, score_final, extra_time, penalties


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

SESSION_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-GB,en;q=0.9",
}


def fetch_page(session: requests.Session, year: int, verbose: bool = False):
    """Try each candidate URL. Returns (url, html_text) or (None, None)."""
    for url in build_urls(year):
        try:
            resp = session.get(url, headers=SESSION_HEADERS, timeout=20)
            if resp.status_code == 200:
                if verbose:
                    print(f"    ✓ {url}")
                return url, resp.text
            if verbose:
                print(f"    ✗ HTTP {resp.status_code}: {url}")
        except requests.RequestException as e:
            if verbose:
                print(f"    ✗ Error: {url}  ({e})")
    return None, None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

COMP_CODES = {"CL", "EL", "CW", "CO", "ECL", "UC"}


def is_comp(val: str) -> bool:
    return val.upper().strip() in COMP_CODES


def _parse_table(table: Tag, season: str) -> list:
    matches = []
    rows = table.find_all("tr")

    # ---- Step 1: detect column positions from header row ----
    col_map = {}
    known_headers = {
        "cup":   ["cup", "comp", "competition"],
        "round": ["round", "rd", "rnd"],
        "home":  ["home", "team a", "team1", "home team"],
        "hctry": ["hctry", "hc", "cnt a", "ctry a", "country a", "hcountry"],
        "score": ["score", "result", "res"],
        "away":  ["away", "team b", "team2", "away team"],
        "actry": ["actry", "ac", "cnt b", "ctry b", "country b", "acountry"],
        "date":  ["date"],
    }
    for row in rows:
        ths = [th.get_text(strip=True).lower() for th in row.find_all("th")]
        if not ths:
            # Also try <td> rows that look like headers (bold, or first row)
            tds = [td.get_text(strip=True).lower() for td in row.find_all("td")]
            if tds and tds[0] in ("cup", "comp", "cl", "el", "cw"):
                ths = tds
        for idx, h in enumerate(ths):
            for key, aliases in known_headers.items():
                if h in aliases and key not in col_map:
                    col_map[key] = idx
        if col_map:
            break

    # ---- Step 2: parse each data row ----
    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all("td")]
        if not cells:
            continue

        # --- Path A: use col_map ---
        if col_map and "cup" in col_map and "score" in col_map:
            try:
                comp = cells[col_map["cup"]].upper()
                if not is_comp(comp):
                    continue
                round_ = cells[col_map["round"]] if "round" in col_map and col_map["round"] < len(cells) else ""
                team_a = cells[col_map["home"]] if "home" in col_map and col_map["home"] < len(cells) else ""
                ctry_a = cells[col_map["hctry"]] if "hctry" in col_map and col_map["hctry"] < len(cells) else ""
                raw_score = cells[col_map["score"]] if col_map["score"] < len(cells) else ""
                team_b = cells[col_map["away"]] if "away" in col_map and col_map["away"] < len(cells) else ""
                ctry_b = cells[col_map["actry"]] if "actry" in col_map and col_map["actry"] < len(cells) else ""
                date = cells[col_map["date"]] if "date" in col_map and col_map["date"] < len(cells) else ""
                s90, sfinal, et, pen = parse_score(raw_score)
                matches.append(Match(season, comp, round_, date,
                                     team_a, ctry_a, team_b, ctry_b,
                                     s90, sfinal, et, pen))
                continue
            except (IndexError, KeyError):
                pass  # fall through to heuristic

        # --- Path B: heuristic — find score cell as anchor ---
        score_idx = None
        for i, c in enumerate(cells):
            if re.match(r"^\d+-\d+", c.strip()) or c.strip().lower() in ("w/o", "?"):
                score_idx = i
                break
        if score_idx is None:
            continue

        # Detect optional date prefix
        offset = 1 if (cells and re.match(r"\d{1,2}[./-]\d{1,2}[./-]\d{2,4}", cells[0])) else 0

        # Need at least: [date?,] comp, round, ..., team_a, ctry_a, score, team_b, ctry_b
        if score_idx < offset + 4:
            continue

        comp = cells[offset].upper()
        if not is_comp(comp):
            continue

        round_ = cells[offset + 1] if offset + 1 < len(cells) else ""
        date = cells[0] if offset else ""
        ctry_a = cells[score_idx - 1] if score_idx >= 1 else ""
        team_a = cells[score_idx - 2] if score_idx >= 2 else ""
        raw_score = cells[score_idx]
        team_b = cells[score_idx + 1] if score_idx + 1 < len(cells) else ""
        ctry_b = cells[score_idx + 2] if score_idx + 2 < len(cells) else ""

        s90, sfinal, et, pen = parse_score(raw_score)
        matches.append(Match(season, comp, round_, date,
                             team_a, ctry_a, team_b, ctry_b,
                             s90, sfinal, et, pen))
    return matches


# Legacy plain-text <pre> block
LINE_RE = re.compile(
    r"^(?P<cup>CL|EL|CW|CO|ECL)\s+"
    r"(?P<round>\S+)\s+"
    r"(?P<team_a>.+?)\s{2,}"
    r"(?P<ctry_a>[A-Z][a-z]{1,3})\s+"
    r"(?P<score>\d+-\d+(?:\s+\S+)?)\s+"
    r"(?P<team_b>.+?)\s{2,}"
    r"(?P<ctry_b>[A-Z][a-z]{1,3})\s*$"
)


def _parse_pre(text: str, season: str) -> list:
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
            season, m.group("cup").upper(), m.group("round"), "",
            m.group("team_a").strip(), m.group("ctry_a"),
            m.group("team_b").strip(), m.group("ctry_b"),
            s90, sfinal, et, pen,
        ))
    return matches


def parse_html(soup: BeautifulSoup, season: str) -> list:
    table = soup.find("table")
    if table:
        return _parse_table(table, season)
    pre = soup.find("pre")
    if pre:
        return _parse_pre(pre.get_text(), season)
    return []


# ---------------------------------------------------------------------------
# Diagnose mode — prints raw structure so you can debug
# ---------------------------------------------------------------------------

def diagnose(session: requests.Session, year: int, delay: float):
    label = season_label(year)
    print(f"\n{'='*65}")
    print(f"SEASON {label}  (year param = {year})")
    print("="*65)

    url, html = fetch_page(session, year, verbose=True)
    if html is None:
        print("  → FAILED: no URL returned data.")
        return

    soup = BeautifulSoup(html, "html.parser")

    table = soup.find("table")
    if table:
        rows = table.find_all("tr")
        print(f"\n  <table> found — showing first 6 rows:")
        for row in rows[:6]:
            cells = [c.get_text(strip=True) for c in row.find_all(["td", "th"])]
            print(f"    {cells}")
    else:
        print("  No <table> found.")

    pre = soup.find("pre")
    if pre:
        lines = [l for l in pre.get_text().splitlines() if l.strip()]
        print(f"\n  <pre> found — first 8 non-empty lines:")
        for l in lines[:8]:
            print(f"    {l}")
    else:
        print("  No <pre> found.")

    matches = parse_html(soup, label)
    print(f"\n  → Parsed {len(matches)} matches.")
    if matches:
        print(f"  First: {matches[0]}")
    else:
        print("  *** 0 matches — copy the rows above and share for debugging ***")

    time.sleep(delay)


# ---------------------------------------------------------------------------
# Main scrape loop
# ---------------------------------------------------------------------------

def scrape_season(session: requests.Session, year: int, delay: float) -> list:
    _, html = fetch_page(session, year)
    if html is None:
        print(f"  [WARN] All URLs failed for {season_label(year)}", file=sys.stderr)
        return []
    matches = parse_html(BeautifulSoup(html, "html.parser"), season_label(year))
    time.sleep(delay)
    return matches


def run(start: int, end: int, output: str, delay: float, diagnose_mode: bool):
    session = requests.Session()
    years = list(range(start, end + 1))

    if diagnose_mode:
        print("DIAGNOSE MODE — inspecting first 3 seasons.")
        print("Run without --diagnose once parsing looks correct.\n")
        for year in years[:3]:
            diagnose(session, year, delay)
        return

    all_matches = []
    print(f"Scraping {len(years)} seasons "
          f"({season_label(start)} → {season_label(end)}) …\n")

    for year in years:
        label = season_label(year)
        print(f"  {label} … ", end="", flush=True)
        matches = scrape_season(session, year, delay)
        print(f"{len(matches)} matches")
        all_matches.extend(matches)

    print(f"\nTotal: {len(all_matches)} matches")
    print(f"Writing → {output}")

    with open(output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for m in all_matches:
            writer.writerow({
                "season": m.season, "competition": m.competition,
                "round": m.round, "date": m.date,
                "team_a": m.team_a, "country_a": m.country_a,
                "team_b": m.team_b, "country_b": m.country_b,
                "score_90min": m.score_90min, "score_final": m.score_final,
                "extra_time": m.extra_time, "penalties": m.penalties,
            })

    print("Done ✓")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Scrape UEFA club competition results from kassiesa.net"
    )
    parser.add_argument("--start", type=int, default=1981,
                        help="First season end-year (e.g. 1981 = 1980/81 season)")
    parser.add_argument("--end",   type=int, default=2025,
                        help="Last season end-year  (e.g. 2025 = 2024/25 season)")
    parser.add_argument("--output", default="uefa_results.csv")
    parser.add_argument("--delay",  type=float, default=2.0,
                        help="Seconds between requests (be polite, default 2.0)")
    parser.add_argument("--diagnose", action="store_true",
                        help="Print raw HTML structure for first 3 seasons to debug parsing")
    args = parser.parse_args()

    if args.start > args.end:
        sys.exit("Error: --start must be <= --end")

    run(args.start, args.end, args.output, args.delay, args.diagnose)
