#!/usr/bin/env python3
"""
Daily rule-based update of event dates in Google Sheets (column H).

Intended to run around 23:00. On calendar day D (e.g. 7/8 at 11pm):
  1. Pre-screen: Available (column A) == "Y"
  2. Only rows whose beginning date (column I) equals *today* (D)
  3. Roll column H as if the next day had already started (effective date = D+1)

So at 11pm on 7/8, rows with I=7/8 and A=Y are updated with threshold 8/8:
  - 7/8                         -> unchanged (single-day; let it lapse)
  - 7/8-16/8                    -> 8/8-16/8
  - 7/8, 11/8, 14/8, 20/8, ...  -> 11/8, 14/8, 20/8, ...
  - 7/8-8/8                     -> 8/8

Usage:
  python update_event_dates.py --dry-run
  python update_event_dates.py
  python update_event_dates.py --worksheet "Event(new)"  # same as default
  python update_event_dates.py --today 7/8/2026 --dry-run   # simulate 11pm on 7/8
  python update_event_dates.py --self-test                  # no Sheets needed

Env (same as other scripts):
  GOOGLE_SHEET_ID
  GOOGLE_SERVICE_ACCOUNT_JSON / GOOGLE_APPLICATION_CREDENTIALS
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import date, datetime, timedelta

from dotenv import load_dotenv

load_dotenv(override=True)

try:
    import gspread
    from google.oauth2 import service_account as gcp_service_account
except ImportError:
    gspread = None
    gcp_service_account = None

GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1G_8RMWjf0T9sNdMxKYy_Fc051I6zhdLLy6ehLak4CX4")

# Google Sheets serial date origin
_SHEETS_EPOCH = date(1899, 12, 30)

_DATE_TOKEN = re.compile(r"^\s*(\d{1,2})/(\d{1,2})\s*$")
_RANGE_TOKEN = re.compile(
    r"^\s*(\d{1,2})/(\d{1,2})\s*-\s*(\d{1,2})/(\d{1,2})\s*$"
)


def load_credentials():
    creds = None
    try:
        from xplore_automation import GOOGLE_CLOUD_CREDENTIALS as _c

        if _c:
            creds = _c
    except ImportError:
        pass
    if creds is None:
        path = (
            os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
            or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
        )
        if not path and os.path.isfile("global-headlines-474905-9494f258e0a5.json"):
            path = "global-headlines-474905-9494f258e0a5.json"
        if path and os.path.isfile(path):
            with open(path, encoding="utf-8") as f:
                creds = json.load(f)
    return creds


def format_dm(d: date) -> str:
    return f"{d.day}/{d.month}"


def parse_dm(token: str, year: int) -> date | None:
    m = _DATE_TOKEN.match(token or "")
    if not m:
        return None
    day, month = int(m.group(1)), int(m.group(2))
    try:
        return date(year, month, day)
    except ValueError:
        return None


def parse_beginning_cell(value, today: date) -> date | None:
    """Parse column I (formula result): Sheets serial, datetime, or date-like string."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            return _SHEETS_EPOCH + timedelta(days=int(value))
        except Exception:
            return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    s = str(value).strip()
    if not s or s.startswith("#"):  # #VALUE!, #N/A, etc.
        return None
    # D/M or D/M/YYYY or ISO
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    dm = parse_dm(s, today.year)
    if dm:
        return dm
    # "August 7, 2026" style
    for fmt in ("%B %d, %Y", "%b %d, %Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    return None


def update_date_string(h: str, today: date) -> str | None:
    """
    Apply roll rules to column H text.
    Returns new string, or None if unchanged / should not update.
    """
    raw = (h or "").strip()
    if not raw or raw.upper() == "N/A":
        return None

    year = today.year

    # Comma-separated list of discrete dates
    if "," in raw:
        parts = [p.strip() for p in raw.split(",") if p.strip()]
        if len(parts) <= 1:
            return None
        parsed: list[date] = []
        for p in parts:
            d = parse_dm(p, year)
            if d is None:
                return None  # unparseable — leave alone
            parsed.append(d)
        kept = [d for d in parsed if d >= today]
        if kept == parsed:
            return None
        if not kept:
            return None  # all past — leave for Available/lapse logic
        return ", ".join(format_dm(d) for d in kept)

    # Date range start-end
    m = _RANGE_TOKEN.match(raw)
    if m:
        start = date(year, int(m.group(2)), int(m.group(1)))
        end = date(year, int(m.group(4)), int(m.group(3)))
        # Year wrap for ranges that cross New Year (rare): if end < start, end is next year
        if end < start:
            try:
                end = date(year + 1, end.month, end.day)
            except ValueError:
                return None
        if start >= today:
            return None
        if end < today:
            return None  # fully expired range
        new_start = today
        if new_start == end:
            return format_dm(new_start)
        if new_start > end:
            return None
        return f"{format_dm(new_start)}-{format_dm(end)}"

    # Single date — do nothing
    if parse_dm(raw, year) is not None:
        return None

    return None


DEFAULT_WORKSHEET = "Event(new)"


def open_worksheet(name: str):
    if gspread is None or gcp_service_account is None:
        raise RuntimeError("gspread / google-auth not installed")
    creds_info = load_credentials()
    if not creds_info:
        raise RuntimeError(
            "No Google credentials. Set GOOGLE_SERVICE_ACCOUNT_JSON "
            "or place global-headlines-*.json in the working directory."
        )
    creds = gcp_service_account.Credentials.from_service_account_info(
        creds_info,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    ss = client.open_by_key(GOOGLE_SHEET_ID)
    try:
        return ss.worksheet(name)
    except gspread.exceptions.WorksheetNotFound as e:
        tabs = [ws.title for ws in ss.worksheets()]
        raise RuntimeError(
            f"Worksheet {name!r} not found. Available tabs: {tabs}"
        ) from e


def is_available_yes(value) -> bool:
    if value is None:
        return False
    return str(value).strip().upper() in {"Y", "YES", "TRUE", "1"}


def process_worksheet(
    worksheet_name: str,
    run_date: date,
    *,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    At ~23:00 on run_date: update H for rows with A=Y and I==run_date,
    rolling dates relative to tomorrow (run_date + 1).
    Returns (candidates, updated).
    """
    tomorrow = run_date + timedelta(days=1)
    ws = open_worksheet(worksheet_name)

    # A=Available, H=Date, I=beginning date. UNFORMATTED_VALUE for serial dates in I.
    cells = ws.get("A2:I", value_render_option="UNFORMATTED_VALUE")
    if not cells:
        print(f"[{worksheet_name}] no data rows")
        return 0, 0

    updates: list[dict] = []
    candidates = 0

    for i, row in enumerate(cells):
        row_num = i + 2  # sheet row
        a_val = row[0] if len(row) > 0 else ""
        h_val = row[7] if len(row) > 7 else ""
        i_val = row[8] if len(row) > 8 else ""

        if not is_available_yes(a_val):
            continue

        beginning = parse_beginning_cell(i_val, run_date)
        if beginning != run_date:
            continue

        candidates += 1
        h_str = "" if h_val is None else str(h_val).strip()
        if isinstance(h_val, (int, float)) and not isinstance(h_val, bool):
            print(f"  [skip] row {row_num}: H looks numeric ({h_val})")
            continue

        # Roll as of tomorrow (11pm run prepares dates for the next calendar day)
        new_h = update_date_string(h_str, tomorrow)
        if new_h is None:
            print(
                f"  [keep] row {row_num}: A=Y I={format_dm(beginning)} H={h_str!r}"
            )
            continue
        print(
            f"  [{'DRY' if dry_run else 'upd'}] row {row_num}: "
            f"{h_str!r} -> {new_h!r}"
        )
        updates.append(
            {
                "range": f"H{row_num}",
                "values": [[new_h]],
            }
        )

    if updates and not dry_run:
        ws.batch_update(updates, value_input_option="USER_ENTERED")

    print(
        f"[{worksheet_name}] run_date={format_dm(run_date)} "
        f"roll_as_of={format_dm(tomorrow)}: "
        f"{candidates} candidate(s) (A=Y & I=today), {len(updates)} update(s)"
        + (" (dry-run)" if dry_run else "")
    )
    return candidates, len(updates)


def parse_today_arg(s: str | None) -> date:
    if not s:
        return date.today()
    s = s.strip()
    for fmt in ("%d/%m/%Y", "%Y-%m-%d", "%d/%m/%y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    # D/M → current year
    d = parse_dm(s, date.today().year)
    if d:
        return d
    raise argparse.ArgumentTypeError(f"Invalid --today: {s!r} (use D/M/YYYY or YYYY-MM-DD)")


def self_test() -> None:
    # 11pm on 7/8 -> roll as of 8/8
    run_date = date(2026, 8, 7)
    effective = run_date + timedelta(days=1)  # 8/8
    cases = [
        ("7/8", None),
        ("7/8-16/8", "8/8-16/8"),
        ("7/8, 11/8, 14/8, 20/8, 21/8, 25/8", "11/8, 14/8, 20/8, 21/8, 25/8"),
        ("7/8-8/8", "8/8"),
        ("8/8-16/8", None),  # start already on effective day
        ("11/8, 14/8", None),  # all future
        ("N/A", None),
        ("6/8-7/8", None),  # fully past relative to 8/8
    ]
    failed = 0
    for inp, expected in cases:
        got = update_date_string(inp, effective)
        ok = got == expected
        print(f"  {'OK' if ok else 'FAIL'}: {inp!r} -> {got!r} (expected {expected!r})")
        if not ok:
            failed += 1
    y = parse_beginning_cell("7/8/2026", run_date)
    assert y == date(2026, 8, 7), y
    serial = (date(2026, 8, 7) - _SHEETS_EPOCH).days
    assert parse_beginning_cell(serial, run_date) == date(2026, 8, 7)
    assert is_available_yes("Y") and is_available_yes("y")
    assert not is_available_yes("N")
    if failed:
        raise SystemExit(f"self-test: {failed} failure(s)")
    print("self-test: all passed")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "At ~23:00: for Available=Y rows whose beginning date (I) is today, "
            "roll multi-day dates in H as of tomorrow."
        )
    )
    parser.add_argument(
        "--worksheet",
        action="append",
        dest="worksheets",
        help=f"Worksheet tab name (repeatable). Default: {DEFAULT_WORKSHEET}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print updates without writing to the sheet.",
    )
    parser.add_argument(
        "--today",
        type=parse_today_arg,
        default=None,
        help="Override run date (D/M/YYYY or YYYY-MM-DD); simulates 11pm on that day.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run local rule tests and exit (no Google Sheets).",
    )
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return

    run_date = args.today or date.today()
    tomorrow = run_date + timedelta(days=1)
    worksheets = args.worksheets or [DEFAULT_WORKSHEET]

    print(f"Sheet: {GOOGLE_SHEET_ID}")
    print(
        f"Run date (11pm day): {run_date.isoformat()} ({format_dm(run_date)}); "
        f"filter A=Y & I=={format_dm(run_date)}; "
        f"roll H as of {format_dm(tomorrow)}"
    )
    print(f"Tabs: {', '.join(worksheets)}" + (" [dry-run]" if args.dry_run else ""))

    total_c = total_u = 0
    for name in worksheets:
        try:
            c, u = process_worksheet(name, run_date, dry_run=args.dry_run)
            total_c += c
            total_u += u
        except Exception as e:
            print(f"[{name}] ERROR: {e}", file=sys.stderr)
            raise SystemExit(1) from e

    print(f"Done. {total_c} candidate(s), {total_u} update(s).")


if __name__ == "__main__":
    main()
