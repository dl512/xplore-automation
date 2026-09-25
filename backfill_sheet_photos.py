#!/usr/bin/env python3
"""
Backfill missing Photo URLs on the first two Google Sheet tabs (events + exhibitions).

For each row where Available == "Y" and Photo is N/A (or empty), fetches the IG post
image from the Link column and uploads to GCS (ig-photo bucket) — same path as
process_ig_read_sheet.extract_photo / manage_photo.

Usage:
  python backfill_sheet_photos.py              # both first tabs
  python backfill_sheet_photos.py --dry-run    # list rows only, no fetch/write
  python backfill_sheet_photos.py --limit 5    # process at most 5 rows per tab
  python backfill_sheet_photos.py --tab 0      # first tab only
"""

import argparse
import os
import sys
import time

from dotenv import load_dotenv

load_dotenv(override=True)

from process_ig_read_sheet import (
    DELAY_BETWEEN_POSTS,
    GOOGLE_CLOUD_CREDENTIALS,
    GOOGLE_SHEET_ID,
    extract_photo,
    gcp_service_account,
    gspread,
)

PHOTO_MISSING = {"", "n/a", "na", "none", "null"}
AVAILABLE_YES = {"y", "yes", "true", "1"}
SHEET_READ_RETRIES = 5
SHEET_WRITE_RETRIES = 5
SHEET_RETRY_BASE_SEC = 3
SHEET_HTTP_TIMEOUT = 120


def _configure_stdout():
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass


def _sheets_available():
    return bool(gspread and gcp_service_account and GOOGLE_CLOUD_CREDENTIALS)


def _retry(label, fn, retries=SHEET_READ_RETRIES):
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt >= retries:
                break
            wait = SHEET_RETRY_BASE_SEC * attempt
            print(f"  [WARN] {label} failed ({e!r}), retry {attempt}/{retries} in {wait}s...")
            time.sleep(wait)
    raise last_err


def open_spreadsheet():
    creds = gcp_service_account.Credentials.from_service_account_info(
        GOOGLE_CLOUD_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    client = gspread.authorize(creds)
    if hasattr(client, "http_client") and hasattr(client.http_client, "timeout"):
        client.http_client.timeout = SHEET_HTTP_TIMEOUT
    return client.open_by_key(GOOGLE_SHEET_ID)


def _header_index(headers, *names):
    lowered = {h.strip().lower(): i for i, h in enumerate(headers) if h and str(h).strip()}
    for name in names:
        key = name.strip().lower()
        if key in lowered:
            return lowered[key]
    return None


def _pad_cols(*cols):
    n = max(len(c) for c in cols) if cols else 0
    return [list(c) + [""] * (n - len(c)) for c in cols]


def _is_available_y(value):
    return (value or "").strip().lower() in AVAILABLE_YES


def _photo_missing(value):
    return (value or "").strip().lower() in PHOTO_MISSING


def _is_ig_link(url):
    u = (url or "").lower()
    return "instagram.com" in u and ("/p/" in u or "/reel/" in u)


def _load_rows_csv_fallback(worksheet):
    """Read tab via public CSV export (lighter than get_all_values on large sheets)."""
    import pandas as pd

    gid = worksheet.id
    url = (
        f"https://docs.google.com/spreadsheets/d/{GOOGLE_SHEET_ID}/export"
        f"?format=csv&gid={gid}"
    )
    df = pd.read_csv(url)
    headers = [str(c) for c in df.columns]
    rows = [headers]
    for _, series in df.iterrows():
        rows.append([str(v) if v == v else "" for v in series.tolist()])
    return rows


def _load_sheet_rows(worksheet):
    """Load rows using column-wise reads (smaller API payloads than get_all_values)."""
    headers = _retry(
        f"read header row ({worksheet.title})",
        lambda: worksheet.row_values(1),
    )
    if not headers:
        return []

    avail_idx = _header_index(headers, "Available", "Availability")
    photo_idx = _header_index(headers, "Photo")
    link_idx = _header_index(headers, "Link")
    name_idx = _header_index(headers, "Event name", "Event Name", "Name")

    if avail_idx is None or photo_idx is None or link_idx is None:
        raise ValueError(
            f"Tab '{worksheet.title}' missing required columns "
            f"(need Available/Availability, Photo, Link). Headers: {headers[:10]}"
        )

    col_1based = [avail_idx + 1, photo_idx + 1, link_idx + 1]
    if name_idx is not None:
        col_1based.append(name_idx + 1)

    def read_col(col):
        return _retry(
            f"read column {col} ({worksheet.title})",
            lambda c=col: worksheet.col_values(c),
        )

    try:
        avail_col, photo_col, link_col = read_col(col_1based[0]), read_col(col_1based[1]), read_col(col_1based[2])
        name_col = read_col(col_1based[3]) if name_idx is not None else []
    except Exception as e:
        print(f"  [WARN] column read failed ({e!r}), trying CSV export fallback...")
        return _load_rows_csv_fallback(worksheet)

    avail_col, photo_col, link_col, name_col = _pad_cols(
        avail_col, photo_col, link_col, name_col or [""]
    )
    rows = [headers]
    for i in range(1, len(avail_col)):
        row = [""] * len(headers)
        row[avail_idx] = avail_col[i]
        row[photo_idx] = photo_col[i]
        row[link_idx] = link_col[i]
        if name_idx is not None and i < len(name_col):
            row[name_idx] = name_col[i]
        rows.append(row)
    return rows


def find_rows_needing_photo(worksheet):
    """Return list of dicts: row_num (1-based sheet row), link, event_name, photo_col (1-based)."""
    rows = _load_sheet_rows(worksheet)
    if not rows:
        return []

    headers = rows[0]
    avail_idx = _header_index(headers, "Available", "Availability")
    photo_idx = _header_index(headers, "Photo")
    link_idx = _header_index(headers, "Link")
    name_idx = _header_index(headers, "Event name", "Event Name", "Name")

    photo_col_1based = photo_idx + 1
    out = []
    for sheet_row, row in enumerate(rows[1:], start=2):
        if avail_idx >= len(row) or photo_idx >= len(row) or link_idx >= len(row):
            continue
        if not _is_available_y(row[avail_idx]):
            continue
        if not _photo_missing(row[photo_idx]):
            continue
        link = (row[link_idx] or "").strip()
        if not link or not _is_ig_link(link):
            continue
        name = (row[name_idx] if name_idx is not None and name_idx < len(row) else "") or ""
        out.append(
            {
                "row_num": sheet_row,
                "link": link,
                "event_name": name.strip() or "(no name)",
                "photo_col": photo_col_1based,
            }
        )
    return out


def update_photo_cell(worksheet, row_num, photo_col, photo_url):
    _retry(
        f"update row {row_num} ({worksheet.title})",
        lambda: worksheet.update_cell(row_num, photo_col, photo_url),
        retries=SHEET_WRITE_RETRIES,
    )


def backfill_worksheet(worksheet, *, dry_run=False, limit=None, delay=None):
    pending = find_rows_needing_photo(worksheet)
    if limit is not None:
        pending = pending[: max(0, limit)]

    print(f"\n[{worksheet.title}] {len(pending)} row(s) with Photo N/A and Available=Y")
    updated = 0
    failed = 0

    for i, item in enumerate(pending, start=1):
        label = f"{item['event_name'][:50]} — row {item['row_num']}"
        print(f"  [{i}/{len(pending)}] {label}")
        print(f"    Link: {item['link'][:80]}")

        if dry_run:
            continue

        try:
            photo_url = extract_photo(item["link"])
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"    [FAIL] extract error: {e}")
            failed += 1
            if delay and i < len(pending):
                time.sleep(delay)
            continue

        if not photo_url:
            print("    [FAIL] could not fetch/upload photo")
            failed += 1
            if delay and i < len(pending):
                time.sleep(delay)
            continue

        try:
            update_photo_cell(worksheet, item["row_num"], item["photo_col"], photo_url)
        except Exception as e:
            print(f"    [FAIL] sheet update: {e}")
            failed += 1
            if delay and i < len(pending):
                time.sleep(delay)
            continue

        print(f"    [OK] Photo -> {photo_url}")
        updated += 1

        if delay and i < len(pending):
            time.sleep(delay)

    return {"pending": len(pending), "updated": updated, "failed": failed}


def main():
    parser = argparse.ArgumentParser(
        description="Backfill Photo column (M) from IG Link (G) for Available=Y rows on first two sheet tabs."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List matching rows without fetching photos or updating the sheet.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Max rows to process per tab (default: all).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=DELAY_BETWEEN_POSTS,
        help=f"Seconds between IG fetches (default: {DELAY_BETWEEN_POSTS}).",
    )
    parser.add_argument(
        "--tab",
        choices=("0", "1", "both"),
        default="both",
        help="Which tab to process: 0=first, 1=second, both=first two (default).",
    )
    args = parser.parse_args()
    _configure_stdout()

    if not _sheets_available():
        print(
            "Error: need gspread + Google service account credentials "
            "(GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS)."
        )
        return 1

    if not args.dry_run and not os.path.exists("cookies.pkl") and not os.path.exists(
        "session-instaloader"
    ):
        print(
            "[WARN] No cookies.pkl or session-instaloader — Instaloader may fail on private posts. "
            "Run process_ig_read_sheet.py --save-cookies if needed."
        )

    spreadsheet = open_spreadsheet()
    worksheets = spreadsheet.worksheets()
    if len(worksheets) < 1:
        print("Error: spreadsheet has no worksheets.")
        return 1

    if args.tab == "both":
        targets = worksheets[:2]
    else:
        idx = int(args.tab)
        if idx >= len(worksheets):
            print(f"Error: tab index {idx} not found (only {len(worksheets)} tab(s)).")
            return 1
        targets = [worksheets[idx]]

    print(f"Sheet: {GOOGLE_SHEET_ID}")
    print(f"Tabs: {', '.join(ws.title for ws in targets)}")

    totals = {"pending": 0, "updated": 0, "failed": 0}
    for ws in targets:
        try:
            stats = backfill_worksheet(
                ws, dry_run=args.dry_run, limit=args.limit, delay=args.delay
            )
        except Exception as e:
            print(f"\n[ERROR] Tab '{ws.title}' failed: {e}")
            continue
        for k in totals:
            totals[k] += stats[k]

    mode = "dry-run" if args.dry_run else "done"
    print(
        f"\n{mode.capitalize()}: {totals['pending']} candidate row(s), "
        f"{totals['updated']} updated, {totals['failed']} failed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
