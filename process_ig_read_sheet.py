#!/usr/bin/env python3
"""
Read sheet (queue): Instagram links (column A). Caption is fetched with yt-dlp,
not from the sheet. Write sheet (database): Event(new) + input.

Workflow: load existing links from the write sheet into a temp list. Walk the
read sheet top to bottom. If a link is already in the temp list, delete that
read-sheet row. Otherwise fetch caption via yt-dlp, extract, write, add the
link to the temp list, then delete the row. Duplicate links on the read sheet
are processed at most once.

Read sheet:  https://docs.google.com/spreadsheets/d/1UkOAZZEv780FHvPwras-MqXeVLoCFAmdYtWjymdHfX8
  A=link (required). B/C used only if yt-dlp omits username/date.

No Instagram login. Photo is the yt-dlp thumbnail URL written as-is (not GCS).

Usage:
  python process_ig_read_sheet.py --save-cookies --no-headless

  python process_ig_read_sheet.py
  python process_ig_read_sheet.py --no-headless

  python process_ig_read_sheet.py --no-extract

  python process_ig_read_sheet.py --from-ig-saved

  LLM: tries GPT models (gpt-5-mini, gpt-4o-mini, gpt-3.5-turbo) in order; if each
  call errors or returns blank event fields, falls back to google/gemma-2-9b-it.
  Override with LLM_PRIMARY_MODELS=id1,id2 and LLM_FALLBACK_MODEL=id (optional).
"""

import argparse
import csv
import json
import os
import pickle
import re
import sys
import time
import uuid
from datetime import datetime

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

load_dotenv(override=True)

from extraction_details import (
    extract_info_with_model_fallback,
    _primary_models_from_env,
    _fallback_model_from_env,
    sheet_available_formula,
    sheet_beginning_date_formula,
)

try:
    from update_event_dates import normalize_date_for_sheet
except ImportError:
    def normalize_date_for_sheet(h, as_of=None):
        return (h or "").strip()

# Google Sheet + GCS (standalone: set GOOGLE_SERVICE_ACCOUNT_JSON or GOOGLE_APPLICATION_CREDENTIALS)
gspread = None
gcp_service_account = None
gcs_storage = None
try:
    import gspread as _gspread
    from google.oauth2 import service_account as _gcp_sa
    from google.cloud import storage as _gcs_storage

    gspread = _gspread
    gcp_service_account = _gcp_sa
    gcs_storage = _gcs_storage
except ImportError:
    pass

GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1G_8RMWjf0T9sNdMxKYy_Fc051I6zhdLLy6ehLak4CX4")
GOOGLE_CLOUD_CREDENTIALS = None
INPUT_SHEET_TAB = os.getenv("GOOGLE_INPUT_SHEET_TAB", "input")
EVENT_SHEET_TAB = os.getenv("GOOGLE_EVENT_SHEET_TAB", "Event(new)")
SAVED_LINKS_SHEET_ID = os.getenv(
    "SAVED_LINKS_SHEET_ID",
    "1UkOAZZEv780FHvPwras-MqXeVLoCFAmdYtWjymdHfX8",
)
SAVED_LINKS_SHEET_TAB = os.getenv("SAVED_LINKS_SHEET_TAB", "Sheet1")
SAVED_LINKS_URL_COL = 1
WRITE_SHEET_LINK_COL = 7  # column G "Link" on Event(new) / input
_IG_POST_URL_RE = re.compile(
    r"https?://(?:www\.)?instagram\.com/(?:p|reel)/[A-Za-z0-9_-]+",
    re.I,
)

try:
    from xplore_automation import GOOGLE_CLOUD_CREDENTIALS as _X_CREDS, GOOGLE_SHEET_ID as _X_SID

    if _X_CREDS:
        GOOGLE_CLOUD_CREDENTIALS = _X_CREDS
    if _X_SID:
        GOOGLE_SHEET_ID = _X_SID
except ImportError:
    pass

if GOOGLE_CLOUD_CREDENTIALS is None:
    _cred_path = (
        os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON", "").strip()
        or os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "").strip()
    )
    if _cred_path and os.path.isfile(_cred_path):
        try:
            with open(_cred_path, encoding="utf-8") as _cf:
                GOOGLE_CLOUD_CREDENTIALS = json.load(_cf)
        except Exception as _e:
            print(f"[WARN] Could not load Google credentials from {_cred_path}: {_e}")

_SHEETS_AVAILABLE = bool(
    gspread is not None
    and gcp_service_account is not None
    and GOOGLE_CLOUD_CREDENTIALS
)

BUCKET_NAME = "ig-photo"
_storage_bucket = None
_ig_requests_session = None
IG_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def extract_shortcode_from_url(url):
    """Parse shortcode from /p/ or /reel/ Instagram URLs."""
    if not url:
        return ""
    m = re.search(r"/(?:p|reel)/([A-Za-z0-9_-]+)", url)
    return m.group(1) if m else ""


def ig_post_key(raw):
    """Dedup key: Instagram shortcode, so www /p/ vs /reel/ still match."""
    return extract_shortcode_from_url(ig_url_from_text(raw) or raw or "")


def get_requests_session():
    """requests.Session with cookies.pkl (for post HTML when Selenium driver is unavailable)."""
    global _ig_requests_session
    if _ig_requests_session is not None:
        return _ig_requests_session
    session = requests.Session()
    session.headers.update(IG_REQUEST_HEADERS)
    if os.path.exists(COOKIES_FILE):
        try:
            session.get(INSTAGRAM_HOME, timeout=15)
            with open(COOKIES_FILE, "rb") as f:
                for cookie in pickle.load(f):
                    session.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie.get("domain", ".instagram.com"),
                        path=cookie.get("path", "/"),
                    )
        except Exception as e:
            print(f"[WARN] could not load {COOKIES_FILE} into requests session: {e}")
    _ig_requests_session = session
    return session


def _load_instaloader_session(L):
    """Load Instaloader session from session file and/or Selenium cookies.pkl."""
    try:
        if os.path.exists("session-instaloader"):
            L.load_session_from_file("session-instaloader")
    except Exception:
        pass
    if os.path.exists(COOKIES_FILE):
        try:
            L.context._session.get(INSTAGRAM_HOME)
            with open(COOKIES_FILE, "rb") as f:
                for cookie in pickle.load(f):
                    L.context._session.cookies.set(
                        cookie["name"],
                        cookie["value"],
                        domain=cookie.get("domain", ".instagram.com"),
                        path=cookie.get("path", "/"),
                    )
        except Exception:
            pass

def get_storage_bucket():
    """Return GCS bucket for photo upload (ig-photo), or None if credentials unavailable."""
    global _storage_bucket
    if _storage_bucket is not None:
        return _storage_bucket
    if not GOOGLE_CLOUD_CREDENTIALS or not gcs_storage:
        return None
    try:
        creds = gcp_service_account.Credentials.from_service_account_info(GOOGLE_CLOUD_CREDENTIALS)
        client = gcs_storage.Client(credentials=creds)
        _storage_bucket = client.bucket(BUCKET_NAME)
        return _storage_bucket
    except Exception:
        return None


def manage_photo(image_url):
    """Download image and upload to GCS (ig-photo), return public URL. Same logic as xplore_automation."""
    bucket = get_storage_bucket()
    if not bucket or not image_url:
        return ""
    try:
        response = requests.get(
            image_url,
            timeout=15,
            headers={**IG_REQUEST_HEADERS, "Referer": "https://www.instagram.com/"},
        )
        if response.status_code != 200:
            return ""
        local_file = "downloaded_image.jpg"
        with open(local_file, "wb") as f:
            f.write(response.content)
        timestamp = int(time.time())
        unique_filename = f"image_{timestamp}.jpg"
        blob = bucket.blob(unique_filename)
        blob.upload_from_filename(local_file)
        os.remove(local_file)
        return f"https://storage.googleapis.com/{BUCKET_NAME}/{unique_filename}"
    except Exception:
        return ""


def instagram_media_url(url):
    """Direct JPEG URL for a post/reel shortcode (works without GraphQL login)."""
    shortcode = extract_shortcode_from_url(url)
    if not shortcode:
        return ""
    return f"https://www.instagram.com/p/{shortcode}/media/?size=l"


def extract_og_image_url(url, driver=None):
    """Parse og:image / twitter:image from post HTML (requests+cookies or Selenium)."""
    soup = get_content_sync(url, driver=driver)
    if not soup:
        return ""
    for attrs in (
        {"property": "og:image"},
        {"name": "twitter:image"},
        {"property": "twitter:image"},
    ):
        tag = soup.find("meta", attrs=attrs)
        if tag and tag.get("content"):
            return tag["content"].strip()
    return ""


def extract_photo(url, driver=None):
    """Download the public /media/?size=l JPEG and upload to GCS. No Instagram login."""
    shortcode = extract_shortcode_from_url(url)
    if not shortcode:
        return ""
    return manage_photo(instagram_media_url(url)) or ""

# Profile username whose saved posts we want (must be the logged-in user)
DEFAULT_USERNAME = "xplore.hk"
COOKIES_FILE = "cookies.pkl"
INSTAGRAM_HOME = "https://www.instagram.com"
WAIT_TIMEOUT = 15
DELAY_BETWEEN_POSTS = 2  # seconds between fetching posts (rate limit)
DEFAULT_PROCESSED_TRACKER = "processed_ig_links.txt"


def normalize_post_url(url):
    """Match dedup logic in extract_post_links: no query string, no trailing slash."""
    if not url:
        return ""
    return url.strip().split("?")[0].rstrip("/")


def load_processed_links(tracker_path):
    """Load set of normalized URLs already processed (txt: one per line, or csv with 'url' column)."""
    out = set()
    if not tracker_path or not os.path.isfile(tracker_path):
        return out
    try:
        if tracker_path.lower().endswith(".csv"):
            with open(tracker_path, encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                for row in reader:
                    if not row:
                        continue
                    cell = (row[0] or "").strip()
                    if not cell or cell.lower() == "url":
                        continue
                    out.add(normalize_post_url(cell))
        else:
            with open(tracker_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    out.add(normalize_post_url(line))
    except Exception as e:
        print(f"[WARN] Could not read tracker {tracker_path}: {e}")
    return out


def append_processed_link(tracker_path, normalized_url, processed_set):
    """Append one normalized URL to tracker and update processed_set."""
    if not tracker_path or not normalized_url or normalized_url in processed_set:
        return
    try:
        if tracker_path.lower().endswith(".csv"):
            file_exists = os.path.isfile(tracker_path) and os.path.getsize(tracker_path) > 0
            with open(tracker_path, "a", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                if not file_exists:
                    w.writerow(["url"])
                w.writerow([normalized_url])
        else:
            with open(tracker_path, "a", encoding="utf-8") as f:
                f.write(normalized_url + "\n")
        processed_set.add(normalized_url)
    except Exception as e:
        print(f"[WARN] Could not append to tracker {tracker_path}: {e}")


def get_content_sync(url, driver=None):
    """Fetch post HTML. Prefer logged-in Selenium driver; fallback to requests + cookies.pkl."""
    if driver is not None:
        try:
            safe_driver_get(driver, url, wait_after=1.5)
            return BeautifulSoup(driver.page_source, "html.parser")
        except Exception as e:
            print(f"  [WARN] selenium fetch failed {url}: {e}")
    try:
        session = get_requests_session()
        r = session.get(url, timeout=30)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
        print(f"  [WARN] fetch HTTP {r.status_code} for {url}")
    except Exception as e:
        print(f"  [WARN] fetch failed {url}: {e}")
    return None


def extract_username_and_details(soup):
    """Extract username, caption/details, and post date from post page (meta description)."""
    meta = soup.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        match = re.search(r"([^ ]+) on (.+?): ([\S\s]+)", meta["content"])
        if match:
            username = match.group(1)
            date_part = match.group(2)
            details = match.group(3)
            date_match = re.search(r"([A-Za-z]+)\s+(\d{1,2}),\s+(\d{4})", date_part)
            post_date = f"{date_match.group(1)} {date_match.group(2)}, {date_match.group(3)}" if date_match else None
            return username, details, post_date
    meta_tw = soup.find("meta", attrs={"name": "twitter:title"})
    if meta_tw:
        m = re.search(r"@([\w._]+)", meta_tw.get("content", ""))
        if m:
            og = soup.find("meta", property="og:title")
            return m.group(1), (og.get("content", "") if og else ""), None
    return "", "", None


def init_google_sheets(tab_name=None):
    """
    Open a worksheet and find the first empty row.
    Default tab: input. Returns (sheet, current_row) or (None, None).
    """
    tab = tab_name or INPUT_SHEET_TAB
    if not _SHEETS_AVAILABLE or not GOOGLE_CLOUD_CREDENTIALS:
        return None, None
    try:
        creds = gcp_service_account.Credentials.from_service_account_info(
            GOOGLE_CLOUD_CREDENTIALS,
            scopes=["https://www.googleapis.com/auth/spreadsheets"],
        )
        client = gspread.authorize(creds)
        spreadsheet = client.open_by_key(GOOGLE_SHEET_ID)
        try:
            names = [ws.title for ws in spreadsheet.worksheets()]
            if tab in names:
                sheet = spreadsheet.worksheet(tab)
            elif tab == INPUT_SHEET_TAB and "input" in names:
                sheet = spreadsheet.worksheet("input")
            else:
                print(f"[WARN] Worksheet {tab!r} not found; tabs={names}")
                return None, None
        except Exception:
            return None, None
        row = find_first_empty_row(sheet)
        return sheet, row
    except Exception as e:
        print(f"[WARN] Google Sheets init failed ({tab}): {e}")
        return None, None


def init_input_and_event_sheets():
    """Open input + Event tabs for routing. Returns (input_sheet, input_row, event_sheet, event_row)."""
    input_sheet, input_row = init_google_sheets(INPUT_SHEET_TAB)
    event_sheet, event_row = init_google_sheets(EVENT_SHEET_TAB)
    return input_sheet, input_row, event_sheet, event_row


def _gspread_client():
    if not _SHEETS_AVAILABLE or not GOOGLE_CLOUD_CREDENTIALS:
        return None
    creds = gcp_service_account.Credentials.from_service_account_info(
        GOOGLE_CLOUD_CREDENTIALS,
        scopes=["https://www.googleapis.com/auth/spreadsheets"],
    )
    return gspread.authorize(creds)


def open_saved_links_sheet(sheet_id=None, tab_name=None):
    """Open the Instagram URL queue spreadsheet (default: ig_post_links / Sheet1)."""
    client = _gspread_client()
    if client is None:
        print("[WARN] Google Sheets not available (missing gspread/credentials).")
        return None
    sid = sheet_id or SAVED_LINKS_SHEET_ID
    tab = tab_name or SAVED_LINKS_SHEET_TAB
    try:
        spreadsheet = client.open_by_key(sid)
        names = [ws.title for ws in spreadsheet.worksheets()]
        if tab in names:
            return spreadsheet.worksheet(tab)
        if spreadsheet.sheet1:
            print(f"[WARN] Tab {tab!r} not found; using {spreadsheet.sheet1.title!r}. tabs={names}")
            return spreadsheet.sheet1
        print(f"[WARN] No worksheets in queue spreadsheet {sid}.")
        return None
    except Exception as e:
        sa = (GOOGLE_CLOUD_CREDENTIALS or {}).get("client_email", "the service account")
        print(f"[WARN] Could not open links queue sheet {sid}: {e}")
        print(f"  Share that spreadsheet with {sa} (Editor).")
        return None


def normalize_queue_post_date(raw):
    """Sheet C is often '2026-09-25 16:00:06'; LLM prompt expects 'September 25, 2026'."""
    s = (raw or "").strip()
    if not s:
        return None
    try:
        datetime.strptime(s, "%B %d, %Y")
        return s
    except ValueError:
        pass
    for fmt in (
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%m/%d/%Y %H:%M:%S",
        "%m/%d/%Y",
    ):
        try:
            sample = s[:19] if fmt.startswith("%Y-%m-%d %H:%M:%S") else s
            dt = datetime.strptime(sample, fmt)
            return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except ValueError:
            continue
    return s


def _ytdlp_post_date(info):
    """yt-dlp timestamp or upload_date → 'September 25, 2026'."""
    ts = info.get("timestamp") if info else None
    if ts not in (None, ""):
        try:
            dt = datetime.fromtimestamp(int(ts))
            return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except (TypeError, ValueError, OSError):
            pass
    upload_date = str((info or {}).get("upload_date") or "").strip()
    if len(upload_date) == 8 and upload_date.isdigit():
        try:
            dt = datetime.strptime(upload_date, "%Y%m%d")
            return f"{dt.strftime('%B')} {dt.day}, {dt.year}"
        except ValueError:
            pass
    return ""


def _ytdlp_photo_url(info):
    """Pick the best image URL from yt-dlp thumbnail / thumbnails. No download."""
    info = info or {}
    candidates = []
    top = (info.get("thumbnail") or "").strip()
    if top.startswith("http"):
        candidates.append((0, top))
    for t in info.get("thumbnails") or []:
        if not isinstance(t, dict):
            continue
        u = (t.get("url") or "").strip()
        if not u.startswith("http"):
            continue
        try:
            w = int(t.get("width") or 0)
            h = int(t.get("height") or 0)
        except (TypeError, ValueError):
            w, h = 0, 0
        candidates.append((w * h, u))
    if not candidates:
        return ""
    sized = [c for c in candidates if c[0] > 0]
    if sized:
        return max(sized, key=lambda c: c[0])[1]
    return candidates[-1][1]


def _ytdlp_username(info):
    """Prefer the IG handle (channel) over a numeric uploader_id."""
    info = info or {}
    channel = str(info.get("channel") or "").strip().lstrip("@")
    uploader_id = str(info.get("uploader_id") or "").strip().lstrip("@")
    uploader = str(info.get("uploader") or "").strip().lstrip("@")
    if channel:
        return channel
    if uploader_id and not uploader_id.isdigit():
        return uploader_id
    return uploader_id or uploader


def fetch_ig_post_via_ytdlp(post_url):
    """Fetch Instagram caption, photo URL, and username/date with yt-dlp.

    Does not download media. Photo is the CDN thumbnail URL as-is.
    Returns a dict or None if extraction fails.
    """
    if yt_dlp is None:
        print("  [WARN] yt-dlp is not installed. Run: pip install yt-dlp")
        return None
    ydl_opts = {
        "skip_download": True,
        "ignore_no_formats_error": True,
        "quiet": True,
        "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(post_url, download=False)
    except Exception as e:
        print(f"  [WARN] yt-dlp failed for {post_url[:70]}: {e}")
        return None
    if not info:
        return None
    if info.get("_type") in {"playlist", "multi_video"}:
        entries = [e for e in (info.get("entries") or []) if e]
        if entries:
            info = entries[0]
    caption = (info.get("description") or info.get("title") or "").strip()
    link = info.get("webpage_url") or post_url
    return {
        "caption": caption,
        "url": ig_url_from_text(link) or post_url,
        "username": _ytdlp_username(info),
        "post_date": _ytdlp_post_date(info) or None,
        "photo_url": _ytdlp_photo_url(info),
    }


def ig_url_from_text(raw):
    """Normalized /p/ or /reel/ URL, or empty."""
    text = (raw or "").strip()
    if not text:
        return ""
    m = _IG_POST_URL_RE.search(text)
    url = normalize_post_url(m.group(0) if m else text)
    if "/p/" not in url and "/reel/" not in url:
        return ""
    return url


def load_write_sheet_links(*sheets):
    """Temp set of Instagram links already stored on the write sheet (column G)."""
    known = set()
    for sheet in sheets:
        if sheet is None:
            continue
        title = getattr(sheet, "title", "?")
        try:
            col = sheet.col_values(WRITE_SHEET_LINK_COL)
        except Exception as e:
            print(f"[WARN] Could not read Link column from {title!r}: {e}")
            continue
        n = 0
        for cell in col[1:]:
            key = ig_post_key(cell)
            if key and key not in known:
                known.add(key)
                n += 1
        print(f"  Write sheet {title!r}: {n} existing Instagram link(s).")
    return known


def parse_read_sheet_row(row, row_number):
    """Parse one read-sheet row. Returns 'header', None (skip), or a post dict.

    Caption is not read from the sheet; process_one_post fetches it with yt-dlp.
    """
    raw = (row[0] if len(row) > 0 else "").strip()
    username = (row[1] if len(row) > 1 else "").strip().lstrip("@")
    post_date_raw = (row[2] if len(row) > 2 else "").strip()
    if row_number == 1 and raw.lower() in {"url", "link", "links", "instagram"}:
        return "header"
    url = ig_url_from_text(raw)
    if not url:
        return None
    return {
        "row": row_number,
        "url": url,
        "username": username,
        "post_date": normalize_queue_post_date(post_date_raw),
    }


def delete_read_sheet_row(sheet, row_index):
    """Delete one row from the read sheet. Returns 1 on success, 0 on failure."""
    if sheet is None or not row_index or row_index < 1:
        return 0
    try:
        sheet.delete_rows(row_index)
        return 1
    except Exception as e:
        print(f"  [WARN] Could not delete read-sheet row {row_index}: {e}")
        return 0


def delete_queue_url(sheet, url, url_col=SAVED_LINKS_URL_COL):
    """Delete every row whose URL matches (normalized). Returns number of rows removed."""
    if sheet is None or not url:
        return 0
    norm = normalize_post_url(url)
    try:
        col = sheet.col_values(url_col)
    except Exception as e:
        print(f"  [WARN] Could not re-read queue to delete {url}: {e}")
        return 0
    rows = [
        i
        for i, cell in enumerate(col, start=1)
        if normalize_post_url(cell) == norm
    ]
    if not rows:
        return 0
    try:
        for i in reversed(rows):
            sheet.delete_rows(i)
        return len(rows)
    except Exception as e:
        print(f"  [WARN] Could not delete queue row(s) {rows} for {url}: {e}")
        return 0


def find_first_empty_row(sheet):
    """Next append row = one past the last non-empty Event name (col C)."""
    try:
        col = sheet.col_values(3)  # Event name
        if not col or len(col) <= 1:
            col = sheet.col_values(1)  # Available fallback
        last = 1  # header
        for i, val in enumerate(col, start=1):
            if i == 1:
                continue
            if str(val).strip():
                last = i
        return last + 1
    except Exception:
        return 2


def _is_na_field(value) -> bool:
    s = str(value or "").strip().upper()
    return s in {"", "N/A", "NA", "NONE", "NULL"}


def needs_manual_review(event_dict) -> bool:
    """True if Event name or Date is missing/N/A — keep on input for fact-check."""
    return _is_na_field(event_dict.get("Event name")) or _is_na_field(
        event_dict.get("Date")
    )


def write_event_to_sheet(sheet, current_row, event_dict):
    r = current_row
    # Clamp ongoing ranges/lists so Available (I >= TODAY) stays Y when still current
    date_str = normalize_date_for_sheet(event_dict.get("Date", ""))
    row_data = [
        sheet_available_formula(r),
        event_dict.get("Cost", ""),
        event_dict.get("Event name", ""),
        event_dict.get("Category", ""),
        event_dict.get("Category(2)", ""),
        event_dict.get("Organizer", ""),
        event_dict.get("Link", ""),
        date_str,
        sheet_beginning_date_formula(r),
        event_dict.get("placeholder2", ""),
        event_dict.get("Time", ""),
        event_dict.get("Location", ""),
        event_dict.get("Photo", ""),
        event_dict.get("Area", ""),
        event_dict.get("code", ""),
    ]
    sheet.update(
        f"A{r}:O{r}",
        [row_data],
        value_input_option="USER_ENTERED",
    )
    return sheet, current_row + 1


def write_event_routed(
    event_dict,
    input_sheet,
    input_row,
    event_sheet,
    event_row,
):
    """
    Route: N/A name/date -> input (manual check); else -> Event(new).
    Returns (input_sheet, input_row, event_sheet, event_row, dest_label|None).
    """
    key_fields = ["Event name", "Date", "Time", "Location"]
    if not any(
        event_dict.get(f) and str(event_dict.get(f)).strip() for f in key_fields
    ):
        return input_sheet, input_row, event_sheet, event_row, None

    if needs_manual_review(event_dict):
        if input_sheet is None:
            return input_sheet, input_row, event_sheet, event_row, None
        input_sheet, input_row = write_event_to_sheet(
            input_sheet, input_row, event_dict
        )
        return input_sheet, input_row, event_sheet, event_row, INPUT_SHEET_TAB

    if event_sheet is None:
        # Fallback to input if Event tab unavailable
        if input_sheet is None:
            return input_sheet, input_row, event_sheet, event_row, None
        input_sheet, input_row = write_event_to_sheet(
            input_sheet, input_row, event_dict
        )
        return input_sheet, input_row, event_sheet, event_row, f"{INPUT_SHEET_TAB}(fallback)"

    event_sheet, event_row = write_event_to_sheet(event_sheet, event_row, event_dict)
    return input_sheet, input_row, event_sheet, event_row, EVENT_SHEET_TAB


def build_event_info(username, response, tags, category, url, photo_url=None):
    """Build list of event dicts for sheet. Photo is a URL string or N/A."""
    photo = (photo_url or "").strip() or "N/A"
    events = []
    for event in response:
        events.append({
            "Available": "Y",
            "Cost": event.get("cost", ""),
            "Event name": event.get("event_name", ""),
            "Category": tags,
            "Category(2)": category,
            "Organizer": username,
            "Link": url,
            "Date": event.get("date", ""),
            "placeholder1": "",
            "placeholder2": "",
            "Time": event.get("time", ""),
            "Location": event.get("venue", ""),
            "Photo": photo,
            "Area": "N/A",
            "code": f"{uuid.uuid4()}-{int(time.time())}",
        })
    return events


def configure_driver(driver, page_load_timeout=45):
    driver.set_page_load_timeout(page_load_timeout)
    driver.set_script_timeout(30)


def safe_driver_get(driver, url, wait_after=1.5):
    """Navigate with timeout; stop loading if Instagram hangs."""
    try:
        driver.get(url)
    except TimeoutException:
        print(f"[WARN] Page load timeout for {url}; stopping load and continuing.")
        try:
            driver.execute_script("window.stop();")
        except Exception:
            pass
    if wait_after:
        time.sleep(wait_after)


def create_driver(headless=True):
    # DISPLAY is Linux/X11 only; Windows and macOS headed Chrome do not need it.
    if (
        not headless
        and sys.platform not in ("win32", "darwin")
        and not os.environ.get("DISPLAY")
    ):
        print(
            "Non-headless Chrome needs a GUI display (DISPLAY is unset).\n"
            "On a plain SSH droplet you cannot interactively log in this way.\n"
            "  • Recommended: run --save-cookies on your PC, then scp cookies.pkl here.\n"
            "  • Or on the VM only: sudo apt install -y xvfb\n"
            "      xvfb-run -a python process_ig_read_sheet.py --save-cookies --no-headless\n"
            "    (virtual framebuffer; still awkward for 2FA — PC is easier.)\n",
            file=sys.stderr,
        )
        sys.exit(1)

    options = Options()
    options.page_load_strategy = "eager"
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-gpu")
    options.add_argument("--window-size=1920,1080")
    options.add_argument(
        "--user-agent="
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    )
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    configure_driver(driver)
    return driver


_SELENIUM_COOKIE_KEYS = {
    "name",
    "value",
    "domain",
    "path",
    "expiry",
    "secure",
    "httpOnly",
    "sameSite",
}


def cookie_for_selenium(cookie):
    """Drop fields Chrome rejects; skip expired cookies. Returns None if unusable."""
    if not cookie or not cookie.get("name") or cookie.get("value") is None:
        return None
    c = {k: cookie[k] for k in _SELENIUM_COOKIE_KEYS if k in cookie}
    if "expiry" in c:
        try:
            exp = int(float(c["expiry"]))
            if exp < time.time() - 60:
                return None
            c["expiry"] = exp
        except (TypeError, ValueError):
            c.pop("expiry", None)
    same = c.get("sameSite")
    if same is None or str(same).lower() in ("unspecified", ""):
        c.pop("sameSite", None)
    elif str(same).lower() in ("none", "no_restriction"):
        c["sameSite"] = "None"
        c["secure"] = True
    elif str(same).lower() == "lax":
        c["sameSite"] = "Lax"
    elif str(same).lower() == "strict":
        c["sameSite"] = "Strict"
    else:
        c.pop("sameSite", None)
    # Chrome/Selenium reject cookie values with commas (Instagram `rur` often has them).
    if "," in str(c.get("value") or ""):
        return None
    return c


def load_cookies(driver):
    """Add cookies.pkl on instagram.com, then reload so the session actually applies."""
    if not os.path.exists(COOKIES_FILE):
        return False
    safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    try:
        with open(COOKIES_FILE, "rb") as f:
            cookies = pickle.load(f)
        added = []
        skipped = []
        sessionid_ok = False
        for cookie in cookies:
            prepared = cookie_for_selenium(cookie)
            if not prepared:
                skipped.append((cookie or {}).get("name") or "?")
                continue
            try:
                driver.add_cookie(prepared)
                added.append(prepared["name"])
                if prepared["name"] == "sessionid":
                    sessionid_ok = True
            except Exception:
                skipped.append(prepared["name"])
        saved_at = time.strftime(
            "%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(COOKIES_FILE))
        )
        print(f"Loaded {len(added)} cookie(s) from {COOKIES_FILE} (saved {saved_at}).")
        if skipped:
            print(f"  Skipped cookie(s): {', '.join(skipped)}")
        if not sessionid_ok:
            print(
                "[WARN] sessionid was not applied; Instagram will treat this as logged out."
            )
        # Reload so Instagram JS sees the session (do not jump straight to /saved/).
        safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
        dismiss_instagram_popups(
            driver, allow_escape=not looks_like_account_chooser(driver)
        )
        return True
    except Exception as e:
        print(f"Warning: could not load cookies: {e}")
        return False


def _el_displayed(el):
    try:
        return el.is_displayed()
    except Exception:
        return False


def has_visible_login_form(driver):
    """Visible username/password fields only (hidden inputs exist on every IG page)."""
    try:
        if any(
            _el_displayed(el)
            for el in driver.find_elements(By.CSS_SELECTOR, "input[type='password']")
        ):
            return True
        for name in ("password", "username", "email"):
            if any(_el_displayed(el) for el in driver.find_elements(By.NAME, name)):
                return True
    except Exception:
        pass
    return False


def has_logged_in_nav(driver):
    """Left/bottom app chrome that only exists after a real login."""
    labels = (
        "Home",
        "New post",
        "Create",
        "Notifications",
        "Messages",
        "Reels",
        "Search and explore",
        "Profile",
        "首頁",
        "首页",
        "建立",
        "通知",
        "訊息",
        "消息",
        "探索",
    )
    try:
        return bool(
            driver.execute_script(
                """
                const want = new Set(arguments[0].map((s) => s.toLowerCase()));
                for (const n of document.querySelectorAll('[aria-label]')) {
                    const v = (n.getAttribute('aria-label') || '').trim().toLowerCase();
                    if (want.has(v)) return true;
                }
                return !!document.querySelector('svg[aria-label="Home"], svg[aria-label="首頁"]');
                """,
                list(labels),
            )
        )
    except Exception:
        pass
    for label in labels:
        try:
            if any(
                _el_displayed(el)
                for el in driver.find_elements(
                    By.CSS_SELECTOR, f"[aria-label='{label}']"
                )
            ):
                return True
        except Exception:
            continue
    return False


def is_logged_in(driver):
    """True if the current page looks like an authenticated Instagram session."""
    url = (driver.current_url or "").lower()
    if any(p in url for p in ("/challenge/", "/accounts/suspended")):
        return False
    if has_visible_login_form(driver):
        return False
    if has_logged_in_nav(driver) or current_nav_username(driver):
        return True
    return False


def _session_is_ready(driver, username=None):
    """True when we have an authenticated feed, not a login wall or account picker."""
    if has_visible_login_form(driver):
        return False
    if looks_like_account_chooser(driver) and not has_logged_in_nav(driver):
        return False
    if username and is_current_account(driver, username):
        return True
    return is_logged_in(driver)


def ensure_logged_in(driver, interactive=False, username=None, wait_seconds=180):
    """
    Confirm session after cookies. If logged out, watch the browser for login and
    click `username` on Instagram's account list. Does not block on Enter first.
    """
    username = (username or "").strip().lstrip("@") or None
    if _session_is_ready(driver, username):
        return True
    print(
        "Instagram session is not logged in "
        "(cookies expired, rejected, or Instagram requires a fresh login)."
    )
    if not interactive:
        print(
            "Re-run with a visible browser and log in:\n"
            "  python process_ig_read_sheet.py --save-cookies --no-headless"
        )
        return False

    print("Watching the browser (no need to press Enter)...")
    print("  1. Finish login / 2FA in Chrome.")
    if username:
        print(
            f"  2. If several accounts appear, the script will click '{username}'. "
            "You can also click it yourself."
        )
    print("  3. Wait until that account's home feed is visible.")

    deadline = time.time() + max(30, int(wait_seconds))
    last_click = 0.0
    last_status = 0.0
    while time.time() < deadline:
        dismiss_instagram_popups(
            driver, rounds=1, allow_escape=not looks_like_account_chooser(driver)
        )
        on_feed = has_logged_in_nav(driver)
        if (
            username
            and not on_feed
            and time.time() - last_click > 4
            and (
                looks_like_account_chooser(driver)
                or username_chip_visible(driver, username)
            )
        ):
            print(f"Account list detected; clicking '{username}'...")
            clicked = click_listed_account(driver, username)
            print("  Clicked." if clicked else "  Could not click the username chip.")
            last_click = time.time()
            time.sleep(2)
        if _session_is_ready(driver, username):
            save_cookies(driver)
            print("Login looks good; cookies saved.")
            return True
        now = time.time()
        if now - last_status > 15:
            print(
                f"  Still waiting... ({driver.current_url}) "
                f"login_form={has_visible_login_form(driver)} "
                f"chooser={looks_like_account_chooser(driver)} "
                f"nav={has_logged_in_nav(driver)} "
                f"profile={current_nav_username(driver) or '-'} "
                f"chip={username_chip_visible(driver, username) if username else '-'}"
            )
            last_status = now
        time.sleep(1.5)

    print(
        "Timed out waiting for the feed. If you are logged in now, "
        "press Enter in this terminal (not in Chrome)."
    )
    try:
        input()
    except EOFError:
        return False
    save_cookies(driver)
    safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    dismiss_instagram_popups(
        driver, allow_escape=not looks_like_account_chooser(driver)
    )
    if username:
        click_listed_account(driver, username)
        time.sleep(2)
    if _session_is_ready(driver, username):
        return True
    print("Still not logged in. Make sure the home feed is visible, then try again.")
    return False


def save_cookies(driver):
    cookies = driver.get_cookies()
    with open(COOKIES_FILE, "wb") as f:
        pickle.dump(cookies, f)
    print(f"Saved {len(cookies)} cookies to {COOKIES_FILE}")


_NON_PROFILE_PATHS = {
    "p",
    "reel",
    "reels",
    "stories",
    "accounts",
    "explore",
    "direct",
    "saved",
    "legal",
    "about",
    "privacy",
    "emailsignup",
    "challenge",
}

_ACCOUNT_CHOOSER_TEXT = (
    "choose an account",
    "choose your account",
    "continue as",
    "switch accounts",
    "switch account",
    "logged in as",
    "log in as",
    "recent logins",
    "saved accounts",
    "which account",
    "選擇帳號",
    "選擇帳戶",
    "選擇賬戶",
    "切換帳號",
    "切換帳戶",
    "切換账户",
    "繼續以",
    "继续以",
    "最近登入",
    "最近登录",
)

_SWITCH_MENU_TEXT = (
    "Switch accounts",
    "Switch account",
    "切換帳號",
    "切換帳戶",
    "切換账户",
)


def _js_click(driver, el):
    try:
        driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
        time.sleep(0.2)
        el.click()
        return True
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", el)
            return True
        except Exception:
            return False


def _username_from_profile_href(href):
    if not href:
        return ""
    m = re.search(r"instagram\.com/([A-Za-z0-9._]+)/?$", href.split("?")[0].rstrip("/"))
    if not m:
        m = re.search(r"^/([A-Za-z0-9._]+)/?$", href.split("?")[0])
    if not m:
        return ""
    user = m.group(1)
    if user.lower() in _NON_PROFILE_PATHS:
        return ""
    return user


def _visible_body_text(driver):
    try:
        return (driver.find_element(By.TAG_NAME, "body").text or "").lower()
    except Exception:
        return ""


def looks_like_account_chooser(driver):
    """Use visible text only — page_source always contains login copy in IG's JS."""
    text = _visible_body_text(driver)
    return any(s.lower() in text for s in _ACCOUNT_CHOOSER_TEXT)


def username_chip_visible(driver, username):
    target = (username or "").strip().lstrip("@")
    if not target:
        return False
    try:
        for el in driver.find_elements(
            By.XPATH,
            f"//*[normalize-space()='{target}' or normalize-space()='@{target}']",
        ):
            if _el_displayed(el):
                return True
    except Exception:
        pass
    return target.lower() in _visible_body_text(driver)


def current_nav_username(driver):
    """Username of the logged-in profile in the left/bottom nav, if detectable."""
    selectors = [
        "svg[aria-label='Profile']",
        "svg[aria-label='個人檔案']",
        "svg[aria-label='个人主页']",
        "img[alt$=\"profile picture\"]",
        "img[alt$='的個人檔案相片']",
    ]
    for sel in selectors:
        try:
            for el in driver.find_elements(By.CSS_SELECTOR, sel):
                try:
                    a = el.find_element(By.XPATH, "./ancestor::a[1]")
                except Exception:
                    continue
                user = _username_from_profile_href(a.get_attribute("href") or "")
                if user:
                    return user
        except Exception:
            continue
    return ""


def is_current_account(driver, username):
    target = (username or "").strip().lstrip("@").lower()
    if not target:
        return False
    return (current_nav_username(driver) or "").lower() == target


def click_listed_account(driver, username):
    """Click a username chip on Instagram's account chooser / switcher list."""
    target = (username or "").strip().lstrip("@")
    if not target:
        return False
    xpaths = [
        f"//button[contains(normalize-space(.), '{target}')]",
        f"//div[@role='button'][contains(normalize-space(.), '{target}')]",
        f"//a[contains(normalize-space(.), '{target}')]",
        f"//*[@role='link'][contains(normalize-space(.), '{target}')]",
        f"//span[normalize-space()='{target}']/ancestor::*[@role='button'][1]",
        f"//span[normalize-space()='{target}']/ancestor::button[1]",
        f"//span[normalize-space()='{target}']/ancestor::a[1]",
        f"//span[normalize-space()='{target}']/ancestor::div[@role='button'][1]",
        f"//span[contains(normalize-space(), '{target}')]/ancestor::*[@role='button' or self::button or self::a][1]",
        f"//*[normalize-space()='{target}']",
    ]
    seen = set()
    for xp in xpaths:
        try:
            els = driver.find_elements(By.XPATH, xp)
        except Exception:
            continue
        for el in els:
            try:
                if not el.is_displayed():
                    continue
                key = el.id
                if key in seen:
                    continue
                seen.add(key)
            except Exception:
                continue
            if _js_click(driver, el):
                return True
    try:
        return bool(
            driver.execute_script(
                """
                const target = arguments[0].toLowerCase();
                const nodes = Array.from(document.querySelectorAll(
                    'button, a, div[role="button"], [role="link"], span, li'
                ));
                for (const n of nodes) {
                    const t = (n.innerText || n.textContent || '').replace(/\\s+/g, ' ').trim();
                    if (!t || t.length > 48) continue;
                    const low = t.toLowerCase().replace(/^@/, '');
                    if (
                        low !== target
                        && low !== 'continue as ' + target
                        && !low.startsWith(target + ' ')
                        && !low.endsWith(' ' + target)
                    ) {
                        continue;
                    }
                    const style = window.getComputedStyle(n);
                    if (style.display === 'none' || style.visibility === 'hidden') continue;
                    n.scrollIntoView({block: 'center'});
                    n.click();
                    return true;
                }
                return false;
                """,
                target.lstrip("@").lower(),
            )
        )
    except Exception:
        return False


def _click_by_selectors(driver, selectors, timeout=4):
    wait = WebDriverWait(driver, timeout)
    for by, value in selectors:
        try:
            el = wait.until(EC.presence_of_element_located((by, value)))
            if el.is_displayed() and _js_click(driver, el):
                return True
        except Exception:
            continue
    return False


def switch_to_account(driver, target_username, interactive=False):
    """
    Select target_username after login.

    Handles Instagram's post-login account cards ("Choose an account") as well as
    More/Settings -> Switch accounts. Returns True if we are on that account, or
    if interactive login recovered the session.
    """
    target = (target_username or "").strip().lstrip("@")
    if not target:
        return True

    # If Instagram is already showing account cards, click before navigating away.
    if click_listed_account(driver, target):
        print(f"Clicked '{target}' on the account list.")
        time.sleep(3)
        dismiss_instagram_popups(driver, allow_escape=False)
        if is_current_account(driver, target) or (
            is_logged_in(driver) and not looks_like_account_chooser(driver)
        ):
            return True

    safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    dismiss_instagram_popups(driver, allow_escape=not looks_like_account_chooser(driver))

    if is_current_account(driver, target):
        print(f"Already on '{target}'.")
        return True

    if click_listed_account(driver, target):
        print(f"Clicked '{target}' on the account list.")
        time.sleep(3)
        dismiss_instagram_popups(driver, allow_escape=False)
        if is_current_account(driver, target) or (
            is_logged_in(driver) and not looks_like_account_chooser(driver)
        ):
            return True

    menu_opened = _click_by_selectors(
        driver,
        [
            (By.XPATH, "//span[normalize-space()='More']/ancestor::*[@role='link' or self::a][1]"),
            (By.XPATH, "//span[normalize-space()='Settings']/ancestor::*[@role='link' or self::a][1]"),
            (By.CSS_SELECTOR, "svg[aria-label='Settings']"),
            (By.CSS_SELECTOR, "[aria-label='Settings']"),
            (By.XPATH, "//span[normalize-space()='更多']"),
            (By.XPATH, "//span[normalize-space()='設定']"),
            (By.XPATH, "//span[normalize-space()='设置']"),
        ],
    )
    if menu_opened:
        print("Opened More/Settings menu.")
        time.sleep(1.5)
        switch_opened = _click_by_selectors(
            driver,
            [
                (By.LINK_TEXT, t) for t in _SWITCH_MENU_TEXT
            ]
            + [
                (By.PARTIAL_LINK_TEXT, t) for t in _SWITCH_MENU_TEXT
            ]
            + [
                (By.XPATH, f"//*[contains(normalize-space(.), '{t}')]")
                for t in _SWITCH_MENU_TEXT
            ]
            + [
                (By.XPATH, f"//span[normalize-space()='{t}']")
                for t in _SWITCH_MENU_TEXT
            ],
        )
        if switch_opened:
            print("Opened 'Switch accounts'.")
            time.sleep(2)
        if click_listed_account(driver, target):
            print(f"Switched to '{target}'.")
            time.sleep(3)
            if is_current_account(driver, target) or is_logged_in(driver):
                return True

    if is_current_account(driver, target):
        return True

    print(f"Could not auto-select '{target}' from the account list.")
    if not interactive:
        print(
            "Re-run with a visible browser and click that account:\n"
            "  python process_ig_read_sheet.py --no-headless"
        )
        return False

    print(
        f"In the browser: click '{target}' (or log into it), wait until that account's "
        "home feed is visible, then return here and press Enter."
    )
    try:
        input()
    except EOFError:
        return False
    save_cookies(driver)
    safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    dismiss_instagram_popups(driver, allow_escape=not looks_like_account_chooser(driver))
    if is_current_account(driver, target) or (
        is_logged_in(driver) and not looks_like_account_chooser(driver)
    ):
        print(f"Continuing as '{target}'.")
        return True
    print(f"Still not on '{target}'. Saved posts need that account's login.")
    return False


_DISMISS_BUTTON_TEXTS = (
    "Not Now",
    "Not now",
    "Later",
    "No thanks",
    "Maybe Later",
    "Decline",
    "Cancel",
    "Skip",
    "稍後再說",
    "暫時不要",
    "稍後",
    "以後再說",
    "不用了",
    "取消",
    "跳過",
)


def dismiss_instagram_popups(driver, rounds=3, allow_escape=True):
    """Dismiss Instagram overlays (notifications, save-login, cookies, etc.)."""
    for _ in range(rounds):
        dismissed = False
        # ESC can close the account picker before we click xplore.hk.
        if allow_escape and not looks_like_account_chooser(driver):
            try:
                driver.find_element(By.TAG_NAME, "body").send_keys(Keys.ESCAPE)
                time.sleep(0.3)
            except Exception:
                pass

        for text in _DISMISS_BUTTON_TEXTS:
            try:
                xpath = (
                    f"//button[contains(normalize-space(.), '{text}')]"
                    f"|//div[@role='button'][contains(normalize-space(.), '{text}')]"
                    f"|//span[contains(normalize-space(.), '{text}')]/ancestor::button[1]"
                    f"|//span[contains(normalize-space(.), '{text}')]/ancestor::div[@role='button'][1]"
                )
                for btn in driver.find_elements(By.XPATH, xpath):
                    if btn.is_displayed():
                        btn.click()
                        time.sleep(0.8)
                        dismissed = True
                        break
            except Exception:
                pass
            if dismissed:
                break

        if not dismissed:
            for label in ("Close", "關閉", "Dismiss"):
                try:
                    for btn in driver.find_elements(
                        By.XPATH,
                        f"//button[@aria-label='{label}']"
                        f"|//div[@role='button'][@aria-label='{label}']",
                    ):
                        if btn.is_displayed():
                            btn.click()
                            time.sleep(0.8)
                            dismissed = True
                            break
                except Exception:
                    pass
                if dismissed:
                    break

        if not dismissed:
            break
        time.sleep(0.5)


def go_to_saved_posts_grid(driver, username, url=None):
    """
    Go to saved posts grid: open /saved/all-posts/ (or click 'All posts' on /saved/).
    """
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    if url and "all-posts" in url:
        safe_driver_get(driver, url, wait_after=3)
    else:
        # Direct grid URL is more reliable than click-through
        safe_driver_get(
            driver,
            f"https://www.instagram.com/{username}/saved/all-posts/",
            wait_after=3,
        )
    dismiss_instagram_popups(driver)

    if "all-posts" not in (driver.current_url or ""):
        safe_driver_get(
            driver,
            f"https://www.instagram.com/{username}/saved/",
            wait_after=3,
        )
        for selector in [
            (By.LINK_TEXT, "All posts"),
            (By.PARTIAL_LINK_TEXT, "All posts"),
            (By.XPATH, "//span[text()='All posts']/ancestor::a"),
            (By.XPATH, "//*[contains(text(), 'All posts')]/ancestor::a"),
            (By.XPATH, "//a[contains(@href, 'all-posts')]"),
            (By.LINK_TEXT, "All"),
            (By.PARTIAL_LINK_TEXT, "All"),
        ]:
            try:
                el = wait.until(EC.element_to_be_clickable((selector[0], selector[1])))
                el.click()
                print("Clicked 'All posts'.")
                break
            except Exception:
                continue
        time.sleep(3)

    # Wait until the grid has at least one post link (or timeout)
    try:
        wait.until(
            EC.presence_of_element_located(
                (By.XPATH, "//a[contains(@href, '/p/') or contains(@href, '/reel/')]")
            )
        )
    except Exception:
        print(
            f"[WARN] No post links visible yet on {driver.current_url}. "
            "Check you are logged in as the account that owns these saved posts."
        )
    time.sleep(2)
    return True


def extract_post_links(driver, scroll_pauses=3, max_scrolls=50):
    """Scroll the saved posts page and collect all /p/ and /reel/ links."""
    seen = set()
    links = []
    last_count = 0
    no_new_count = 0

    def collect():
        nonlocal links, seen
        elements = driver.find_elements(
            By.XPATH,
            "//main//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
        )
        if not elements:
            elements = driver.find_elements(
                By.XPATH,
                "//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
            )
        for el in elements:
            try:
                href = el.get_attribute("href")
                if not href:
                    continue
                base = href.split("?")[0].rstrip("/")
                if "/liked_by" in base or "/comments" in base:
                    continue
                if base not in seen:
                    seen.add(base)
                    links.append(href.split("?")[0])
            except Exception:
                pass

    collect()
    if not links:
        time.sleep(5)
        collect()

    for _ in range(max_scrolls):
        if len(links) == last_count:
            no_new_count += 1
            if no_new_count >= 3:
                break
        else:
            no_new_count = 0
        last_count = len(links)

        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pauses)
        collect()

    return links


def process_one_post(post, driver, input_sheet, input_row, event_sheet, event_row):
    """Extract and write one post. status is 'done' or 'retry'.

    Returns (status, input_sheet, input_row, event_sheet, event_row, written_input, written_event).
    """
    if isinstance(post, str):
        post = {"url": post}
    url = post.get("url") or ""
    username = (post.get("username") or "").strip().lstrip("@")
    details_str = (post.get("caption") or "").strip()
    post_date = post.get("post_date")
    photo_url = ""
    written_input = 0
    written_event = 0

    time.sleep(DELAY_BETWEEN_POSTS)

    if driver is None:
        meta = fetch_ig_post_via_ytdlp(url)
        if not meta:
            print(f"  [retry later] yt-dlp failed: {url[:70]}")
            return "retry", input_sheet, input_row, event_sheet, event_row, 0, 0
        details_str = (meta.get("caption") or "").strip()
        if meta.get("url"):
            url = meta["url"]
        if meta.get("username"):
            username = meta["username"]
        if meta.get("post_date"):
            post_date = meta["post_date"]
        photo_url = (meta.get("photo_url") or "").strip()
        print(
            f"  [yt-dlp] {len(details_str)} chars"
            f"{'  @' + username if username else ''}"
            f"{'  ' + str(post_date) if post_date else ''}"
            f"{'  photo' if photo_url else '  no photo'}"
        )
        if not details_str:
            print(f"  [retry later] yt-dlp returned no caption: {url[:70]}")
            return "retry", input_sheet, input_row, event_sheet, event_row, 0, 0
    elif not details_str:
        soup = get_content_sync(url, driver=driver)
        if not soup:
            print(f"  [retry later] fetch failed: {url[:70]}")
            return "retry", input_sheet, input_row, event_sheet, event_row, 0, 0
        fetched_user, details, fetched_date = extract_username_and_details(soup)
        details_str = (details or "").strip()
        if not username:
            username = (fetched_user or "").strip()
        if not post_date:
            post_date = fetched_date

    if not details_str:
        print(f"  [skip] no caption: {url[:70]}")
        return "done", input_sheet, input_row, event_sheet, event_row, 0, 0

    try:
        response, tags, category = extract_info_with_model_fallback(
            details_str, post_date=post_date
        )
    except Exception as e:
        print(f"  [WARN] extract_info (all models) failed for {url}: {e}")
        return "retry", input_sheet, input_row, event_sheet, event_row, 0, 0

    if driver is not None and not photo_url:
        photo_url = extract_photo(url)
    event_list = build_event_info(
        username, response, tags, category, url, photo_url=photo_url
    )
    write_failed = False
    for event_dict in event_list:
        try:
            (
                input_sheet,
                input_row,
                event_sheet,
                event_row,
                dest,
            ) = write_event_routed(
                event_dict,
                input_sheet,
                input_row,
                event_sheet,
                event_row,
            )
            if dest == INPUT_SHEET_TAB or (
                dest and dest.startswith(INPUT_SHEET_TAB)
            ):
                written_input += 1
                print(
                    f"    -> {dest}: "
                    f"{event_dict.get('Event name', '')!r} / "
                    f"{event_dict.get('Date', '')!r}"
                )
            elif dest == EVENT_SHEET_TAB:
                written_event += 1
                print(
                    f"    -> {dest}: "
                    f"{event_dict.get('Event name', '')!r} / "
                    f"{event_dict.get('Date', '')!r}"
                )
        except Exception as e:
            print(f"  [WARN] write failed: {e}")
            write_failed = True
    if write_failed:
        print(f"  Left in read sheet after write error: {url[:60]}...")
        return (
            "retry",
            input_sheet,
            input_row,
            event_sheet,
            event_row,
            written_input,
            written_event,
        )
    return (
        "done",
        input_sheet,
        input_row,
        event_sheet,
        event_row,
        written_input,
        written_event,
    )


def process_read_sheet(
    read_sheet,
    known_links,
    input_sheet,
    input_row,
    event_sheet,
    event_row,
    keep_queue=False,
    dry_run=False,
):
    """Walk the read sheet top to bottom.

    If the link is already in known_links (write sheet or seen earlier this run),
    delete that read-sheet row without extracting. Otherwise add it, extract, then
    delete.
    """
    written_input = 0
    written_event = 0
    deleted_queue = 0
    skipped_dup = 0
    retried = 0

    try:
        vals = read_sheet.get_all_values()
    except Exception as e:
        print(f"[WARN] Could not read the read sheet: {e}")
        return {
            "written_input": 0,
            "written_event": 0,
            "deleted_queue": 0,
            "skipped_dup": 0,
            "retried": 0,
        }

    i = 0
    while i < len(vals):
        row_number = i + 1
        parsed = parse_read_sheet_row(vals[i], row_number)
        if parsed == "header" or parsed is None:
            i += 1
            continue

        url = parsed["url"]
        key = ig_post_key(url)
        extra = ""
        if parsed.get("username"):
            extra += f"  @{parsed['username']}"
        if parsed.get("post_date"):
            extra += f"  {parsed['post_date']}"

        if key and key in known_links:
            print(f"  [dup] already processed — delete row {row_number}:{extra} {url}")
            skipped_dup += 1
            if dry_run or keep_queue:
                i += 1
                continue
            if delete_read_sheet_row(read_sheet, row_number):
                deleted_queue += 1
                vals.pop(i)
            else:
                i += 1
            continue

        print(f"  [new]{extra} {url}")
        if key:
            known_links.add(key)
        if dry_run:
            i += 1
            continue

        (
            status,
            input_sheet,
            input_row,
            event_sheet,
            event_row,
            w_in,
            w_ev,
        ) = process_one_post(
            parsed, None, input_sheet, input_row, event_sheet, event_row
        )
        written_input += w_in
        written_event += w_ev
        if status == "retry":
            retried += 1
            i += 1
            continue

        if keep_queue:
            i += 1
            continue
        if delete_read_sheet_row(read_sheet, row_number):
            deleted_queue += 1
            vals.pop(i)
        else:
            i += 1

    return {
        "written_input": written_input,
        "written_event": written_event,
        "deleted_queue": deleted_queue,
        "skipped_dup": skipped_dup,
        "retried": retried,
    }


def process_post_urls(
    posts,
    driver,
    input_sheet,
    input_row,
    event_sheet,
    event_row,
    tracker_path=None,
    processed_urls=None,
    queue_sheet=None,
):
    """Extract each post (--from-ig-saved). Returns write counts."""
    if processed_urls is None:
        processed_urls = set()
    written_input = 0
    written_event = 0
    deleted_queue = 0
    retried = 0

    for i, post in enumerate(posts):
        if isinstance(post, str):
            post = {"url": post}
        url = post.get("url") or ""
        norm = normalize_post_url(url)
        (
            status,
            input_sheet,
            input_row,
            event_sheet,
            event_row,
            w_in,
            w_ev,
        ) = process_one_post(
            post, driver, input_sheet, input_row, event_sheet, event_row
        )
        written_input += w_in
        written_event += w_ev
        if status == "retry":
            retried += 1
            continue
        if tracker_path:
            append_processed_link(tracker_path, norm, processed_urls)
        if queue_sheet is not None:
            n = delete_queue_url(queue_sheet, url)
            deleted_queue += n
            print(f"  Removed {n} row(s) from queue sheet.")
        print(f"  Processed {i + 1}/{len(posts)}: {url[:60]}...")

    return {
        "written_input": written_input,
        "written_event": written_event,
        "deleted_queue": deleted_queue,
        "retried": retried,
    }


def login_instagram(driver, interactive, switch_account=None):
    loaded = load_cookies(driver)
    if not loaded and not interactive:
        print(
            f"No {COOKIES_FILE} found. Run with --save-cookies first to log in and create it."
        )
        return False
    if not loaded:
        safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    if not ensure_logged_in(driver, interactive=interactive, username=switch_account):
        return False
    if switch_account:
        if not switch_to_account(driver, switch_account, interactive=interactive):
            print(
                "Account switch failed. Click that account in the browser if it is visible, "
                "or re-run: python process_ig_read_sheet.py --no-headless"
            )
            return False
        time.sleep(2)
    return True


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read sheet -> write sheet: fetch captions with yt-dlp, skip duplicates, "
            "extract new ones, then delete the read-sheet row."
        )
    )
    parser.add_argument(
        "--save-cookies",
        action="store_true",
        help="Open browser for you to log in; then save cookies to cookies.pkl and exit.",
    )
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Run browser visible (useful for debugging or first-time login).",
    )
    parser.add_argument(
        "--username",
        default=DEFAULT_USERNAME,
        help=f"Account whose saved posts to open (e.g. xplore.hk). Default: {DEFAULT_USERNAME}",
    )
    parser.add_argument(
        "--switch-account",
        metavar="USERNAME",
        default=None,
        help="Optional: switch to this Instagram account after login (queue mode does not require it).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Go directly to this URL instead of clicking Saved -> All posts.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Walk the read sheet and print dup/new only; do not extract, write, or delete.",
    )
    parser.add_argument(
        "--from-ig-saved",
        action="store_true",
        help="Crawl Instagram saved posts instead of reading the links queue spreadsheet.",
    )
    parser.add_argument(
        "--links-sheet",
        default=SAVED_LINKS_SHEET_ID,
        metavar="ID",
        help="Google Sheet ID for the read sheet (queue). Default: SAVED_LINKS_SHEET_ID.",
    )
    parser.add_argument(
        "--links-tab",
        default=SAVED_LINKS_SHEET_TAB,
        help=f"Read-sheet worksheet name. Default: {SAVED_LINKS_SHEET_TAB}.",
    )
    parser.add_argument(
        "--keep-queue",
        action="store_true",
        help="Do not delete rows from the read sheet.",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_PROCESSED_TRACKER,
        metavar="PATH",
        help=(
            "Used only with --from-ig-saved. File of processed post URLs. "
            f"Default: {DEFAULT_PROCESSED_TRACKER}"
        ),
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="With --from-ig-saved, process every crawled URL even if it is in the tracker.",
    )
    args = parser.parse_args()

    headless = not args.no_headless and not args.save_cookies
    driver = None

    try:
        if args.save_cookies:
            driver = create_driver(headless=headless)
            driver.get(INSTAGRAM_HOME)
            print(
                "In the browser: finish login (password, 2FA, 'Save login info?' if shown).\n"
                "Wait until the home feed loads, then return here and press Enter to save cookies.pkl."
            )
            input()
            save_cookies(driver)
            return

        interactive = not headless
        tracker_path = None
        processed_urls = set()
        posts = []

        if args.from_ig_saved:
            driver = create_driver(headless=headless)
            target_account = args.switch_account or args.username
            if not login_instagram(
                driver, interactive, switch_account=target_account
            ):
                return
            go_to_saved_posts_grid(driver, args.username, url=args.url)
            if not is_logged_in(driver):
                print(
                    "Saved posts page is showing a login wall. "
                    "Log in in the browser if it is visible, or re-run with --save-cookies --no-headless."
                )
                return
            print("Scrolling to load saved posts...")
            print(f"  Page: {driver.current_url}")
            posts = [{"url": u} for u in extract_post_links(driver)]
            if not args.no_extract and not args.no_tracker:
                tracker_path = os.path.abspath(args.tracker)
                processed_urls = load_processed_links(tracker_path)
            if not posts:
                print("No Instagram post links to process.")
                return
            to_process = (
                [
                    p
                    for p in posts
                    if normalize_post_url(p.get("url")) not in processed_urls
                ]
                if tracker_path
                else posts
            )
            print()
            for p in posts:
                url = p.get("url") or ""
                tag = ""
                if processed_urls and normalize_post_url(url) in processed_urls:
                    tag = " [already processed]"
                print(f"{url}{tag}")
            if args.no_extract:
                return
            print("\nExtracting event info and writing to Google Sheet...")
            print(
                f"  [LLM] primary: {', '.join(_primary_models_from_env())} "
                f"-> fallback: {_fallback_model_from_env()}"
            )
            input_sheet, input_row, event_sheet, event_row = (
                init_input_and_event_sheets()
            )
            stats = process_post_urls(
                to_process,
                driver,
                input_sheet,
                input_row,
                event_sheet,
                event_row,
                tracker_path=tracker_path,
                processed_urls=processed_urls,
            )
            print(
                f"\nWrote {stats['written_event']} event(s) to {EVENT_SHEET_TAB}, "
                f"{stats['written_input']} to {INPUT_SHEET_TAB} (manual review)."
            )
            skipped_tracker = len(posts) - len(to_process)
            if skipped_tracker:
                print(f"Skipped {skipped_tracker} link(s) already in tracker.")
            return

        read_sheet = open_saved_links_sheet(args.links_sheet, args.links_tab)
        if read_sheet is None:
            return
        print(f"Read sheet: {args.links_sheet} / {read_sheet.title}")

        input_sheet, input_row, event_sheet, event_row = init_input_and_event_sheets()
        if input_sheet is None and event_sheet is None:
            print(
                "[WARN] Write sheet not available; cannot build the existing-link list."
            )
            return
        print(
            f"Write sheet: {INPUT_SHEET_TAB}="
            f"{'ok@'+str(input_row) if input_sheet else 'missing'}, "
            f"{EVENT_SHEET_TAB}="
            f"{'ok@'+str(event_row) if event_sheet else 'missing'}"
        )
        print("Loading existing Instagram links from the write sheet...")
        known_links = load_write_sheet_links(input_sheet, event_sheet)
        print(f"Temp list: {len(known_links)} unique link(s) already in the database.")

        if args.no_extract:
            print("\nDry run (no extract / no delete):")
        else:
            print("\nExtracting event info and writing to the write sheet...")
            print(
                f"  [LLM] primary: {', '.join(_primary_models_from_env())} "
                f"-> fallback: {_fallback_model_from_env()}"
            )
            print(
                "  Route: complete events -> "
                f"{EVENT_SHEET_TAB}; N/A name/date -> {INPUT_SHEET_TAB}"
            )

        stats = process_read_sheet(
            read_sheet,
            known_links,
            input_sheet,
            input_row,
            event_sheet,
            event_row,
            keep_queue=args.keep_queue,
            dry_run=args.no_extract,
        )
        print(
            f"\nWrote {stats['written_event']} event(s) to {EVENT_SHEET_TAB}, "
            f"{stats['written_input']} to {INPUT_SHEET_TAB} (manual review)."
        )
        if stats["skipped_dup"]:
            print(
                f"Skipped {stats['skipped_dup']} duplicate row(s) "
                "(already on the write sheet or seen earlier in the read sheet)."
            )
        if stats["deleted_queue"]:
            print(f"Deleted {stats['deleted_queue']} row(s) from the read sheet.")
        if stats["retried"]:
            print(f"Left {stats['retried']} link(s) on the read sheet to retry next run.")

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
