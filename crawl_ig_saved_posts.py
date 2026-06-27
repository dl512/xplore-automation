#!/usr/bin/env python3
"""
Crawl Instagram saved posts, output post links, and optionally extract event info to Google Sheet.

Uses Selenium + Chrome (same setup as xplore_automation.py). Requires login via cookies.
Cookies are read from cookies.pkl (create once by running with --save-cookies and logging in).

Usage:
  # First time: create cookies on a machine with a display (your PC), or on VM:
  #   xvfb-run -a python crawl_ig_saved_posts.py --save-cookies --no-headless
  # Then scp cookies.pkl to servers without a GUI.
  python crawl_ig_saved_posts.py --save-cookies

  # Crawl saved posts, extract event info from each post, append to Google Sheet (input tab).
  # Skips URLs in processed_ig_links.txt (default); appends each URL after it is processed.
  python crawl_ig_saved_posts.py

  # Only crawl and print links (no extraction / sheet)
  python crawl_ig_saved_posts.py --no-extract

  # Custom tracker file (.txt one URL per line, or .csv with header 'url')
  python crawl_ig_saved_posts.py --tracker /path/on/vm/processed_ig_links.csv

  # If you log in as primary but want saved posts for xplore.hk:
  python crawl_ig_saved_posts.py --switch-account xplore.hk --username xplore.hk

  LLM: tries GPT models (gpt-4.1-nano, gpt-4o-mini, gpt-3.5-turbo) in order; if each
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

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager

load_dotenv(override=True)

from extraction_details import (
    extract_info_with_model_fallback,
    _primary_models_from_env,
    _fallback_model_from_env,
)

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


def extract_photo(url):
    """Get post image via Instaloader, upload to GCS. Returns '' on failure (does not block sheet write)."""
    shortcode = extract_shortcode_from_url(url)
    if not shortcode:
        return ""
    try:
        import instaloader
        L = instaloader.Instaloader()
        _load_instaloader_session(L)
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        if post.url:
            return manage_photo(post.url) or ""
    except Exception:
        pass
    return ""

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


def init_google_sheets():
    """Initialize Google Sheets client and open the 'input' worksheet. Returns (sheet, current_row) or (None, None)."""
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
            sheet = spreadsheet.worksheet("input") if "input" in names else spreadsheet.sheet1
        except Exception:
            sheet = spreadsheet.sheet1
        row = find_first_empty_row(sheet)
        return sheet, row
    except Exception as e:
        print(f"[WARN] Google Sheets init failed: {e}")
        return None, None


def find_first_empty_row(sheet):
    try:
        col = sheet.col_values(1)
        for i in range(2, len(col) + 2):
            if i > len(col) or not col[i - 1]:
                return i
        return len(col) + 1
    except Exception:
        return 2


def write_event_to_sheet(sheet, current_row, event_dict):
    row_data = [
        event_dict.get("Available", ""),
        event_dict.get("Cost", ""),
        event_dict.get("Event name", ""),
        event_dict.get("Category", ""),
        event_dict.get("Category(2)", ""),
        event_dict.get("Organizer", ""),
        event_dict.get("Link", ""),
        event_dict.get("Date", ""),
        event_dict.get("placeholder1", ""),
        event_dict.get("placeholder2", ""),
        event_dict.get("Time", ""),
        event_dict.get("Location", ""),
        event_dict.get("Photo", ""),
        event_dict.get("Area", ""),
        event_dict.get("code", ""),
    ]
    sheet.update(f"A{current_row}:O{current_row}", [row_data])
    return sheet, current_row + 1


def build_event_info(username, response, tags, category, url, photo_url=None):
    """Build list of event dicts for sheet (same columns as xplore_automation). Photo from extract_photo or N/A."""
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
    if not headless and not os.environ.get("DISPLAY"):
        print(
            "Non-headless Chrome needs a GUI display (DISPLAY is unset).\n"
            "On a plain SSH droplet you cannot interactively log in this way.\n"
            "  • Recommended: run --save-cookies on your PC, then scp cookies.pkl here.\n"
            "  • Or on the VM only: sudo apt install -y xvfb\n"
            "      xvfb-run -a python crawl_ig_saved_posts.py --save-cookies --no-headless\n"
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
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    configure_driver(driver)
    return driver


def load_cookies(driver):
    safe_driver_get(driver, INSTAGRAM_HOME, wait_after=2)
    if not os.path.exists(COOKIES_FILE):
        return False
    try:
        with open(COOKIES_FILE, "rb") as f:
            cookies = pickle.load(f)
        for cookie in cookies:
            try:
                driver.add_cookie(cookie)
            except Exception:
                pass
        return True
    except Exception as e:
        print(f"Warning: could not load cookies: {e}")
        return False


def save_cookies(driver):
    cookies = driver.get_cookies()
    with open(COOKIES_FILE, "wb") as f:
        pickle.dump(cookies, f)
    print(f"Saved {len(cookies)} cookies to {COOKIES_FILE}")


def switch_to_account(driver, target_username):
    """
    After login you may be on your primary account. Switch to target_username via
    More -> Switch accounts -> [account]. Returns True if switch succeeded (or we're already there).
    Instagram may ask for the secondary account's password; we cannot fill that automatically.
    """
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    driver.get(INSTAGRAM_HOME)
    time.sleep(3)

    # 1) Open menu: "More" (bottom nav) or profile/settings icon
    menu_clicked = False
    for selector in [
        (By.XPATH, "//span[text()='More']/ancestor::a"),
        (By.XPATH, "//*[contains(text(), 'More')]/ancestor::a"),
        (By.XPATH, "//a[contains(@href, '/accounts/')]"),
        (By.CSS_SELECTOR, "svg[aria-label='Profile']"),
        (By.XPATH, "//span[text()='Profile']/ancestor::a"),
        (By.XPATH, "//a[contains(@href, '/" + target_username + "/')]"),
    ]:
        try:
            el = wait.until(EC.element_to_be_clickable((selector[0], selector[1])))
            el.click()
            menu_clicked = True
            print("Opened menu (More/Profile).")
            break
        except Exception:
            continue
    if not menu_clicked:
        print("Could not open More/Profile menu. Try --no-headless to see the page.")
        return False
    time.sleep(2)

    # 2) Click "Switch accounts" or "Switch account"
    switch_clicked = False
    for selector in [
        (By.LINK_TEXT, "Switch accounts"),
        (By.PARTIAL_LINK_TEXT, "Switch accounts"),
        (By.LINK_TEXT, "Switch account"),
        (By.PARTIAL_LINK_TEXT, "Switch account"),
        (By.XPATH, "//*[contains(text(), 'Switch account')]/ancestor::a"),
        (By.XPATH, "//*[contains(text(), 'Switch accounts')]/ancestor::a"),
    ]:
        try:
            el = wait.until(EC.element_to_be_clickable((selector[0], selector[1])))
            el.click()
            switch_clicked = True
            print("Clicked 'Switch accounts'.")
            break
        except Exception:
            continue
    if not switch_clicked:
        print("Could not find 'Switch accounts'. Try --no-headless.")
        return False
    time.sleep(2)

    # 3) Click the account that matches target_username (e.g. xplore.hk)
    account_clicked = False
    for selector in [
        (By.LINK_TEXT, target_username),
        (By.PARTIAL_LINK_TEXT, target_username),
        (By.XPATH, f"//*[contains(text(), '{target_username}')]/ancestor::a"),
        (By.XPATH, f"//a[contains(@href, '/{target_username}/')]"),
        (By.XPATH, f"//span[text()='{target_username}']/ancestor::a"),
    ]:
        try:
            el = wait.until(EC.element_to_be_clickable((selector[0], selector[1])))
            el.click()
            account_clicked = True
            print(f"Switched to '{target_username}'.")
            break
        except Exception:
            continue
    if not account_clicked:
        print(f"Could not click account '{target_username}'. If Instagram asked for a password, run with --no-headless and enter it manually.")
        return False
    time.sleep(3)
    return True


def dismiss_instagram_popups(driver):
    for text in ("Not Now", "Not now", "Later"):
        try:
            for btn in driver.find_elements(
                By.XPATH,
                f"//button[contains(., '{text}')]|//div[@role='button'][contains(., '{text}')]",
            ):
                if btn.is_displayed():
                    btn.click()
                    time.sleep(1)
                    break
        except Exception:
            pass


def go_to_saved_posts_grid(driver, username, url=None):
    """
    Go to saved posts grid: open /saved/all-posts/ (or click 'All posts' on /saved/).
    """
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    if url and "all-posts" in url:
        driver.get(url)
    else:
        # Direct grid URL is more reliable than click-through
        driver.get(f"https://www.instagram.com/{username}/saved/all-posts/")
    time.sleep(3)
    dismiss_instagram_popups(driver)

    if "all-posts" not in (driver.current_url or ""):
        driver.get(f"https://www.instagram.com/{username}/saved/")
        time.sleep(3)
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


def main():
    parser = argparse.ArgumentParser(
        description="Crawl Instagram saved posts and print post links (requires cookies.pkl)."
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
        help="If you log in as a different account (e.g. primary), switch to this user first: More -> Switch accounts -> USERNAME. Use with --username (e.g. --switch-account xplore.hk --username xplore.hk).",
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Go directly to this URL instead of clicking Saved -> All posts.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only crawl and print links; do not extract event info or write to Google Sheet.",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_PROCESSED_TRACKER,
        metavar="PATH",
        help=(
            "File of processed post URLs (.txt one per line, or .csv with header 'url'). "
            "During extraction, skip URLs listed here; append after each post is processed. "
            f"Default: {DEFAULT_PROCESSED_TRACKER}"
        ),
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="Process every crawled URL even if it was handled on a previous run.",
    )
    args = parser.parse_args()

    headless = not args.no_headless and not args.save_cookies
    driver = create_driver(headless=headless)

    try:
        if args.save_cookies:
            driver.get(INSTAGRAM_HOME)
            print(
                "In the browser: finish login (password, 2FA, 'Save login info?' if shown).\n"
                "If Instagram shows multiple accounts (e.g. david__dlau vs xplore.hk), click the one\n"
                "you use for Saved posts (e.g. xplore.hk) and wait until the home feed loads for that account.\n"
                "Then return here and press Enter to save cookies.pkl for this session."
            )
            input()
            save_cookies(driver)
            return

        if not load_cookies(driver):
            print(
                f"No {COOKIES_FILE} found. Run with --save-cookies first to log in and create it."
            )
            return

        # If you're logged in as primary but want saved posts for another account (e.g. xplore.hk)
        if args.switch_account:
            if not switch_to_account(driver, args.switch_account):
                print("Account switch failed. If Instagram asked for a password, run with --no-headless and enter it, then re-run without --no-headless once both accounts are in the session.")
                return
            time.sleep(2)

        # Go to saved posts grid: /saved/ then click "All posts" (more reliable than profile tab, works headless)
        go_to_saved_posts_grid(driver, args.username, url=args.url)

        # Check if we're on a login wall (e.g. "Save Your Login Info?")
        current = driver.current_url
        if "accounts/login" in current or "challenge" in current:
            print(
                "Looks like you are not logged in or Instagram is asking for verification. "
                "Try running with --no-headless and --save-cookies again after logging in."
            )
            return

        print("Scrolling to load saved posts...")
        print(f"  Page: {driver.current_url}")
        links = extract_post_links(driver)
        keep_driver = not args.no_extract and bool(links)
        if not keep_driver:
            driver.quit()
            driver = None

        tracker_path = None
        processed_urls = set()
        if not args.no_extract and not args.no_tracker:
            tracker_path = os.path.abspath(args.tracker)
            processed_urls = load_processed_links(tracker_path)

        new_links = [
            u for u in links if normalize_post_url(u) not in processed_urls
        ]
        if processed_urls:
            print(
                f"\nFound {len(links)} saved post(s) "
                f"({len(new_links)} not yet in tracker: {tracker_path})."
            )
        else:
            print(f"\nFound {len(links)} saved post(s).")
        if tracker_path and not os.path.isfile(tracker_path):
            print(f"  Tracker file will be created on first processed URL: {tracker_path}")
        print()
        for u in links:
            tag = ""
            if processed_urls and normalize_post_url(u) in processed_urls:
                tag = " [already processed]"
            print(f"{u}{tag}")

        if not args.no_extract and links:
            print("\nExtracting event info and writing to Google Sheet...")
            print(
                f"  [LLM] primary: {', '.join(_primary_models_from_env())} "
                f"-> fallback: {_fallback_model_from_env()}"
            )
            if tracker_path:
                print(f"  Tracker: {len(processed_urls)} URL(s) in {tracker_path}")
            sheet, current_row = init_google_sheets()
            if sheet is None:
                print("[WARN] Google Sheet not available; extraction runs but rows are not saved.")
            written_total = 0
            skipped_tracker = 0
            to_process = new_links if tracker_path else links
            for i, url in enumerate(to_process):
                norm = normalize_post_url(url)
                time.sleep(DELAY_BETWEEN_POSTS)
                soup = get_content_sync(url, driver=driver)
                if not soup:
                    continue
                username, details, post_date = extract_username_and_details(soup)
                details_str = (details or "").strip()
                if not details_str:
                    print(f"  [skip] no caption: {url[:70]}")
                    continue
                try:
                    response, tags, category = extract_info_with_model_fallback(
                        details_str, post_date=post_date
                    )
                except Exception as e:
                    print(f"  [WARN] extract_info (all models) failed for {url}: {e}")
                    continue
                photo_url = extract_photo(url)
                event_list = build_event_info(
                    username, response, tags, category, url, photo_url=photo_url
                )
                if sheet is not None:
                    for event_dict in event_list:
                        key_fields = ["Event name", "Date", "Time", "Location"]
                        if not any(
                            event_dict.get(f) and str(event_dict.get(f)).strip()
                            for f in key_fields
                        ):
                            continue
                        try:
                            sheet, current_row = write_event_to_sheet(
                                sheet, current_row, event_dict
                            )
                            written_total += 1
                        except Exception as e:
                            print(f"  [WARN] write failed: {e}")
                if tracker_path:
                    append_processed_link(tracker_path, norm, processed_urls)
                print(f"  Processed {i + 1}/{len(to_process)}: {url[:60]}...")
            skipped_tracker = len(links) - len(to_process)
            if sheet is not None:
                print(f"\nWrote {written_total} event(s) to Google Sheet (input).")
            if skipped_tracker:
                print(f"Skipped {skipped_tracker} link(s) already in tracker.")

    finally:
        try:
            if driver is not None:
                driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
