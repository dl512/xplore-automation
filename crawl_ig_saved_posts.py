#!/usr/bin/env python3
"""
Crawl Instagram saved posts, output post links, and optionally extract event info to Google Sheet.

Uses Selenium + Chrome (same setup as xplore_automation.py). Requires login via cookies.
Cookies are read from cookies.pkl (create once by running with --save-cookies and logging in).

Usage:
  # First time: create cookies (browser opens; log in to IG, then press Enter)
  python crawl_ig_saved_posts.py --save-cookies

  # Crawl saved posts, extract event info from each post, append to Google Sheet (input tab)
  python crawl_ig_saved_posts.py

  # Only crawl and print links (no extraction / sheet)
  python crawl_ig_saved_posts.py --no-extract

  # If you log in as primary but want saved posts for xplore.hk:
  python crawl_ig_saved_posts.py --switch-account xplore.hk --username xplore.hk

  LLM: tries GPT models (gpt-4.1-nano, gpt-4o-mini, gpt-3.5-turbo) in order; if each
  call errors or returns blank event fields, falls back to google/gemma-2-9b-it.
  Override with LLM_PRIMARY_MODELS=id1,id2 and LLM_FALLBACK_MODEL=id (optional).
"""

import argparse
import json
import os
import pickle
import re
import time
import uuid

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from selenium import webdriver
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
    if not bucket:
        return ""
    try:
        response = requests.get(image_url, timeout=15)
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
    """Get post image via Instaloader, upload to GCS, return public URL. Same as xplore_automation."""
    try:
        import instaloader
        L = instaloader.Instaloader()
        try:
            if os.path.exists("session-instaloader"):
                L.load_session_from_file("session-instaloader")
        except Exception:
            pass
        shortcode = url.split("p/")[1].strip("/ ")
        post = instaloader.Post.from_shortcode(L.context, shortcode)
        return manage_photo(post.url) or ""
    except Exception:
        return ""

# Profile username whose saved posts we want (must be the logged-in user)
DEFAULT_USERNAME = "xplore.hk"
COOKIES_FILE = "cookies.pkl"
INSTAGRAM_HOME = "https://www.instagram.com"
WAIT_TIMEOUT = 15
DELAY_BETWEEN_POSTS = 2  # seconds between fetching posts (rate limit)


def get_content_sync(url):
    """Fetch post HTML with requests (no login; public meta tags)."""
    try:
        r = requests.get(url, timeout=15)
        if r.status_code == 200:
            return BeautifulSoup(r.text, "html.parser")
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


def create_driver(headless=True):
    options = Options()
    if headless:
        options.add_argument("--headless")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )
    return driver


def load_cookies(driver):
    driver.get(INSTAGRAM_HOME)
    time.sleep(2)
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


def go_to_saved_posts_grid(driver, username, url=None):
    """
    Go to saved posts grid: open https://www.instagram.com/{username}/saved/ then click 'All posts'.
    More reliable than profile -> Saved tab (works in headless). If url is given and contains
    'all-posts', just open that URL; otherwise open /saved/ and click 'All posts'.
    """
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    if url and "all-posts" in url:
        driver.get(url)
        time.sleep(4)
        return True
    saved_url = f"https://www.instagram.com/{username}/saved/"
    driver.get(saved_url)
    time.sleep(4)
    # Click "All posts" to open the grid (one click on /saved/ page)
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
    return True


def extract_post_links(driver, scroll_pauses=3, max_scrolls=50):
    """Scroll the saved posts page and collect all /p/ and /reel/ links."""
    seen = set()
    links = []
    last_count = 0
    no_new_count = 0

    for _ in range(max_scrolls):
        # Find all post/reel links on current page
        elements = driver.find_elements(
            By.XPATH,
            "//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
        )
        for el in elements:
            try:
                href = el.get_attribute("href")
                if not href:
                    continue
                # Normalize: strip query string for dedup
                base = href.split("?")[0]
                if base not in seen:
                    seen.add(base)
                    links.append(href)
            except Exception:
                pass

        if len(links) == last_count:
            no_new_count += 1
            if no_new_count >= 3:
                break
        else:
            no_new_count = 0
        last_count = len(links)

        # Scroll down to load more
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pauses)

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
    args = parser.parse_args()

    headless = not args.no_headless and not args.save_cookies
    driver = create_driver(headless=headless)

    try:
        if args.save_cookies:
            driver.get(INSTAGRAM_HOME)
            print("Log in to Instagram in the browser, then press Enter here.")
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
        links = extract_post_links(driver)
        driver.quit()

        print(f"\nFound {len(links)} saved post(s). Links:\n")
        for u in links:
            print(u)

        if not args.no_extract and links:
            print("\nExtracting event info and writing to Google Sheet...")
            print(
                f"  [LLM] primary: {', '.join(_primary_models_from_env())} "
                f"-> fallback: {_fallback_model_from_env()}"
            )
            sheet, current_row = init_google_sheets()
            if sheet is None:
                print("[WARN] Google Sheet not available; skipping extract/save.")
            else:
                written_total = 0
                for i, url in enumerate(links):
                    time.sleep(DELAY_BETWEEN_POSTS)
                    soup = get_content_sync(url)
                    if not soup:
                        continue
                    username, details, post_date = extract_username_and_details(soup)
                    details_str = (details or "").strip()
                    if not details_str:
                        continue
                    try:
                        response, tags, category = extract_info_with_model_fallback(
                            details_str, post_date=post_date
                        )
                    except Exception as e:
                        print(f"  [WARN] extract_info (all models) failed for {url}: {e}")
                        continue
                    photo_url = extract_photo(url)
                    event_list = build_event_info(username, response, tags, category, url, photo_url=photo_url)
                    for event_dict in event_list:
                        key_fields = ["Event name", "Date", "Time", "Location"]
                        if not any(event_dict.get(f) and str(event_dict.get(f)).strip() for f in key_fields):
                            continue
                        try:
                            sheet, current_row = write_event_to_sheet(sheet, current_row, event_dict)
                            written_total += 1
                        except Exception as e:
                            print(f"  [WARN] write failed: {e}")
                    print(f"  Processed {i + 1}/{len(links)}: {url[:60]}...")
                print(f"\nWrote {written_total} event(s) to Google Sheet (input).")

    finally:
        try:
            driver.quit()
        except Exception:
            pass


if __name__ == "__main__":
    main()
