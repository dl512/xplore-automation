#!/usr/bin/env python3
"""
Crawl Instagram home feed and optionally detect events, extract details, save to Google Sheet.

Uses cookies.pkl and Selenium (same as crawl_ig_saved_posts.py).
Event detection uses is_event_post() in extraction_details.py (same rules as xplore_automation).

Usage:
  python crawl_ig_home_feed.py

  # Crawl only, print links (no LLM / sheet)
  python crawl_ig_home_feed.py --no-extract

  # Process URLs from a file (e.g. saved terminal output) — skips browser crawl
  python crawl_ig_home_feed.py --links-file home_feed_links.txt

  python crawl_ig_home_feed.py --no-headless --max-scrolls 30
  python crawl_ig_home_feed.py --switch-account xplore.hk
"""

import argparse
import os
import re
import time

from dotenv import load_dotenv
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

load_dotenv(override=True)

from crawl_ig_saved_posts import (
    COOKIES_FILE,
    DELAY_BETWEEN_POSTS,
    INSTAGRAM_HOME,
    WAIT_TIMEOUT,
    append_processed_link,
    build_event_info,
    create_driver,
    extract_photo,
    extract_username_and_details,
    get_content_sync,
    init_google_sheets,
    load_cookies,
    load_processed_links,
    normalize_post_url,
    save_cookies,
    switch_to_account,
    write_event_to_sheet,
)
from extraction_details import (
    _fallback_model_from_env,
    _primary_models_from_env,
    extract_info_with_model_fallback,
    is_event_post,
)

DEFAULT_HOME_TRACKER = "processed_home_feed_links.txt"
_POST_PATH_RE = re.compile(r"/(p|reel)/([a-zA-Z0-9_-]+)/?")


def filter_post_urls(urls):
    """Keep canonical /p/ and /reel/ URLs; drop liked_by, comments, duplicates."""
    seen_keys = set()
    out = []
    for raw in urls:
        norm = normalize_post_url(raw)
        if not norm:
            continue
        lower = norm.lower()
        if any(x in lower for x in ("/liked_by", "/comments", "/embed")):
            continue
        m = _POST_PATH_RE.search(norm)
        if not m:
            continue
        key = m.group(2)
        if key in seen_keys:
            continue
        seen_keys.add(key)
        out.append(norm)
    return out


def read_links_file(path):
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                urls.append(line)
    return urls


def dismiss_instagram_popups(driver):
    for text in ("Not Now", "Not now", "Later"):
        try:
            buttons = driver.find_elements(
                By.XPATH,
                f"//button[contains(., '{text}')]|//div[@role='button'][contains(., '{text}')]",
            )
            for btn in buttons:
                if btn.is_displayed():
                    btn.click()
                    time.sleep(1)
                    break
        except Exception:
            pass


def go_to_home_feed(driver):
    driver.get(INSTAGRAM_HOME)
    time.sleep(4)
    dismiss_instagram_popups(driver)
    wait = WebDriverWait(driver, WAIT_TIMEOUT)
    for selector in [
        (By.TAG_NAME, "article"),
        (By.XPATH, "//main"),
        (By.XPATH, "//a[contains(@href, '/p/')]"),
    ]:
        try:
            wait.until(EC.presence_of_element_located((selector[0], selector[1])))
            break
        except Exception:
            continue
    time.sleep(2)
    return True


def extract_feed_post_links(driver, scroll_pause=3, max_scrolls=20):
    seen = set()
    links = []
    last_count = 0
    no_new_count = 0

    def collect_from_page():
        scopes = [
            "//article//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
            "//main//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
            "//a[contains(@href, '/p/') or contains(@href, '/reel/')]",
        ]
        for xpath in scopes:
            elements = driver.find_elements(By.XPATH, xpath)
            if not elements:
                continue
            for el in elements:
                try:
                    href = el.get_attribute("href")
                    if not href:
                        continue
                    base = normalize_post_url(href)
                    if not base or base in seen:
                        continue
                    if "/p/" not in base and "/reel/" not in base:
                        continue
                    if any(x in base.lower() for x in ("/liked_by", "/comments")):
                        continue
                    seen.add(base)
                    links.append(base)
                except Exception:
                    pass
            if links:
                break

    for _ in range(max_scrolls):
        collect_from_page()
        if len(links) == last_count:
            no_new_count += 1
            if no_new_count >= 3:
                break
        else:
            no_new_count = 0
        last_count = len(links)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(scroll_pause)

    return filter_post_urls(links)


def is_login_wall(driver):
    url = driver.current_url or ""
    return "accounts/login" in url or "challenge" in url


def _is_event_yes(result):
    if not result:
        return False
    return str(result).strip().upper().startswith("Y")


def process_links_for_events(
    links,
    sheet,
    current_row,
    tracker_path=None,
    processed_urls=None,
):
    """For each URL: tracker skip, is_event check, extract, write to Google Sheet."""
    session_history = []
    written_total = 0
    skipped_not_event = 0
    skipped_tracker = 0

    if processed_urls is None:
        processed_urls = set()

    for i, url in enumerate(links):
        norm = normalize_post_url(url)
        if tracker_path and norm in processed_urls:
            skipped_tracker += 1
            print(f"  [skip tracker] {i + 1}/{len(links)}: {url[:70]}")
            continue

        if i > 0:
            time.sleep(DELAY_BETWEEN_POSTS)

        soup = get_content_sync(url)
        if not soup:
            print(f"  [skip] fetch failed: {url[:70]}")
            continue

        username, details, post_date = extract_username_and_details(soup)
        details_str = str(details).strip() if details else ""
        if not details_str:
            print(f"  [skip] no caption: {url[:70]}")
            continue

        is_event_result = is_event_post(
            details_str, session_history, post_date=post_date
        )
        if not _is_event_yes(is_event_result):
            skipped_not_event += 1
            print(f"  [not event] {i + 1}/{len(links)}: {url[:70]}")
            if tracker_path:
                append_processed_link(tracker_path, norm, processed_urls)
            continue

        print(f"  [event] {i + 1}/{len(links)}: {username} — {url[:70]}")
        try:
            response, tags, category = extract_info_with_model_fallback(
                details_str, post_date=post_date
            )
        except Exception as e:
            print(f"  [WARN] extract failed: {e}")
            continue

        photo_url = extract_photo(url)
        event_list = build_event_info(
            username, response, tags, category, url, photo_url=photo_url
        )

        for event in event_list:
            key_fields = ["Event name", "Date", "Time", "Location"]
            if not any(
                event.get(f) and str(event.get(f)).strip() for f in key_fields
            ):
                continue
            if sheet is not None:
                try:
                    sheet, current_row = write_event_to_sheet(
                        sheet, current_row, event
                    )
                    written_total += 1
                except Exception as e:
                    print(f"  [WARN] write failed: {e}")
                    continue
            session_history.append(
                f"Event name: {event.get('Event name', '')}, "
                f"Date: {event.get('Date', '')}, "
                f"Time: {event.get('Time', '')}, "
                f"Location: {event.get('Location', '')}"
            )

        if tracker_path:
            append_processed_link(tracker_path, norm, processed_urls)

    return {
        "written": written_total,
        "skipped_not_event": skipped_not_event,
        "skipped_tracker": skipped_tracker,
        "sheet": sheet,
        "current_row": current_row,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Crawl Instagram home feed; optionally detect events and write to Google Sheet."
    )
    parser.add_argument("--save-cookies", action="store_true")
    parser.add_argument("--no-headless", action="store_true")
    parser.add_argument("--switch-account", metavar="USERNAME", default=None)
    parser.add_argument("--max-scrolls", type=int, default=20)
    parser.add_argument("--scroll-pause", type=float, default=3)
    parser.add_argument(
        "--output",
        metavar="PATH",
        default=None,
        help="Write crawled URLs (one per line) to this file.",
    )
    parser.add_argument(
        "--links-file",
        metavar="PATH",
        default=None,
        help="Process URLs from file instead of crawling the feed.",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="Only crawl / list links; no is_event check or sheet writes.",
    )
    parser.add_argument(
        "--tracker",
        default=DEFAULT_HOME_TRACKER,
        metavar="PATH",
        help=f"Skip URLs already listed here; append after each post is handled. Default: {DEFAULT_HOME_TRACKER}",
    )
    parser.add_argument(
        "--no-tracker",
        action="store_true",
        help="Do not use the processed-links tracker file.",
    )
    args = parser.parse_args()

    links = []

    if args.links_file:
        if not os.path.isfile(args.links_file):
            print(f"Links file not found: {args.links_file}")
            return
        raw = read_links_file(args.links_file)
        links = filter_post_urls(raw)
        print(f"Loaded {len(raw)} line(s) from {args.links_file} -> {len(links)} unique post URL(s).")
    else:
        headless = not args.no_headless and not args.save_cookies
        driver = create_driver(headless=headless)
        try:
            if args.save_cookies:
                driver.get(INSTAGRAM_HOME)
                print(
                    "Log in in the browser, wait for the home feed, then press Enter to save cookies.pkl."
                )
                input()
                save_cookies(driver)
                return

            if not load_cookies(driver):
                print(f"No {COOKIES_FILE} found. Run with --save-cookies first.")
                return

            driver.get(INSTAGRAM_HOME)
            time.sleep(2)
            driver.refresh()
            time.sleep(2)

            if args.switch_account:
                if not switch_to_account(driver, args.switch_account):
                    print("Account switch failed.")
                    return
                time.sleep(2)

            print("Opening home feed...")
            go_to_home_feed(driver)
            if is_login_wall(driver):
                print("Not logged in. Run with --save-cookies.")
                return

            print(f"Scrolling feed (max {args.max_scrolls} scrolls)...")
            links = extract_feed_post_links(
                driver,
                scroll_pause=args.scroll_pause,
                max_scrolls=args.max_scrolls,
            )
        finally:
            try:
                driver.quit()
            except Exception:
                pass

    print(f"\n{len(links)} post URL(s):\n")
    for u in links:
        print(u)

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            for u in links:
                f.write(u + "\n")
        print(f"\nWrote {len(links)} URL(s) to {out_path}")

    if args.no_extract or not links:
        return

    print("\nChecking events and writing to Google Sheet...")
    print(
        f"  [LLM] primary: {', '.join(_primary_models_from_env())} "
        f"-> fallback: {_fallback_model_from_env()}"
    )

    sheet, current_row = init_google_sheets()
    if sheet is None:
        print("[WARN] Google Sheet not available; extraction runs but rows are not saved.")

    tracker_path = None
    processed_urls = set()
    if not args.no_tracker:
        tracker_path = os.path.abspath(args.tracker)
        processed_urls = load_processed_links(tracker_path)
        if processed_urls:
            print(f"  Tracker: {len(processed_urls)} URL(s) in {tracker_path}")

    stats = process_links_for_events(
        links,
        sheet,
        current_row,
        tracker_path=tracker_path,
        processed_urls=processed_urls,
    )
    print(
        f"\nDone. Wrote {stats['written']} event row(s). "
        f"Skipped: {stats['skipped_tracker']} tracker, "
        f"{stats['skipped_not_event']} not events."
    )


if __name__ == "__main__":
    main()
