#!/usr/bin/env python3
"""
Interactive extraction-prompt tuner (one saved post at a time).

Does NOT write to Google Sheets. For each post:
  1. Show INPUT (caption / post details + post date)
  2. Run current extraction prompt (gpt-5-mini cascade)
  3. Show OUTPUT (events, tags, category)
  4. You type a comment
  5. Pause so you can paste the comment in Cursor chat for a prompt revision
  6. Optionally re-run the same post after the prompt is updated, then next

Usage:
  # Crawl IG saved posts, then review interactively
  python tune_extraction_prompt.py

  # Reuse last crawled URL list (still opens logged-in Chrome to fetch captions)
  python tune_extraction_prompt.py --from-cache --start 0

  # Resume after quitting mid-session
  python tune_extraction_prompt.py --from-cache --resume

  # Only first N posts
  python tune_extraction_prompt.py --from-cache --start 0 --limit 5

Captions require a logged-in Selenium session (cookies.pkl). Requests-only
fetch almost always returns empty — the browser is kept open by default.

During review:
  After comment → Enter = next post | r = reload prompt + re-extract | q = quit
  Multiline comment: type lines, then a single '.' on its own line (or empty line).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from datetime import datetime, timezone

from dotenv import load_dotenv

load_dotenv(override=True)

import crawl_ig_saved_posts as crawl
import extraction_details as ed

CACHE_URLS_FILE = "extraction_prompt_tuning_urls.txt"
CHECKPOINT_FILE = "extraction_prompt_tuning_checkpoint.json"
LOG_FILE = "extraction_prompt_tuning_log.jsonl"


def _banner(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n  {title}\n{line}")


def _section(title: str) -> None:
    print(f"\n--- {title} ---")


def load_urls_file(path: str) -> list[str]:
    urls = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            u = line.strip()
            if u and not u.startswith("#"):
                urls.append(u)
    return urls


def save_urls_cache(urls: list[str], path: str = CACHE_URLS_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for u in urls:
            f.write(u + "\n")
    print(f"[INFO] Cached {len(urls)} URL(s) → {path}")


def load_checkpoint(path: str = CHECKPOINT_FILE) -> int:
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return int(data.get("next_index", 0))
    except Exception:
        return 0


def save_checkpoint(index: int, path: str = CHECKPOINT_FILE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            {"next_index": index, "updated_at": datetime.now(timezone.utc).isoformat()},
            f,
            indent=2,
        )


def append_log(entry: dict, path: str = LOG_FILE) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_multiline_comment() -> str:
    print(
        "\nYour comment on this extraction (what is wrong / what you want):\n"
        "  • Type one or more lines\n"
        "  • End with a single '.' on its own line, or an empty line\n"
    )
    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "." or line == "":
            if lines or line.strip() == ".":
                break
            # First empty line with no content yet — treat as empty comment
            break
        lines.append(line)
    return "\n".join(lines).strip()


def open_logged_in_driver(args):
    """Create Chrome with cookies.pkl. Returns driver or exits on failure."""
    headless = not args.no_headless
    driver = crawl.create_driver(headless=headless)
    if not crawl.load_cookies(driver):
        try:
            driver.quit()
        except Exception:
            pass
        print(
            f"No {crawl.COOKIES_FILE} found. "
            "Run: python crawl_ig_saved_posts.py --save-cookies"
        )
        sys.exit(1)
    # Land on IG so session cookies apply
    crawl.safe_driver_get(driver, crawl.INSTAGRAM_HOME, wait_after=1.5)
    if args.switch_account:
        if not crawl.switch_to_account(driver, args.switch_account):
            print("Account switch failed.")
            sys.exit(1)
        time.sleep(2)
    return driver


def crawl_saved_post_urls(driver, args) -> list[str]:
    crawl.go_to_saved_posts_grid(driver, args.username, url=args.url)
    current = driver.current_url
    if "accounts/login" in current or "challenge" in current:
        print("Not logged in / challenge page. Re-run cookie save.")
        sys.exit(1)
    print("Scrolling to load saved posts...")
    links = crawl.extract_post_links(driver)
    print(f"Found {len(links)} saved post(s).")
    return links


def fetch_post(url: str, driver=None):
    """
    Returns (username, details, post_date, err).
    err is a short reason when caption is missing.
    """
    soup = crawl.get_content_sync(url, driver=driver)
    if not soup:
        how = "selenium" if driver is not None else "requests"
        return "", "", None, f"fetch failed ({how})"
    username, details, post_date = crawl.extract_username_and_details(soup)
    if not (details or "").strip():
        # Help diagnose login wall / changed meta
        meta = soup.find("meta", attrs={"name": "description"})
        title = soup.title.string if soup.title else None
        if meta and meta.get("content"):
            preview = meta["content"][:120].replace("\n", " ")
            return username, details, post_date, f"meta present but no caption parse: {preview!r}"
        return username, details, post_date, f"no description meta (title={title!r})"
    return username, details, post_date, None


def format_events(response) -> str:
    if not response:
        return "(no events)"
    parts = []
    for i, ev in enumerate(response, 1):
        if not isinstance(ev, dict):
            parts.append(f"  [{i}] {ev!r}")
            continue
        parts.append(
            f"  [{i}] event_name: {ev.get('event_name', '')}\n"
            f"      date:       {ev.get('date', '')}\n"
            f"      time:       {ev.get('time', '')}\n"
            f"      venue:      {ev.get('venue', '')}\n"
            f"      cost:       {ev.get('cost', '')}"
        )
    return "\n".join(parts)


def run_extraction(details: str, post_date):
    """Reload extraction_details so prompt edits apply without restarting."""
    importlib.reload(ed)
    return ed.extract_info_with_model_fallback(details, post_date=post_date)


def review_one(url: str, index: int, total: int, driver=None) -> str:
    """
    Review one post. Returns action: 'next' | 'quit'.
    """
    _banner(f"Post {index + 1} / {total}")
    print(f"URL: {url}")

    if driver is None:
        print("[WARN] No browser session — Instagram usually hides captions from requests.")
    username, details, post_date, err = fetch_post(url, driver=driver)
    details_str = (details or "").strip()
    if not details_str:
        print(f"[skip] no caption: {err or 'unknown'}")
        return "next"

    _section("INPUT (post details fed to LLM)")
    print(f"Organizer: @{username or '?'}")
    print(f"Post date: {post_date or '(unknown)'}")
    print()
    print(details_str)

    while True:
        _section("OUTPUT (current extraction)")
        print(
            f"[LLM] primaries={', '.join(ed._primary_models_from_env())} "
            f"-> fallback={ed._fallback_model_from_env()}"
        )
        try:
            response, tags, category = run_extraction(details_str, post_date)
        except Exception as e:
            print(f"[ERROR] extraction failed: {e}")
            response, tags, category = [], "", ""

        print()
        print("Events:")
        print(format_events(response))
        print(f"\nTags:     {tags}")
        print(f"Category: {category}")

        comment = read_multiline_comment()

        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "index": index,
            "url": url,
            "username": username,
            "post_date": post_date,
            "details": details_str,
            "response": response,
            "tags": tags,
            "category": category,
            "comment": comment,
            "primary_models": ed._primary_models_from_env(),
            "fallback_model": ed._fallback_model_from_env(),
        }
        append_log(entry)
        print(f"[INFO] Logged → {LOG_FILE}")

        if comment:
            print(
                "\n>>> Paste your comment (and this post's input/output if useful) "
                "in Cursor chat so we can revise the extraction prompt.\n"
                ">>> After the prompt is updated, come back here."
            )
        else:
            print("\n[INFO] Empty comment — no revision request logged.")

        print(
            "\nNext action:\n"
            "  [Enter]  next post\n"
            "  r        reload extraction_details.py + re-run THIS post\n"
            "  q        quit (checkpoint saved)\n"
        )
        try:
            action = input("> ").strip().lower()
        except EOFError:
            action = "q"

        if action in ("", "n", "next"):
            return "next"
        if action == "q":
            return "quit"
        if action == "r":
            print("\n[INFO] Reloading prompt module and re-extracting…")
            continue
        print("Unknown command; use Enter, r, or q.")


def main():
    parser = argparse.ArgumentParser(
        description="Tune extraction prompt interactively on saved IG posts (no sheet writes)."
    )
    parser.add_argument(
        "--from-cache",
        action="store_true",
        help=f"Use URL list from {CACHE_URLS_FILE} instead of crawling.",
    )
    parser.add_argument(
        "--urls",
        metavar="FILE",
        help="Text file with one Instagram post URL per line.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=f"Continue from index in {CHECKPOINT_FILE}.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max number of posts to review in this run.",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=None,
        help="0-based start index (overrides --resume if set).",
    )
    parser.add_argument("--username", default=crawl.DEFAULT_USERNAME)
    parser.add_argument("--switch-account", default=None, metavar="USERNAME")
    parser.add_argument("--url", default=None, help="Saved-posts grid URL override.")
    parser.add_argument(
        "--no-headless",
        action="store_true",
        help="Show browser window (default is headless).",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open Selenium (requests only — usually fails to get captions).",
    )
    args = parser.parse_args()

    driver = None
    try:
        if not args.no_browser:
            print("[INFO] Opening logged-in Chrome to fetch post captions…")
            driver = open_logged_in_driver(args)

        if args.urls:
            urls = load_urls_file(args.urls)
            save_urls_cache(urls)
        elif args.from_cache:
            if not os.path.isfile(CACHE_URLS_FILE):
                print(f"No cache at {CACHE_URLS_FILE}. Run without --from-cache first.")
                sys.exit(1)
            urls = load_urls_file(CACHE_URLS_FILE)
            print(f"[INFO] Loaded {len(urls)} URL(s) from {CACHE_URLS_FILE}")
        else:
            if driver is None:
                print("Need a browser to crawl saved posts (omit --no-browser).")
                sys.exit(1)
            urls = crawl_saved_post_urls(driver, args)
            if not urls:
                print("No saved posts found.")
                sys.exit(1)
            save_urls_cache(urls)

        if args.start is not None:
            start = max(0, args.start)
        elif args.resume:
            start = load_checkpoint()
            print(f"[INFO] Resuming at index {start}")
        else:
            start = 0

        end = len(urls)
        if args.limit is not None:
            end = min(end, start + max(0, args.limit))

        if start >= len(urls):
            print(f"Start index {start} is past end ({len(urls)} URLs). Done.")
            return

        print(
            f"\nReviewing posts [{start} .. {end - 1}] of {len(urls)} total.\n"
            f"Log: {LOG_FILE}\n"
            f"Fetch: {'Selenium (cookies)' if driver else 'requests only'}\n"
            f"Models: {', '.join(ed._primary_models_from_env())} "
            f"-> {ed._fallback_model_from_env()}\n"
        )

        for i in range(start, end):
            time.sleep(crawl.DELAY_BETWEEN_POSTS)
            action = review_one(urls[i], i, len(urls), driver=driver)
            next_index = i + 1
            save_checkpoint(next_index)
            if action == "quit":
                print(f"\nStopped. Checkpoint → next index {next_index}.")
                print(
                    "Resume with: python tune_extraction_prompt.py --from-cache --resume"
                )
                break
        else:
            print(f"\nFinished reviewing posts up to index {end - 1}.")
            save_checkpoint(end)
    finally:
        if driver is not None:
            try:
                driver.quit()
            except Exception:
                pass


if __name__ == "__main__":
    main()
