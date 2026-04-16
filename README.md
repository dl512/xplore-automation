# xplore-automation

Instagram **saved posts** crawler with optional event extraction to Google Sheets and photo upload to GCS (`ig-photo`).

## What’s in this repo

| File | Role |
|------|------|
| `crawl_ig_saved_posts.py` | Selenium crawl, `extract_info_with_model_fallback`, sheet write |
| `extraction_details.py` | Event JSON extraction, tags, category |
| `llm_env.py` | `AI_GATEWAY_API_KEY` (Vercel) or `OPENAI_API_KEY` + `BASE_URL` |

**Not committed (stay on the VM only):** `.env`, `cookies.pkl`, Google service account JSON, Instaloader session files.

## VM setup (DigitalOcean / Linux)

1. **Chrome / Chromium** for Selenium (install distro package or Google Chrome).

2. **Python 3.10+**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -U pip
   pip install -r requirements.txt
   ```

3. **Clone**
   ```bash
   git clone https://github.com/dl512/xplore-automation.git
   cd xplore-automation
   ```

4. **Secrets on the VM**
   - Copy `.env.example` → `.env` and fill in keys.
   - Upload your Google service account JSON (e.g. `~/secrets/xplore-sa.json`) and set:
     `GOOGLE_SERVICE_ACCOUNT_JSON=/home/you/secrets/xplore-sa.json`
   - Run once with display (or X11) to create cookies:
     ```bash
     python crawl_ig_saved_posts.py --save-cookies --no-headless
     ```
     That writes `cookies.pkl` (gitignored).

5. **Run**
   ```bash
   python crawl_ig_saved_posts.py
   ```
   Links-only (no LLM / sheet):
   ```bash
   python crawl_ig_saved_posts.py --no-extract
   ```

## LLM routing

- **Vercel AI Gateway:** set `AI_GATEWAY_API_KEY` (see [Vercel AI Gateway OpenAI-compatible API](https://vercel.com/docs/ai-gateway/sdks-and-apis/openai-chat-completions)).
- **Otherwise:** `OPENAI_API_KEY` + `BASE_URL` (e.g. OpenRouter).

Model IDs use `provider/model` form (e.g. `google/gemma-2-9b-it`). Override order with `LLM_PRIMARY_MODELS` and `LLM_FALLBACK_MODEL` if needed.

## Optional: monorepo with `xplore_automation.py`

If you place this next to a module named `xplore_automation` that defines `GOOGLE_CLOUD_CREDENTIALS` / `GOOGLE_SHEET_ID`, those are used when present; otherwise credentials load from `GOOGLE_SERVICE_ACCOUNT_JSON`.
