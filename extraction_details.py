#!/usr/bin/env python3
"""
Event Details Extraction Module

This module contains functions for extracting event information from Instagram posts.
Used by both xplore_automation.py and extract_from_ig_txt.py.
LLM routing: set AI_GATEWAY_API_KEY for Vercel AI Gateway, or OPENAI_API_KEY + BASE_URL (see llm_env.py).

Functions:
    - create_chain: Create LangChain for event extraction
    - extract_tags: Extract relevant tags from event details
    - select_category: Select category for event
    - extract_info: Main function to extract event information from details
    - extract_info_with_model_fallback: Try GPT models via OpenRouter, then Gemma
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_core.utils.json import parse_json_markdown
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from llm_env import get_openai_compatible_api_key, get_openai_compatible_base_url
from typing import List
import os
import re
import json

# Tags and categories
TAGS_LIST = [
    "展覽", "讀書會", "文學", "哲學", "導賞團", "歷史", "社區", "文化", "世界",
    "藝術", "攝影", "電影", "劇場", "棟篤笑", "聚會", "大自然", "性別關係",
    "親子", "身心靈", "宗教", "手作", "運動", "咖啡", "酒", "音樂", "市集", "其他"
]

CATEGORY_LIST = [
    "Live Music Show", "舞蹈", "劇與電影", "辯論與搞笑", "Gathering", "旅遊分享",
    "運動", "讀書會", "文學活動", "香港導賞團", "認識香港歷史", "哲學與思考",
    "性別關係", "市集", "環保與大自然", "藝術", "靜心活動", "心靈健康",
    "手作工作坊", "社區活動"
]

EMPTY_EVENT = {
    'event_name': "",
    'date': "",
    'time': "",
    'venue': "",
    'cost': ""
}

# Keys expected in each event dict (for normalizing repaired parse output)
EVENT_KEYS = ("event_name", "date", "time", "venue", "cost")

# OpenRouter: extract_info_with_model_fallback tries these first, then Gemma.
DEFAULT_LLM_PRIMARY_MODELS = [
    "openai/gpt-4.1-nano",
    "openai/gpt-4o-mini",
    "openai/gpt-3.5-turbo",
]
DEFAULT_LLM_FALLBACK_MODEL = "google/gemma-2-9b-it"
_LLM_OPENROUTER_TIMEOUT = 120


def _primary_models_from_env():
    raw = os.getenv("LLM_PRIMARY_MODELS", "").strip()
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return list(DEFAULT_LLM_PRIMARY_MODELS)


def _fallback_model_from_env():
    m = os.getenv("LLM_FALLBACK_MODEL", DEFAULT_LLM_FALLBACK_MODEL).strip()
    return m or DEFAULT_LLM_FALLBACK_MODEL


def create_openrouter_chat_llm(model: str, timeout: int | None = None) -> ChatOpenAI:
    """ChatOpenAI pointed at Vercel AI Gateway, OpenRouter, or OpenAI (see llm_env)."""
    api_key = get_openai_compatible_api_key()
    base_url = get_openai_compatible_base_url()
    t = timeout if timeout is not None else _LLM_OPENROUTER_TIMEOUT
    return ChatOpenAI(
        model=model,
        openai_api_key=api_key,
        openai_api_base=base_url,
        timeout=t,
        max_retries=1,
    )


def _response_has_usable_events(response) -> bool:
    """True if at least one event has a non-empty name, date, time, or venue."""
    if not response:
        return False
    key_fields = ("event_name", "date", "time", "venue")
    for ev in response:
        if not isinstance(ev, dict):
            continue
        if any(ev.get(f) and str(ev.get(f)).strip() for f in key_fields):
            return True
    return False


_RAW_PREVIEW_LIMIT = 2500


def _extraction_debug_verbose() -> bool:
    return os.getenv("EXTRACTION_DEBUG", "").strip().lower() in ("1", "true", "yes", "on")


def _preview_text(text: str, max_len: int = _RAW_PREVIEW_LIMIT) -> str:
    if not text:
        return "(empty)"
    s = text.strip()
    if len(s) <= max_len:
        return s
    return s[:max_len] + f"\n… [{len(s)} chars total]"


def _raw_from_ai_message(msg) -> str:
    raw = getattr(msg, "content", None)
    if raw is None:
        return ""
    if isinstance(raw, list):
        parts = []
        for block in raw:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        return "".join(parts)
    return str(raw)


def repair_llm_json(text):
    """Fix common LLM JSON mistakes so the event array parses (e.g. '}. ' instead of '}, ')."""
    if not text or not isinstance(text, str):
        return text
    # Fix "}\s*.\s*,\s*" -> "}, " (period used instead of comma between array elements)
    text = re.sub(r"\}\s*\.\s*\n\s*,", "},\n", text)
    text = re.sub(r"\}\s*\.\s*\n\s*\]", "}\n]", text)
    text = re.sub(r"\}\s*\.\s*,", "},", text)
    text = re.sub(r"\}\s*\.\s*\]", "}]", text)
    # Trailing commas before ] or } (invalid JSON; common in LLM output)
    prev = None
    while prev != text:
        prev = text
        text = re.sub(r",(\s*\])", r"\1", text)
        text = re.sub(r",(\s*\})", r"\1", text)
    return text


def _event_dicts_from_parsed(obj):
    """Turn parsed JSON (various shapes) into a list of event dicts or None."""
    events = None
    if isinstance(obj, list):
        if obj and all(isinstance(x, dict) for x in obj):
            events = obj
    elif isinstance(obj, dict):
        events = obj.get("event")
        if isinstance(events, dict):
            events = [events]
        elif not isinstance(events, list):
            # Single event as flat object (no "event" wrapper)
            if any(obj.get(k) is not None for k in EVENT_KEYS):
                events = [obj]
            else:
                events = None
    if not isinstance(events, list):
        return None
    out = []
    for item in events:
        if not isinstance(item, dict):
            continue
        out.append({k: (item.get(k) or "") for k in EVENT_KEYS})
    return out if out else None


def _parse_events_from_raw(raw_text):
    """Parse event list from raw LLM string (repaired). Returns list of event dicts or None."""
    repaired = repair_llm_json(raw_text)
    try:
        obj = parse_json_markdown(repaired)
    except Exception:
        try:
            obj = json.loads(repaired)
        except Exception:
            return None
    return _event_dicts_from_parsed(obj)


def create_chain(model, post_date=None):
    """Create LangChain for event extraction.

    Returns (chain, prompt, parser): ``chain`` is prompt | model | parser; callers can run
    ``(prompt | model).invoke`` to log raw text before ``parser.invoke``.
    """
    class EventDetails(BaseModel):
        event_name: str = Field(description="Name of the event")
        date: str = Field(description="Date of the event")
        time: str = Field(description="Time of the event")
        venue: str = Field(description="Venue of the event")
        cost: str = Field(description="Cost of the event")
    
    class Event(BaseModel):
        event: List[EventDetails] = Field(description="Details of each event")

    parser = JsonOutputParser(pydantic_object=Event)

    post_date_context = ""
    if post_date:
        # Parse post_date to extract day and month for use in date ranges
        try:
            from datetime import datetime
            post_dt = datetime.strptime(post_date, "%B %d, %Y")
            post_day = post_dt.day
            post_month = post_dt.month
            post_date_dm = f"{post_day}/{post_month}"
        except:
            post_date_dm = None
        
        post_date_context = f"\n\nIMPORTANT CONTEXT: This Instagram post was posted on {post_date}."
        if post_date_dm:
            post_date_context += f" The post date in D/M format is {post_date_dm}."
        post_date_context += f"\n\nCRITICAL: The POST DATE ({post_date_dm if post_date_dm else post_date}) and the EVENT DATE are COMPLETELY DIFFERENT. "
        post_date_context += f"The post date is when the post was published, NOT when the event happens. "
        post_date_context += f"NEVER use the post date as the event date unless the post explicitly says '即日起' (from now/starting today) or similar phrases. "
        post_date_context += f"For example, if the post date is {post_date_dm if post_date_dm else 'February 11'} but the post says '日期：2月14日', extract 14/2 (the event date), NOT {post_date_dm if post_date_dm else '11/2'} (the post date). "
        post_date_context += f"\nCRITICAL EXAMPLE: If the post says '一連三日' (three consecutive days) but does NOT specify which dates, you MUST output 'N/A'. Do NOT invent dates like '{post_date_dm if post_date_dm else '11/2'}-13/2' based on the post date. The post date is NOT the event date. "
        post_date_context += f"\nThe post date is ONLY used for: (1) resolving ambiguous dates like '6/2' (could be June 2 or February 6), and (2) understanding '即日起' to mean starting from the post date. "
        if post_date_dm:
            post_date_context += f"For example, if the post says '即日起至 2026.06.07', extract the date range as {post_date_dm}-7/6 (using post date as start). "
        post_date_context += f"In all other cases, extract the event dates as stated in the post, completely ignoring the post date. If no specific dates are mentioned, output 'N/A' - NEVER invent dates based on the post date."

    prompt = ChatPromptTemplate.from_messages([
        ("system", f'''
Given the following event details:
{{details}}{post_date_context}

CRITICAL: Before extracting, carefully scan the ENTIRE post to identify if there are MULTIPLE SEPARATE EVENTS. Look for:
- Different location markers (e.g., @維園 vs @啟德, different venue names)
- Clearly separated sections with different dates, times, or venues
- Multiple event titles or headers (e.g., "【一早共修@維園】" and "【一早共修@啟德】")
If you find multiple events with different locations, dates, or times, you MUST create SEPARATE event entries for EACH one.

Please extract the information as follows:
- **event_name**: 
  1. Language preference:
     - If both Chinese and English titles are available, use the Chinese title only.
     - If only one language is available, use that language (do not translate).
     - When both CN and EN titles are present in the text, they represent the same event - extract only one title, not two separate events.
  
  2. CRITICAL - Extract the FULL title:
     - Extract the COMPLETE title as it appears in the post, including all important parts.
     - If the title includes brackets like 【】with important information (e.g., "【起勢搖滾京港青年交流團】2月15日啟動禮"), extract the FULL title including the brackets and date information.
     - Do NOT shorten titles unnecessarily. For example, "【起勢搖滾京港青年交流團】2月15日啟動禮" should be extracted as-is, NOT shortened to just "起勢搖滾京港青年交流團".
     - Only remove decorative elements that are clearly not part of the title (e.g., standalone emojis, promotional phrases like "2026年強勢登場").
     - When in doubt, include more information rather than less to ensure the title is complete and descriptive.
  
  3. Cleaning the title (only when necessary):
     - Remove decorative brackets like 【】ONLY if they are clearly decorative and not part of the actual title structure.
     - Remove promotional text, years (e.g., "2026年強勢登場"), emojis used for emphasis, or irrelevant marketing phrases.
     - Do not include hashtags, social media handles (@username), or promotional language in the title.
     - Keep the core event title that accurately describes the event, but preserve important structural elements like 【】if they are part of the title.
  
  4. Understanding context - CRITICAL: Read the ENTIRE post carefully to find the ACTUAL event name:
     - Read through the ENTIRE text from beginning to end to understand what the actual event is, not just the first sentence or headline.
     - The opening headline is often a promotional hook / call-to-action / marketing tagline and frequently NOT the actual event name.
     - The actual event name is usually stated more formally later in the post, often with quotation marks (「」) or brackets (【】), and is typically followed by words like "展覽", "活動", "工作坊", "音樂會", etc.
    - If the best-looking title candidate contains marketing/CTA wording (e.g., "最後召集", "不容錯過", "把握機會", "倒數", "即將", "即刻報名"), ignore it as the event_name when a separate formal event title exists elsewhere in the post (typically near date/time/venue).
     - Look for the formal event title that appears in the detailed event information section (usually near dates, times, and venue information). This is the actual event name.
    - Promotional/CTA headlines should be treated as marketing, not the event identity.
     - The event might be a course introduction meeting, workshop session, or specific gathering mentioned later in the text.
     - Extract the title of the actual event being promoted, which may differ from the opening hook or promotional text. Always prioritize the formal event name stated in the event details section.
  
  5. Including important context:
     - Do not omit important information that describes the nature of the event (e.g., "火苗讀書會130" should not be shortened to just the book title).
     - Include event type/format when it's part of the event identity (e.g., "讀書會", "工作坊", "課程簡介會", "啟動禮").
     - If the title includes date information (e.g., "2月15日啟動禮"), preserve it as it's part of the event name.
  
  6. CRITICAL - Multiple events in a single post:
     - When a post contains multiple related events (e.g., a main exhibition and an extension activity like an artist sharing session), the later event(s) should include context from the main/overall event in their title.
     - For example, if the main event is "【動漫「墨」搏】系列第二場展覽 —— 《推演之間》" and there's a later event "延伸活動 —— 藝術家分享會", the second event's title should be "【動漫「墨」搏】系列第二場展覽 —— 《推演之間》延伸活動 —— 藝術家分享會" (including the main event context), NOT just "延伸活動 —— 藝術家分享會".
     - This ensures that each event entry is self-contained and readers understand the full context of what the event is about.
     - Only include the main event context if it's relevant and helps clarify what the later event is about. If events are completely unrelated, do not force this connection.
  
  7. For movies, dramas, theater productions, musicals, or similar performing arts events:
     - Use 《》 (Chinese quotation marks) around the work's name/title.
     - CRITICAL: For movie screenings with special events (e.g., 映後分享, 謝票場, 周末謝票場, 導演分享), you MUST include both the movie name AND the event type. Simply extracting the movie name alone is insufficient.
     - Format for movie screenings with special events: 《電影名》+ 活動類型 (e.g., 《地母》映後分享, 《今天應該很高興》周末謝票場)
     - The event name must clearly indicate what the event is about - a movie name alone does not tell the reader it's a post-screening discussion or special screening event.
     - If available, include the activity type (e.g., 原創粵語音樂劇, 詩歌音樂劇場作品) before the title for theater/drama productions.
     - If the title itself is not self-explanatory (e.g., just 《劇名》), include the nature of the performance (e.g., 音樂劇, 舞台劇) after the title if such information is available in the text.
     - Format examples: 
       * 原創粵語音樂劇《一束光——高錕的記憶》
       * 《異曲同夢》音樂劇 (when 音樂劇 is mentioned in the text)
       * 《地母》映後分享 (movie screening with post-screening discussion)
       * 《今天應該很高興》周末謝票場 (movie screening with special thank-you session)
       * 《劇名》 (when no additional context is available)
     - Do not make up activity types if they're not mentioned in the text.
  
  8. For other events, extract the FULL title that accurately represents the event being promoted, including all important parts.

- **date**: CRITICAL - You MUST output dates ONLY in D/M format (day/month, no leading zeros, no year).
  - CRITICAL: DO NOT INVENT OR GUESS DATES. Only extract dates that are EXPLICITLY mentioned in the post text. If no date is mentioned anywhere in the post, you MUST output "N/A". Never make up dates based on assumptions, context clues, patterns, or the post date. If the post does not contain a specific date, time, or date range, the date field must be "N/A".
  - ABSOLUTELY FORBIDDEN: Do NOT infer dates from vague temporal references. For example:
    * If the post says "一連三日" (three consecutive days) but does NOT specify which dates, output "N/A". Do NOT guess dates like "14-16/2" based on the post date.
    * If the post says "將舉行" (will be held) or "即將舉行" (will be held soon) without specific dates, output "N/A".
    * If the post mentions a duration (e.g., "一連X日", "為期X天") but no start or end dates, output "N/A".
    * Phrases like "mark實免費報名日期" (mark the free registration date) do NOT indicate event dates - they refer to registration dates, not event dates.
  - ONLY extract dates when they are EXPLICITLY stated in formats like: "2月14日", "14/2", "February 14", "日期：14/2", "14-16/2", etc. Vague references without specific dates must result in "N/A".
  - EXTRACT ALL DATES AND EVENTS: 
    * If the post mentions multiple dates (e.g., "6/2 (FRI) ... 7/2 (SAT) ..."), you MUST extract ALL of them.
    * CRITICAL: If the post contains multiple events with different locations, dates, or times (e.g., "【一早共修@維園】" with dates 28/2, 28/3 and "【一早共修@啟德】" with date 14/3), you MUST create SEPARATE event entries for EACH event. Each event should have its own event_name, date, time, and venue.
    * Look for clear separators like different location markers (e.g., @維園 vs @啟德), different venue names, or clearly separated sections in the post.
    * If events share the same details (same name, venue, time), you can list all dates in comma-separated format. But if they have different locations, times, or are clearly different events, create separate entries.
    * Example: If a post has "【一早共修@維園】日期：28/2, 28/3" and "【一早共修@啟德】日期：14/3", create TWO separate events - one for 維園 and one for 啟德.
  - For a single date: D/M format ONLY (e.g., 1/2 for February 1, NOT 01/02; 12/3 for March 12, NOT 12/03)
  - For multiple dates: D/M, D/M, D/M format ONLY (comma-separated, e.g., 6/2, 7/2 for February 6 and February 7)
  - For a date range: D/M-D/M format ONLY (e.g., 1/2-5/2 for February 1 to February 5, or 12/3-15/3 for March 12 to March 15)
  - If NO date is mentioned: Output "N/A" (do NOT guess, estimate, or infer dates from other information)
  - ABSOLUTELY NO leading zeros: Use 1/2 (NOT 01/02), 5/2 (NOT 05/02), 6/2 (NOT 06/02), 9/11 (NOT 09/11)
  - ABSOLUTELY NO year in the output: Extract only day and month, never include the year
  - CRITICAL: The POST DATE and EVENT DATE are COMPLETELY DIFFERENT. The post date is when the Instagram post was published. The event date is when the actual event happens. NEVER confuse them.
  - CRITICAL: NEVER INVENT DATES based on the post date. If a post mentions "一連三日" (three consecutive days) or similar duration phrases WITHOUT specifying actual dates, you MUST output "N/A". Do NOT calculate dates by adding days to the post date.
  - When dates are EXPLICITLY stated in the post (in any format), extract them EXACTLY as stated. Do NOT use the post date as the event date. For example:
    * If the post says "日期：2月14日（六）、2月15日（日）", extract "14/2, 15/2" (the event dates), NOT the post date.
    * If the post says "2月4日" (February 4th in Chinese), extract it as 4/2 (February 4), NOT the post date.
    * If the post says "一連三日" but does NOT specify which dates, output "N/A" - do NOT invent dates like "14-16/2" based on the post date.
    * The post date should be COMPLETELY IGNORED when extracting event dates, unless the post explicitly says "即日起" (from now/starting today).
  - The post date is ONLY used for: (1) resolving ambiguous numeric dates, and (2) understanding "即日起" phrases. In all other cases, IGNORE the post date and extract only the event dates mentioned in the post. If no dates are mentioned, output "N/A" - NEVER invent dates.
  - Chinese date formats: When you see Chinese date formats like "2月4日", "3月15日", etc., extract them correctly: "X月Y日" means "Month X, Day Y". For example, "2月4日" = February 4th = 4/2, "3月15日" = March 15th = 15/3.
  - Using post date context - STRICT RULES:
    * The post date is ONLY used for TWO specific purposes:
      1. Resolving ambiguous numeric dates (like "6/2" which could be June 2 or February 6) - use post date to determine the correct month.
      2. Understanding phrases like "即日起" (from now/starting today), "即日開始" (starting today), or "今日起" (from today) - these mean the event starts from the post date.
    * In ALL other cases, COMPLETELY IGNORE the post date. Do NOT use it as the event date.
    * If event dates are clearly stated in the post (e.g., "日期：2月14日", "2月4日", "February 4", "4/2"), extract those dates EXACTLY as stated. The post date is IRRELEVANT.
    * Example: If post date is 11/2 but the post says "日期：2月14日（六）、2月15日（日）", extract "14/2, 15/2" (the event dates), NOT "11/2" (the post date). The post date should be completely ignored in this case.
  - When interpreting dates in the post: Unless the post explicitly specifies otherwise, assume ALL dates mentioned in the post are in UK date format (day/month/year). For example, if the post shows "12.3.2026" or "12/3/2026", interpret this as March 12, 2026 (UK format: DD.MM.YYYY or DD/MM/YYYY), NOT December 3. Extract only as 12/3 (without the year).
  - REMINDER: Output format is ALWAYS D/M (single digit day/month without leading zeros, no year) for ALL cases: single dates, date ranges, and multiple dates. DO NOT MISS ANY DATES mentioned in the post, but also DO NOT INVENT dates that are not mentioned.
- **time**: Provide the time in the format of 8am-10pm ONLY if explicitly mentioned in the post. If no time is mentioned, output 'N/A'. Do NOT invent or guess times.
- **venue**: State the venue ONLY if explicitly mentioned in the details. If no venue is mentioned, output 'N/A'. Do NOT invent or guess venues.
- **cost**: State the cost ONLY if explicitly mentioned in the details. If free is explicitly stated, output 'Free'. If no cost information is mentioned, output 'N/A'. Do NOT invent or guess costs. If available, use the follow format:
for a single price: HK$x
for multiple prices (e.g., a concert): HK$x | HK$y | HK$z
for multiple prices given certain condition: HK$x (member) | HK$y (non-member)
If any information is missing, indicate it as 'N/A'. DO NOT INVENT any information that is not explicitly stated in the post.
Output valid JSON only: use a comma between event objects in the array, not a period. Each event object must be separated by a comma.
{{format_instructions}}
        ''')
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | model | parser
    # prompt + parser exposed for extract_info debug (raw LLM text before JSON parse).
    return chain, prompt, parser


def extract_tags(details, llm):
    """Extract relevant tags from event details"""
    class Tag(BaseModel):
        tags: List[str] = Field(description="List of selected tags")

    parser = JsonOutputParser(pydantic_object=Tag)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
You are an assistant tasked with reviewing an event description. Given the details of the event: {details}, please analyze the content carefully and select the most relevant tags from the following list: {tags_list}. 
Focus on the key themes, topics, and activities described in the event. Your response should prioritize accuracy and relevance, including only those tags that directly reflect important aspects of the event.
Do not introduce any new tag which is not in tags_list
There is no need to return a fixed number of tags; provide as many relevant tags as you see fit.
But do not overkill. As a rule of thumb, generally 1-3 tags are sufficient

Wrap your final output in the following format:
\n{format_instructions}\n
        """)
    ]).partial(format_instructions=parser.get_format_instructions())
    
    chain = prompt | llm | parser

    for _ in range(10):
        try:
            response = chain.invoke({"tags_list": TAGS_LIST, "details": details})["tags"]
            return ", ".join(response)
        except:
            pass
    return "其他"


def _normalize_llm_category_text(content) -> str:
    """Strip leading/trailing whitespace and newlines; use first non-empty line if model returns extra lines."""
    if content is None:
        return ""
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(str(block["text"]))
            else:
                parts.append(str(block))
        text = "".join(parts)
    else:
        text = str(content)
    text = text.strip()
    if not text:
        return ""
    for line in text.splitlines():
        s = line.strip()
        if s:
            return s
    return ""


def select_category(details, llm):
    """Select category for event"""
    prompt = f"Give this event details: {details}, please select one of the below category.\n"
    prompt += str(CATEGORY_LIST)
    prompt += "\nPlease output only the category name. Make sure there is no quotation mark"
    try:
        response = llm.invoke(prompt)
        return _normalize_llm_category_text(getattr(response, "content", None))
    except:
        return ""


def extract_info(details, model, post_date=None):
    """Extract event information from details"""
    _chain, prompt, parser = create_chain(model, post_date=post_date)
    output = None
    model_label = (
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )
    print(f"[DEBUG] extract_info - model: {model_label}")
    print(f"[DEBUG] extract_info - details length: {len(details) if details else 0} chars; post_date={post_date!r}")

    last_raw_text = ""
    try:
        msg = (prompt | model).invoke({"details": details})
        last_raw_text = _raw_from_ai_message(msg)
        print(f"[DEBUG] extract_info - event step: raw LLM output, {len(last_raw_text)} chars")
        print(f"[DEBUG] extract_info - raw preview:\n{_preview_text(last_raw_text)}")
        if _extraction_debug_verbose():
            print(f"[DEBUG] extract_info - FULL raw (EXTRACTION_DEBUG=1):\n{last_raw_text}")

        response = parser.invoke(msg)
        # Debug: Log the full response structure
        print(f"[DEBUG] extract_info - Parsed JSON type: {type(response)}")
        if isinstance(response, dict):
            print(f"[DEBUG] extract_info - Parsed top-level keys: {list(response.keys())}")
        else:
            print(f"[DEBUG] extract_info - Parsed value (non-dict): {response!r}")
        
        # Try to get the event list
        if isinstance(response, dict):
            # Check for JSON schema format with $ref references
            if "$defs" in response and "properties" in response:
                print(f"[DEBUG] extract_info - Detected JSON schema format with $defs")
                # Try to resolve $ref references
                if "properties" in response and "event" in response["properties"]:
                    event_list = response["properties"]["event"]
                    if isinstance(event_list, list) and len(event_list) > 0:
                        resolved_events = []
                        for event_item in event_list:
                            if isinstance(event_item, dict) and "$ref" in event_item:
                                # Resolve $ref: "#/$defs/details" -> look in $defs.details
                                ref_path = event_item["$ref"]
                                if ref_path.startswith("#/$defs/"):
                                    def_name = ref_path.replace("#/$defs/", "")
                                    if def_name in response["$defs"]:
                                        def_data = response["$defs"][def_name]
                                        # Extract properties from the definition
                                        if "properties" in def_data:
                                            resolved_event = {}
                                            for key, value in def_data["properties"].items():
                                                resolved_event[key] = value
                                            resolved_events.append(resolved_event)
                                            print(f"[DEBUG] extract_info - Resolved $ref to actual event data: {resolved_event}")
                                        else:
                                            # If properties not directly in def_data, use def_data itself
                                            resolved_events.append(def_data)
                                    else:
                                        print(f"[DEBUG] extract_info - WARNING: Could not find definition '{def_name}' in $defs")
                                else:
                                    print(f"[DEBUG] extract_info - WARNING: Unsupported $ref format: {ref_path}")
                            elif isinstance(event_item, dict) and any(field in event_item for field in ["event_name", "date", "time", "venue"]):
                                # Already resolved, use directly
                                resolved_events.append(event_item)
                        if resolved_events:
                            output = resolved_events
                            print(f"[DEBUG] extract_info - Resolved {len(resolved_events)} event(s) from JSON schema")
            
            if "event" in response:
                if output is not None:
                    print(
                        f"[DEBUG] extract_info - NOTE: response also has top-level 'event'; "
                        f"replacing prior output (len was {len(output)}) with response['event']"
                    )
                output = response["event"]
                print(f"[DEBUG] extract_info - branch: used response['event'], type={type(output)}, n={len(output) if isinstance(output, list) else 'n/a'}")
            elif "properties" in response and "event" in response["properties"]:
                # Handle case where event is nested under "properties"
                event_list = response["properties"]["event"]
                # Check if it contains $ref references that need resolution
                if isinstance(event_list, list) and len(event_list) > 0:
                    first_item = event_list[0]
                    if isinstance(first_item, dict) and "$ref" in first_item:
                        # Already handled above in $defs section
                        pass
                    else:
                        output = event_list
                        print(f"[DEBUG] extract_info - branch: properties.event (list), len={len(event_list)}")
            else:
                print(f"[DEBUG] extract_info - WARNING: no top-level 'event' and no usable properties.event")
                print(f"[DEBUG] extract_info - Available keys: {list(response.keys())}")
                # Try to find the data in a different structure
                if "properties" in response:
                    print(f"[DEBUG] extract_info - Checking 'properties' key...")
                    props = response["properties"]
                    if isinstance(props, dict):
                        print(f"[DEBUG] extract_info - Properties keys: {list(props.keys())}")
                        # Check if any property contains a list of events
                        for key, value in props.items():
                            if isinstance(value, list) and len(value) > 0:
                                if isinstance(value[0], dict) and any(field in value[0] for field in ["event_name", "date", "time", "venue"]):
                                    output = value
                                    print(f"[DEBUG] extract_info - branch: properties.{key}, len={len(value)}")
                                    break
                if not output:
                    if len(response) == 1:
                        # Maybe the key is different or it's a list directly
                        first_key = list(response.keys())[0]
                        first_value = response[first_key]
                        if isinstance(first_value, list):
                            output = first_value
                            print(f"[DEBUG] extract_info - branch: single key {first_key!r} -> list, len={len(first_value)}")
                        else:
                            output = [EMPTY_EVENT]
                            print(f"[DEBUG] extract_info - branch: single key {first_key!r} -> not a list, using EMPTY_EVENT")
                    else:
                        output = [EMPTY_EVENT]
                        print(f"[DEBUG] extract_info - branch: fallback EMPTY_EVENT (dict had {len(response)} keys, no event list found)")
        elif isinstance(response, list):
            # Response is already a list
            output = response
            print(f"[DEBUG] extract_info - branch: parser returned list directly, len={len(output)}")
        else:
            print(f"[DEBUG] extract_info - WARNING: Unexpected response type: {type(response)}")
            output = [EMPTY_EVENT]
        
        # Ensure output is set
        if output is None:
            output = [EMPTY_EVENT]
            print(f"[DEBUG] extract_info - branch: output was None -> EMPTY_EVENT")

        for i, ev in enumerate(output):
            if not isinstance(ev, dict):
                print(f"[DEBUG] extract_info - event[{i}] is not a dict: {type(ev).__name__} = {ev!r}")
                continue
            types = {k: type(v).__name__ for k, v in ev.items()}
            sample = {k: (v if isinstance(v, str) else repr(v)[:80]) for k, v in ev.items()}
            print(f"[DEBUG] extract_info - event[{i}] field types: {types}")
            print(f"[DEBUG] extract_info - event[{i}] values (truncated): {sample}")
        
        print(f"[DEBUG] extract_info - final event list length: {len(output)}")
        if not output or (len(output) == 1 and all(not v or v == "" for v in output[0].values())):
            print(f"[DEBUG] extract_info - WARNING: Empty or blank event extracted!")
    except OutputParserException as e:
        # LLM sometimes outputs invalid JSON (e.g. "}. " instead of "}, " between events). Repair and re-parse.
        raw = getattr(e, "llm_output", None)
        if not raw:
            msg = str(e)
            if "Invalid json output:" in msg:
                raw = msg.split("Invalid json output:", 1)[-1].strip()
                if "\nFor troubleshooting" in raw:
                    raw = raw.split("\nFor troubleshooting")[0].strip()
        if not raw and last_raw_text:
            raw = last_raw_text
        print(f"[DEBUG] extract_info - OutputParserException: {e!r}")
        print(f"[DEBUG] extract_info - repair source raw length: {len(raw) if raw else 0}")
        if raw:
            print(f"[DEBUG] extract_info - repair raw preview:\n{_preview_text(str(raw))}")
        events = _parse_events_from_raw(raw) if raw else None
        output = events if events else [EMPTY_EVENT]
        if events:
            print(f"[DEBUG] extract_info - Repaired JSON parse: {len(events)} event(s)")
        else:
            print(f"[DEBUG] extract_info - repair failed or empty; using EMPTY_EVENT")
    except Exception as e:
        print(f"[DEBUG] extract_info - ERROR in event LLM/parse step: {e}")
        import traceback
        print(f"[DEBUG] extract_info - Traceback: {traceback.format_exc()}")
        if last_raw_text:
            print(f"[DEBUG] extract_info - last raw (if any) preview:\n{_preview_text(last_raw_text)}")
        output = [EMPTY_EVENT]
        
    try:
        tags = extract_tags(details, model)
        print(f"[DEBUG] extract_info - Tags extracted: {tags}")
    except Exception as e:
        print(f"[DEBUG] extract_info - ERROR extracting tags: {e}")
        import traceback
        print(f"[DEBUG] extract_info - Tags traceback: {traceback.format_exc()}")
        tags = "其他"
        
    try:
        category = select_category(details, model)
        print(f"[DEBUG] extract_info - Category extracted: {category}")
    except Exception as e:
        print(f"[DEBUG] extract_info - ERROR selecting category: {e}")
        import traceback
        print(f"[DEBUG] extract_info - Category traceback: {traceback.format_exc()}")
        category = ""
    
    return output, tags, category


def extract_info_with_model_fallback(details_str: str, post_date=None):
    """
    Try primary (GPT) OpenRouter models in order. On API error or blank event extraction,
    try the next; if all primaries fail, use DEFAULT_LLM_FALLBACK_MODEL (Gemma).

    Uses the same ``extract_info`` as xplore_automation / crawl (tags + category run on
    the model that produced usable events, or on the fallback).

    Env: LLM_PRIMARY_MODELS=id1,id2  and  LLM_FALLBACK_MODEL=id
    """
    primaries = _primary_models_from_env()
    fallback = _fallback_model_from_env()

    for i, model_id in enumerate(primaries):
        try:
            llm = create_openrouter_chat_llm(model_id)
            response, tags, category = extract_info(details_str, llm, post_date=post_date)
        except Exception as e:
            print(f"  [LLM] {model_id} failed: {e}")
            continue
        if _response_has_usable_events(response):
            if i > 0:
                print(f"  [LLM] ok with {model_id} (after earlier primary(s) failed or were blank)")
            return response, tags, category
        print(f"  [LLM] {model_id} returned no usable event fields; trying next")

    print(f"  [LLM] primary model(s) exhausted; trying fallback {fallback}")
    llm = create_openrouter_chat_llm(fallback)
    response, tags, category = extract_info(details_str, llm, post_date=post_date)
    return response, tags, category


