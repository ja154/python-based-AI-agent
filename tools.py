from __future__ import annotations

from datetime import datetime
import re

from langchain_community.tools import DuckDuckGoSearchRun, WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_core.tools import Tool

DEFAULT_SEARCH_MAX_CHARS = 2200
DEFAULT_WIKI_MAX_CHARS = 1600

_search_client = DuckDuckGoSearchRun()
_wiki_client = WikipediaQueryRun(
    api_wrapper=WikipediaAPIWrapper(top_k_results=2, doc_content_chars_max=2200)
)


def _trim_text(value: str, max_chars: int) -> str:
    value = str(value).strip()
    if len(value) <= max_chars:
        return value
    return value[:max_chars] + f"\n... ({len(value) - max_chars} more characters)"


def save_to_txt(data: str, filename: str = "research_output.txt") -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as file_obj:
        file_obj.write(formatted_text)

    return f"Data successfully saved to {filename}"


def web_search_text(query: str, max_chars: int = DEFAULT_SEARCH_MAX_CHARS) -> str:
    if not str(query).strip():
        return "Search skipped: empty query."

    try:
        result = _search_client.run(str(query).strip())
    except Exception as exc:
        return f"Web search unavailable: {exc}"

    return _trim_text(result, max_chars=max_chars)


def wiki_lookup_text(query: str, max_chars: int = DEFAULT_WIKI_MAX_CHARS) -> str:
    if not str(query).strip():
        return "Wikipedia lookup skipped: empty query."

    try:
        result = _wiki_client.run(str(query).strip())
    except Exception as exc:
        return f"Wikipedia lookup unavailable: {exc}"

    return _trim_text(result, max_chars=max_chars)


def extract_urls(text: str, max_items: int = 8) -> list[str]:
    if not text:
        return []

    raw_urls = re.findall(r"https?://[^\s\]\)\}>\"']+", text)
    seen: set[str] = set()
    urls: list[str] = []
    for url in raw_urls:
        normalized = url.rstrip(".,);")
        if normalized in seen:
            continue
        seen.add(normalized)
        urls.append(normalized)
        if len(urls) >= max_items:
            break
    return urls


save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Save text to a local file. Input should be the full text to persist.",
)

search_tool = Tool(
    name="search_web",
    func=web_search_text,
    description="Search the web for a query string and return compact text findings.",
)

wiki_tool = Tool(
    name="search_wikipedia",
    func=wiki_lookup_text,
    description="Look up a topic on Wikipedia and return a compact summary.",
)
