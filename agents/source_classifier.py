import json
import re
import time

from google import genai
from google.genai import errors as genai_errors

from config.settings import GEMINI_API_KEY, LLM_MODEL

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _chat_once(prompt: str, retries: int = 4, base_delay: float = 5.0) -> str:
    for attempt in range(retries):
        try:
            chat = _get_client().chats.create(model=LLM_MODEL)
            response = chat.send_message(prompt)
            return response.text.strip()
        except genai_errors.ServerError as e:
            if attempt < retries - 1 and e.status_code in (503, 429):
                time.sleep(base_delay * (2 ** attempt))
            else:
                raise


def classify_sources(query: str) -> list[str]:
    source_list = (
        '- "esg": GHG Protocol ESG documents covering greenhouse gas accounting, scope 1 & 2 emissions, product lifecycle, value chain, and carbon accounting standards.\n'
        '- "ifrs": IFRS-18 official International Financial Reporting Standards documentation covering presentation and disclosure in financial statements.\n'
        '- "competitors": Pfizer annual report covering competitor financial performance, strategy, revenue, R&D pipeline, and market position.'
    )

    prompt = f"""You are a query routing assistant for a financial knowledge system.

Available data sources:
{source_list}

User query: "{query}"

Task: Determine which data sources are relevant to answer this query.
Return a JSON array of source keys from ["esg", "ifrs", "competitors"].
Return an empty array [] if no internal source is relevant and a web search would be needed.
Return multiple sources if the query spans multiple domains.

Respond with ONLY the JSON array, nothing else. Examples:
- ["ifrs"]
- ["esg", "ifrs"]
- ["competitors"]
- []
"""

    raw = _chat_once(prompt)
    match = re.search(r"\[.*?\]", raw, re.DOTALL)
    if match:
        try:
            sources = json.loads(match.group())
            valid = {"esg", "ifrs", "competitors"}
            return [s for s in sources if s in valid]
        except json.JSONDecodeError:
            pass
    return []


def check_ambiguity(query: str) -> dict:
    prompt = f"""You are an assistant for a financial knowledge system.

User query: "{query}"

Determine if this query is too ambiguous or unclear to answer without clarification.
A query is ambiguous if it could refer to multiple unrelated topics or critical context is missing.

Respond with JSON in this exact format:
{{"is_ambiguous": true/false, "clarification_question": "your question here or empty string"}}
"""

    raw = _chat_once(prompt)
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            result = json.loads(match.group())
            return {
                "is_ambiguous": bool(result.get("is_ambiguous", False)),
                "clarification_question": result.get("clarification_question", ""),
            }
        except json.JSONDecodeError:
            pass
    return {"is_ambiguous": False, "clarification_question": ""}
