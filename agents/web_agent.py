import time

from google import genai
from google.genai import errors as genai_errors
from duckduckgo_search import DDGS

from config.settings import GEMINI_API_KEY, LLM_MODEL

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def run(query: str) -> dict:
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        web_context = ""
        sources = []
        for r in results:
            title = r.get("title", "")
            url = r.get("href", "")
            body = r.get("body", "")
            web_context += f"\n[{title}] ({url})\n{body}\n"
            sources.append({"title": title, "url": url})

        if not web_context.strip():
            return {
                "answer": "Web search returned no results for this query.",
                "sources": [],
                "visual_content": [],
                "source_type": "web",
            }

        prompt = (
            f"Using the following web search results, answer the user's question.\n"
            f"Cite the source titles and URLs for each claim.\n\n"
            f"Web search results:\n{web_context}\n\n"
            f"Question: {query}"
        )

        chat = _get_client().chats.create(model=LLM_MODEL)
        response = None
        for attempt in range(4):
            try:
                response = chat.send_message(prompt)
                break
            except genai_errors.ServerError as e:
                if attempt < 3 and e.status_code in (503, 429):
                    time.sleep(5.0 * (2 ** attempt))
                else:
                    raise

        return {
            "answer": response.text,
            "sources": sources,
            "visual_content": [],
            "source_type": "web",
        }

    except Exception as e:
        return {
            "answer": f"Web search failed: {str(e)}",
            "sources": [],
            "visual_content": [],
            "source_type": "web",
        }
