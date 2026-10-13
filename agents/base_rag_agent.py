import base64
import io
import time
from typing import Any

from google import genai
from google.genai import errors as genai_errors
from google.genai.types import Content, GenerateContentConfig, Part
from PIL import Image

from config.settings import GEMINI_API_KEY, LLM_MODEL
from rag.retriever import retrieve

_client = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def _b64_to_pil(image_b64: str) -> Image.Image | None:
    try:
        img_bytes = base64.b64decode(image_b64)
        return Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        return None


def _build_content_parts(chunks: list[dict]) -> tuple[list[Any], list[dict]]:
    parts: list[Any] = []
    visual_content: list[dict] = []

    for chunk in chunks:
        chunk_type = chunk.get("chunk_type", "text")
        text = chunk.get("text", "")
        file_name = chunk.get("file_name", "")
        page = chunk.get("page_number", 0)

        if chunk_type == "image" and chunk.get("image_data"):
            pil_img = _b64_to_pil(chunk["image_data"])
            if pil_img:
                parts.append(pil_img)
                parts.append(f"[Image from {file_name}, page {page}: {text}]")
                visual_content.append({
                    "type": "image",
                    "caption": text,
                    "file_name": file_name,
                    "page": page,
                    "image_data": chunk["image_data"],
                })
            else:
                parts.append(f"[Image from {file_name}, page {page}: {text}]")
        elif chunk_type == "table":
            parts.append(f"[Table from {file_name}, page {page}]\n{text}")
            visual_content.append({
                "type": "table",
                "content": text,
                "file_name": file_name,
                "page": page,
            })
        else:
            parts.append(f"[From {file_name}, page {page}]\n{text}")

    return parts, visual_content


def run_rag_agent(
    query: str,
    source_type: str,
    source_label: str,
    conversation_history: list[dict] | None = None,
) -> dict:
    chunks = retrieve(query, source_type)

    if not chunks:
        return {
            "answer": f"No relevant information found in {source_label} documents.",
            "sources": [],
            "visual_content": [],
            "source_type": source_type,
        }

    context_parts, visual_content = _build_content_parts(chunks)

    sources = []
    seen = set()
    for chunk in chunks:
        key = f"{chunk['file_name']}:p{chunk['page_number']}"
        if key not in seen:
            seen.add(key)
            sources.append({
                "file_name": chunk["file_name"],
                "page_number": chunk["page_number"],
                "source_type": source_type,
            })

    system_instruction = (
        f"You are a specialized {source_label} document analyst for a financial firm. "
        "Answer questions using ONLY the provided document excerpts. "
        "Cite the file name and page number for each claim. "
        "For tables, reference specific rows/columns. "
        "For images/charts, describe what they show and how it answers the question. "
        "If the documents don't contain enough information, say so clearly. "
        "Be precise and professional."
    )

    client = _get_client()

    gemini_history: list[Content] = []
    if conversation_history:
        for msg in conversation_history[-4:]:
            role = "model" if msg["role"] == "assistant" else "user"
            gemini_history.append(
                Content(role=role, parts=[Part(text=str(msg["content"]))])
            )

    final_parts = context_parts + [
        f"\nBased on the above {source_label} documents, answer this question:\n{query}"
    ]

    config = GenerateContentConfig(system_instruction=system_instruction)

    chat = client.chats.create(
        model=LLM_MODEL,
        history=gemini_history,
        config=config,
    )
    for attempt in range(4):
        try:
            response = chat.send_message(final_parts)
            break
        except genai_errors.ServerError as e:
            if attempt < 3 and e.status_code in (503, 429):
                time.sleep(5.0 * (2 ** attempt))
            else:
                raise

    return {
        "answer": response.text,
        "sources": sources,
        "visual_content": visual_content,
        "source_type": source_type,
    }
