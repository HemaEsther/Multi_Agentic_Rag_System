# Multi-Agentic Multimodal RAG System

An internal GenAI assistant for financial firms that answers complex questions from ESG, IFRS, and competitor documents using multiple specialized agents orchestrated by LangGraph.

---

## Architecture

```
User Query
    ↓
[LangGraph Orchestrator]
    ├── Ambiguity Check (Gemini)
    ├── Source Classification Agent (Gemini)
    ↓ (parallel branches)
[ESG Agent]  [IFRS Agent]  [Competitor Agent]
    ↓              ↓               ↓
         [Answer Synthesizer (Gemini)]
                   ↓
         [Streamlit Web UI]
              ↓ (if no internal sources)
         [Web Search Agent (DuckDuckGo)]
```

## Tech Stack

| Component       | Technology                        |
|----------------|-----------------------------------|
| LLM             | Google Gemini 3.6 Flash (free)   |
| Embeddings      | sentence-transformers (local)     |
| Vector Database | ChromaDB (local)                  |
| PDF Parsing     | docling (multimodal)              |
| Orchestration   | LangGraph                         |
| Web UI          | Streamlit                         |
| Web Search      | DuckDuckGo (free, no API key)     |

---

## Prerequisites

- Python 3.11+
- VS Code with Python extension
- A free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

---

## Setup & Run (VS Code Terminal)

### Step 1 — Activate the virtual environment

```bash
.venv\Scripts\activate
```

### Step 2 — Install all dependencies

```bash
pip install -r requirements.txt
```

> Note: `docling` is already installed. This step adds LangGraph, ChromaDB, Streamlit, Gemini SDK, etc.

### Step 3 — Create your `.env` file

```bash
copy .env.example .env
```

Then open `.env` in VS Code and replace `your_gemini_api_key_here` with your actual Gemini API key.

```
GEMINI_API_KEY=AIza...your_key_here
```

### Step 4 — Ingest documents into ChromaDB (run once)

```bash
python -m rag.ingestion
```

This parses all PDFs in `data/esg/`, `data/ifrs/`, and `data/compititors/` using docling and stores chunks (text + tables + images) in a local ChromaDB database at `./chroma_db/`.

> This step takes 5–15 minutes depending on your machine. It only needs to run once — results are persisted.

### Step 5 — Launch the Streamlit web app

```bash
streamlit run web/app.py
```

Opens automatically at `http://localhost:8501`

---

## Alternative: CLI Mode (for backend testing)

```bash
python main.py
```

Supports the same query flow as the web app but in your terminal. Type `ingest` to re-run ingestion, `quit` to exit.

---

## Project Structure

```
multi_agent_rag/
├── data/
│   ├── esg/             # GHG Protocol PDFs (10 files)
│   ├── ifrs/            # IFRS-18 PDF
│   └── compititors/     # Pfizer annual report
│
├── config/
│   └── settings.py      # API keys, model names, data paths
│
├── rag/
│   ├── ingestion.py     # Docling PDF parsing → ChromaDB
│   ├── embeddings.py    # sentence-transformers wrapper
│   └── retriever.py     # ChromaDB semantic search
│
├── agents/
│   ├── source_classifier.py  # Routes query to correct data source(s)
│   ├── base_rag_agent.py     # Shared ReAct RAG logic (text + tables + images)
│   ├── esg_agent.py          # ESG/GHG Protocol specialist
│   ├── ifrs_agent.py         # IFRS Standards specialist
│   ├── competitor_agent.py   # Competitor annual reports specialist
│   └── web_agent.py          # DuckDuckGo web search fallback
│
├── graph/
│   └── orchestrator.py  # LangGraph state machine
│
├── web/
│   └── app.py           # Streamlit chat UI
│
├── main.py              # CLI entry point
├── requirements.txt     # Python dependencies
├── .env.example         # Environment variable template
└── README.md            # This file
```

---

## Sample Queries to Try

```
What are the latest changes in IFRS-18 related to presentation of financial statements?

How does GHG Protocol define Scope 2 emissions and what are the accounting methods?

What is Pfizer's R&D investment and pipeline strategy from their 2020-2021 annual report?


What are IFRS lease accounting requirements and how does Pfizer handle lease liabilities?

Compare ESG reporting standards with financial disclosure requirements under IFRS.
```

---

## Features

- **Multi-source routing** — automatically detects which documents are relevant per query
- **Parallel agents** — ESG, IFRS, and Competitor agents run concurrently via LangGraph
- **Multimodal answers** — extracts and displays tables and images from PDFs inline
- **Web fallback** — if no internal source matches, prompts user to approve a web search
- **Human-in-the-loop** — feedback buttons (helpful/not helpful) after every answer
- **Follow-up questions** — each answer ends with a relevant suggested follow-up
- **Clarification prompts** — ambiguous queries trigger a clarification request
- **Conversation memory** — multi-turn context maintained across the session

---

## Getting Your Free Gemini API Key

1. Go to [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **Create API key**
4. Copy the key and paste it into your `.env` file

The free tier allows up to 15 requests/minute and 1 million tokens/day with Gemini 1.5 Flash.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `GEMINI_API_KEY not set` | Ensure `.env` file exists and has your key |
| `No module named 'google.generativeai'` | Run `pip install -r requirements.txt` |
| `Collection is empty` | Run `python -m rag.ingestion` first |
| Streamlit port already in use | Run `streamlit run web/app.py --server.port 8502` |
| DuckDuckGo rate limit error | Wait 30 seconds and retry the web search |
