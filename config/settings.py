import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

LLM_MODEL = "gemini-3.6-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

CHROMA_PERSIST_DIR = str(BASE_DIR / "chroma_db")

DATA_DIRS = {
    "esg": str(BASE_DIR / "data" / "esg"),
    "ifrs": str(BASE_DIR / "data" / "ifrs"),
    "competitors": str(BASE_DIR / "data" / "compititors"),
}

COLLECTION_NAMES = {
    "esg": "esg_collection",
    "ifrs": "ifrs_collection",
    "competitors": "competitor_collection",
}

SOURCE_DESCRIPTIONS = {
    "esg": "GHG Protocol ESG documents covering greenhouse gas accounting, scope 1 & 2 emissions, product lifecycle, value chain, and carbon accounting standards.",
    "ifrs": "IFRS-18 official International Financial Reporting Standards documentation covering presentation and disclosure in financial statements.",
    "competitors": "Pfizer annual report covering competitor financial performance, strategy, revenue, R&D pipeline, and market position.",
}

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K_RETRIEVAL = 5
