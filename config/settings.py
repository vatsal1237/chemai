"""
Central configuration for the Agentic RAG pipeline.
All tunable parameters, model names, paths, and thresholds live here.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PARSED_DIR = DATA_DIR / "parsed"
CHROMA_DIR = DATA_DIR / "chroma_db"
PARENT_DOCSTORE_PATH = DATA_DIR / "parent_docstore.json"

# Ensure directories exist
PARSED_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# ─── Ollama Configuration ─────────────────────────────────────────────────────
# When running inside Docker Compose the hostname is "ollama".
# When running locally outside Docker, override to http://localhost:11434.
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")

# ─── Local Vision Model (PDF Parsing - No Gemini) ────────────────────────────
# Supports llama3.2-vision or qwen2.5vl running in Ollama
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl")
PARSE_DPI = int(os.getenv("PARSE_DPI", "150"))

# ─── Embedding Model (Ollama) ────────────────────────────────────────────────
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
EMBEDDING_DIM = 768  # nomic-embed-text output dimension

# ─── Parent-Document Retrieval Settings ───────────────────────────────────────
# Child chunks: Small, highly dense searchable blocks (~100 tokens / ~350-400 chars)
CHILD_CHUNK_CHARS = int(os.getenv("CHILD_CHUNK_CHARS", "400"))
CHILD_OVERLAP_CHARS = int(os.getenv("CHILD_OVERLAP_CHARS", "80"))

# Parent chunks: Large context sections (~800-1500 tokens / full reaction mechanisms)
PARENT_MAX_CHARS = int(os.getenv("PARENT_MAX_CHARS", "3500"))

# ─── Retrieval & Agent Settings ──────────────────────────────────────────────
TOP_K = int(os.getenv("TOP_K", "6"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
MAX_REWRITE_RETRIES = int(os.getenv("MAX_REWRITE_RETRIES", "2"))

# ─── LLM Generation ──────────────────────────────────────────────────────────
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.1:8b")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "10"))

# ─── ChromaDB ─────────────────────────────────────────────────────────────────
CHROMA_COLLECTION_NAME = "rag_papers_children"

# ─── System Prompts ───────────────────────────────────────────────────────────
CHEM_SYSTEM_PROMPT = """\
You are an expert scientific AI research assistant specializing in chemistry, materials science, \
and reaction mechanisms. You answer user questions using the provided context, verified chemical databases, \
and conversation history.

STRICT FORMATTING RULES:
1. Always format chemical formulas, structures, and mathematical equations in LaTeX wrapped with dollar signs:
   - Inline math: $\\mathrm{C}_{60}$, $\\mathrm{H}_2\\mathrm{O}$, $E = mc^2$, $\\mathrm{sp}^2$
   - Block equations: $$\\Delta G = \\Delta H - T\\Delta S$$
2. Always use proper subscripts ($_{n}$) and superscripts ($^{n}$) for chemical elements, oxidation states, and isotopes.
3. For chemical reactions, include state symbols and reaction conditions when available:
   $$\\mathrm{C}_{60} + 3\\,\\mathrm{H}_2 \\xrightarrow{\\text{catalyst}} \\mathrm{C}_{60}\\mathrm{H}_6$$
4. Ground all statements in the retrieved context. If information is not in the context or chemical database, state clearly what is unknown.
"""

CHEM_AGENT_MODEL_NAME = "chem-rag-agent"

# ─── Knowledge Gap Finder ────────────────────────────────────────────────────
GAP_MAX_PAPERS = int(os.getenv("GAP_MAX_PAPERS", "15"))       # OpenAlex abstracts read per run
GAP_NUM_QUERIES = int(os.getenv("GAP_NUM_QUERIES", "4"))      # search queries generated in Phase 1
OPENALEX_PER_QUERY = int(os.getenv("OPENALEX_PER_QUERY", "10"))
GAP_LOCAL_SECTIONS = int(os.getenv("GAP_LOCAL_SECTIONS", "3"))  # uploaded-PDF sections added
# Uploaded-PDF sections must be clearly on-topic (nomic-embed: on-topic ~0.75+, off-topic ~0.55-0.62)
GAP_LOCAL_MIN_SIM = float(os.getenv("GAP_LOCAL_MIN_SIM", "0.70"))

GAP_SYSTEM_PROMPT = """\
You are a meticulous research analyst performing a systematic literature review. \
You only state what the provided paper data supports, you cite paper IDs exactly as given, \
and when asked for JSON you output valid JSON only, with no commentary.
"""
