"""Central configuration. Import this module FIRST in every entry point.

It sets the telemetry-off environment variables *before* crewai / chromadb are
imported, so `crew.kickoff()` never makes an outbound network call.
"""
import os
from pathlib import Path

# --- zero-network guarantees (must be set before importing crewai/chromadb) ---
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_DISABLE_VERSION_CHECK", "true")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")  # chromadb
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
TRANSCRIPT_DIR = BASE_DIR / "transcripts"
LOG_DIR.mkdir(exist_ok=True)
TRANSCRIPT_DIR.mkdir(exist_ok=True)
REQUEST_LOG = LOG_DIR / "requests.jsonl"
CALIBRATION_FILE = TRANSCRIPT_DIR / "calibration.json"

# --- modes ---
# MOCK_LLM=true (default): deterministic, keyless, networkless. Graded mode.
MOCK_LLM = os.getenv("MOCK_LLM", "true").strip().lower() not in ("0", "false", "no")
# EMBED_BACKEND=auto -> SentenceTransformers (the graded setting).
# EMBED_BACKEND=hash -> offline smoke-test embedder, NOT for graded transcripts.
EMBED_BACKEND = os.getenv("EMBED_BACKEND", "auto").strip().lower()
ST_MODEL_NAME = os.getenv("ST_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

# --- RAG ---
CHUNK_SIZE_CHARS = 320      # fixed-size strategy
CHUNK_OVERLAP_CHARS = 80
TOP_K = 3

# --- runtime governance ---
MAX_REQUEST_TOKENS = 6000          # per-request token budget (input + model calls)
COST_PER_1K_TOKENS_USD = 0.002     # illustrative price used for the cost cap
MAX_REQUEST_COST_USD = MAX_REQUEST_TOKENS / 1000 * COST_PER_1K_TOKENS_USD
