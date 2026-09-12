import os

# ============================================================
# TELECOM AI KNOWLEDGE ORCHESTRATION — RUNTIME CONFIG
# ============================================================

APP_NAME = "Telecom AI Knowledge Orchestration Runtime"

# Hosted model provider
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "").strip()

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Primary generator
GENERATOR_MODEL = "google/gemma-4-26b-a4b-it"

# Knowledge router
ROUTER_MODEL = "ibm-granite/granite-4.2-8b"

# Runtime behaviour
DEFAULT_TIMEZONE = os.getenv(
    "DEFAULT_TIMEZONE",
    "Europe/London",
)

MAX_OUTPUT_TOKENS = 1800

# Connected knowledge retrieval
RAG_TOP_K = 5
MCP_TOP_K = 5
MAX_RETRIEVAL_ROUNDS = 3
MAX_EVIDENCE_ITEMS = 5
MAX_EVIDENCE_CHARS = 2500
MAX_CONTEXT_CHARS = 12500

# Hybrid fusion
RRF_K = 60
NEAR_DUPLICATE_THRESHOLD = 0.88

# Live external grounding
MAX_WEB_OPERATIONS = 3

# ============================================================
# VALIDATION
# ============================================================

def validate_runtime_config() -> None:
    """
    Validate only configuration required for hosted generation.
    Large RAG/MCP assets are validated later by their own modules.
    """

    if not OPENROUTER_API_KEY:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not configured. "
            "Set it as a local environment variable or deployment secret."
        )