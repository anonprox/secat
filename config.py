"""
config.py — all settings in one place.
Change model, task count, max turns here.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Always load the project-local .env, independent of the caller's current
# working directory and without requiring shell exports. Environment variables
# already set by the user still take precedence.
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(dotenv_path=PROJECT_ROOT / ".env", override=False)

# API Keys / provider settings
OPENAI_API_KEY   = os.getenv("OPENAI_API_KEY")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
TMDB_API_KEY     = os.getenv("TMDB_API_KEY")

# Keep subprocesses / executed agent code consistent with config-loaded values.
# This is useful when running `python run_experiment.py ...` directly instead
# of sourcing .env in the shell first.
if OPENAI_API_KEY:
    os.environ.setdefault("OPENAI_API_KEY", OPENAI_API_KEY)
if DEEPSEEK_API_KEY:
    os.environ.setdefault("DEEPSEEK_API_KEY", DEEPSEEK_API_KEY)
os.environ.setdefault("DEEPSEEK_BASE_URL", DEEPSEEK_BASE_URL)
if TMDB_API_KEY:
    os.environ.setdefault("TMDB_API_KEY", TMDB_API_KEY)

# Model — change this to swap models (e.g. "gpt-5.4-mini", "gpt-5.4")
DEFAULT_MODEL = os.getenv("MODEL", "gpt-5.4-mini")

# Experiment settings
MAX_TURNS        = 10
MAX_TASKS        = 30
CODE_TIMEOUT_SEC = 30

# Log injection mode
# "ast"   — accurate static analysis via Python AST (recommended default)
# "regex" — legacy regex pattern matching (kept only for comparison)
# "llm"   — uses an LLM to insert logs (extra cost/latency, non-deterministic)
# "wrapper" — (DEFAULT) logs at the library boundary; agent code runs UNMODIFIED.
#             Captures every API call regardless of whether the agent printed it.
#             Most faithful and robust; no AST/LLM edge cases.
LOG_INJECTION_MODE = "wrapper"

# Execution model for agent code:
#   "script" (default) = exec() with a shared namespace. A trailing bare
#       expression prints NOTHING (this exposes the S4 perception failure).
#   "kernel" = a real IPython/Jupyter kernel, exactly like the original CodeAct.
#       A trailing bare expression IS auto-displayed (so the agent "sees" it).
# Run both to demonstrate, as a controlled experiment, that kernel execution
# removes the S4 perception-failure class. Requires: pip install jupyter_client ipykernel
EXECUTION_MODE = "kernel"

# Max chars of an API response body / variable value captured into [S3] logs.
# The grounding check (answer-value-in-response) needs full bodies, so keep this
# large. TMDB responses are typically 2-15 KB. Set lower only to shrink logs.
LOG_BODY_MAX_CHARS = 12000

# TMDB
TMDB_BASE_URL = "https://api.themoviedb.org/3"

# Paths
DATA_DIR    = "data"

# Output organization. All run artifacts live under RESULTS_BASE, which the runner
# sets per run to results/<benchmark>/<agent>/ so multiple datasets and agents stay
# separated. set_output_base() updates the derived dirs below.
RESULTS_BASE = "logs"
RUN_DIR     = "logs/runs"
ERROR_DIR   = "logs/errors"
SUMMARY_DIR = "logs/summary"
RESULT_DIR  = str(PROJECT_ROOT / "results")

def set_output_base(base):
    """Point all run-output dirs under `base` (e.g. results/tmdb/codeact)."""
    global RESULTS_BASE, RUN_DIR, ERROR_DIR, SUMMARY_DIR
    RESULTS_BASE = base
    RUN_DIR     = f"{base}/runs"
    ERROR_DIR   = f"{base}/errors"
    SUMMARY_DIR = f"{base}/summary"
    return base

TMDB_TASKS_FILE = "data/tmdb_tasks.json"

# Registered agents — add new ones here
AGENTS = {
    "codeact":    "agents.codeact_agent",
    "toolcoder":  "agents.toolcoder_agent",
    "oca":        "agents.oca_agent",
}

# ── OCA (Observation-Carrying Answers) — RQ3 agent toggles ────────────────────
# The artifact contains the paper configuration only.
OCA_VERSION            = "v2"
OCA_BUILD_ID           = "oca-v4.1.63-semantic-preservation-certificate-20260912"
OCA_RESULT_TAG         = "oca_generic"
OCA_PHASE_ISOLATION    = True
OCA_LEDGER_CITATIONS   = True
OCA_MAX_PHASE_A_TURNS  = 1
# Generic execution-cost guards. A single plan step may fan out only when its
# validated binding is ``selected_all``; even then the trusted runtime caps the
# number of HTTP calls. These limits are API-agnostic and prevent one generated
# loop from turning a small evaluation into hundreds of requests.
OCA_MAX_FANOUT_CALLS = 30
OCA_MAX_API_CALLS_PER_TASK = 50

# Evaluation isolation.  The clean default never exposes benchmark solutions,
# route lists, expected answers, or benchmark-specific task identifiers to an
# agent.  ``reported95`` is available only through an explicit runner flag to
# preserve the old assistance boundary; it does not revert current validators.
EVALUATION_PROFILE = "isolated"       # isolated | reported95
OCA_LEGACY_BENCHMARK_HINTS = False
OCA_SEMANTIC_ADAPTERS = False

# OCA v2 planning and deterministic compilation.
OCA_PLANNER_ENABLED       = True
# v4.1: the model reads language into typed IR; provider capabilities and OAS
# compile that IR deterministically into routes/bindings/derivations.
OCA_INTENT_COMPILER_ENABLED = True
OCA_INTENT_PLANNER_MAX_TOKENS = 900
OCA_INTENT_REPAIR_ATTEMPTS = 1  # compiler-informed semantic retry only on failed typed compilation
OCA_INTENT_LEGACY_FALLBACK = False
OCA_LEGACY_PLANNER_FALLBACK_ATTEMPTS = 1  # compatibility only; disabled above
OCA_PLANNER_ATTEMPTS      = 2  # retained for direct legacy/tests; not the normal v4 deterministic path
OCA_GENERIC_PLAN_CRITIC = True   # compact answer-free relation-fidelity review; one bounded correction on clear mismatch
OCA_SELECTION_SEMANTIC_GUARD = False  # optional reviewer only; unanchored positive ordinals are checked deterministically
OCA_SELECTION_SEMANTIC_GUARD_MAX_TOKENS = 220
OCA_PLAN_COMPLETION_REPAIR = False
OCA_PLAN_REPAIR_STEPS     = 1
# Action plans historically reached certification before the generic evidence
# repair block.  Keep one bounded remaining-step repair enabled so a valid partial
# action is retried before fail-closed abstention.  The runtime guard still allows
# only requests licensed by the validated plan and observed bindings.
OCA_ACTION_COMPLETION_REPAIR = True
OCA_ACTION_COMPLETION_REPAIR_STEPS = 3
OCA_DETERMINISTIC_PLAN_COMPLETION = True
OCA_DETERMINISTIC_PLAN_COMPLETION_MAX_CALLS = 6
# Historical host-first switch retained for compatibility. The model-first v4.1
# path does not mark plans as deterministic-front-end plans, so CodeAct remains
# in the action-generation loop after typed semantic planning.
OCA_DETERMINISTIC_FRONTEND_HOST_FIRST = False
OCA_DETERMINISTIC_FRONTEND_HOST_MAX_CALLS = 30
OCA_LEDGER_MAX_CHARS      = 24000
OCA_PHASE_B_MAX_TOKENS    = 600
OCA_SEMANTIC_COMMIT       = False
OCA_SEMANTIC_COMMIT_MAX_TOKENS = 220
OCA_ANSWER_REPAIR         = False
OCA_DETERMINISTIC_LINEAGE_COMPLETION = False
OCA_STRUCTURAL_FALLBACK = False
OCA_DETERMINISTIC_ANSWER_FIRST = True
OCA_EARLY_STOP_ON_PLAN_COMPLETE = True

# Token-efficiency controls. ``baseline`` preserves the published prompt path.
# The runner can apply ``receipt`` or ``lean`` profiles without changing the
# lossless observation ledger, evidence compiler, contract, or verifier.
OCA_TOKEN_PROFILE = "baseline"          # baseline | receipt | adaptive | lean | adaptive_lean
OCA_OBSERVATION_MODE = "raw"            # raw | receipt | adaptive
OCA_RECEIPT_MAX_CHARS = 3500
OCA_LOCAL_OUTPUT_MAX_CHARS = 1200
OCA_PHASE_A_HISTORY = "full"             # full | compact
OCA_PLANNER_CATALOG_MODE = "full"        # full | compact (all routes retained)
OCA_PHASE_B_FOCUS_FALLBACK = False
OCA_PHASE_B_DEP_CONTEXT = 6

# Adaptive observation projection: a second planner sees only the response schemas
# of endpoints already selected by the evidence plan, chooses task-specific paths,
# and a deterministic runtime projector applies them. Runtime repair receives only
# field names/types/counts if the live response shape differs from the OAS.
OCA_DETERMINISTIC_OBSERVATION_PLAN = True
OCA_PROJECTION_PLANNER_ATTEMPTS = 1
OCA_ADAPTIVE_RUNTIME_REPAIR = True
OCA_ADAPTIVE_MAX_RUNTIME_REPAIRS = 1

# Strict commit gate and one bounded evidence repair. In v2, "off" contract mode
# is internally promoted to enforced; use OCA_VERSION="v1" for the ungated core.
OCA_CONTRACT_MODE       = "enforced"  # "off" | "advisory" | "enforced"
OCA_REPAIR_MODE         = "typed"     # "off" | "generic" | "typed" | "shuffled"
OCA_REPAIR_ENDPOINT_HINT = False
OCA_EVIDENCE_REPAIR_STEPS = 1

# Accuracy-first bounded recovery. These are generic and answer-free; they never
# expose benchmark gold. Runtime replanning is read-only only.
OCA_RUNTIME_REPLAN_ATTEMPTS = 1
OCA_RUNTIME_REPLAN_PLANNER_ATTEMPTS = 1
OCA_RUNTIME_REPLAN_MAX_CALLS = 12
OCA_ASSET_OWNER_FALLBACK_CANDIDATES = 8  # bounded alternate search owners for empty visual assets
OCA_EMIT_BEST_EFFORT_AFTER_RECOVERY = True
# Last-resort evidence answer is deliberately compact. It must not re-ingest a
# full raw ledger/plan after a local replay defect.
OCA_BEST_EFFORT_LEDGER_CHARS = 12000
OCA_BEST_EFFORT_PAYLOAD_CHARS = 5000
OCA_BEST_EFFORT_MAX_TOKENS = 300
OCA_ACCURACY_SEMANTIC_REVIEW = True  # selective: only generic high-confidence risk shapes
OCA_ACCURACY_COMPLETION_MAX_CALLS = 30  # extend only finite, already-authorized missing GET fan-out
OCA_RUNTIME_REPLAN_MODEL_STEPS = 2      # only if deterministic replay of a recovered plan stalls

# Legacy flag retained only for CLI/config compatibility; the generic OCA core
# does not apply hidden domain/rating heuristics.
OCA_OUTLIER_RETRY = False

# ── Per-agent strategy knobs ──────────────────────────────────────────────────
# ATC: run a black-box probing pre-step to discover tool schemas before chaining.
ATC_PROBING = True
# CodeTool: candidates sampled per step, selected by on-the-spot reward (executability).
# 1 disables best-of-N (plain stepwise). The trained PRM / latent reward is NOT
# included (see codetool_agent.py); pass latent_reward_fn to run() to add it.
CODETOOL_N_CANDIDATES = 3

# Error type → Scope mapping (perception-action loop; see error_classifier.py)
#   S1_Intention | S2_Expression | S3_Execution | S4_Perception | S5_Integration
SCOPE_MAP = {
    "HTTP_404":              "S2_Expression",    # wrong endpoint/resource
    "HTTP_400":              "S2_Expression",    # malformed request
    "NameError":             "S2_Expression",    # referenced undefined name
    "ImportError":           "S2_Expression",    # wrong library/env assumption
    "HTTP_Auth":             "S3_Execution",     # environment/credential
    "ConnectionError":       "S3_Execution",
    "TypeError":             "S3_Execution",     # ran and crashed on data
    "ValueError":            "S3_Execution",
    "KeyError":              "S3_Execution",
    "IndexError":            "S3_Execution",
    "JSONDecodeError":       "S3_Execution",
    "AttributeError":        "S3_Execution",
    "Timeout":               "S3_Execution",
    "unobserved_result":     "S4_Perception",    # ran but agent didn't see result
    "silent":                "S5_Integration",   # answer wrong despite data
    "fabricated_grounding":  "S5_Integration",   # answer value not in any response
    "no_code_generated":     "S1_Intention",     # never formed intent to act
    "prose_instead_of_code": "S1_Intention",
    "unknown":               "Other",
}
