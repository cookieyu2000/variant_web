import os

# ─── Path ────────────────────────────────────────────────
DATA_DIR        = os.getenv("PT_DATA_DIR", "PubTator3_data")
PMID_LIST_FILE  = os.path.join(DATA_DIR, "pmid_list.json")
FULLTEXT_DIR    = os.path.join(DATA_DIR, "full_text")

# ─── Path ───────────────────────────────────────────────
CLASSIFIER_CONFIG_YAML = os.getenv(
    "PT_CLASSIFIER_CONFIG",
    "outputs/BioMedBERTClassifier_256_12_6/config.yaml"
)

# ─── LIME  ──────────────────────────────────────────────────
DEFAULT_NUM_SAMPLES = int(os.getenv("PT_DEFAULT_NUM_SAMPLES", 300))

# ─── Limitations ────────────────────────────────────────────
MAX_INFER_CONNS  = int(os.getenv("PT_MAX_INFER_CONNS", 5))
MAX_STREAM_CONNS = int(os.getenv("PT_MAX_STREAM_CONNS", 3))
STREAM_WORKERS   = int(os.getenv("PT_STREAM_WORKERS", 4))

# ─── Sliding window  ───────────────────────────────────────
WINDOW_SIZE = 512
STRIDE      = 256

# ─── NER  ──────────────────────────────────────────────────────
_NER_REL_PATH = os.getenv("PT_NER_MODEL_DIR", "outputs/ner_weight")
if os.path.isabs(_NER_REL_PATH):
    NER_MODEL_DIR = _NER_REL_PATH
else:
    _pkg_dir      = os.path.dirname(__file__)
    _project_root = os.path.abspath(os.path.join(_pkg_dir, os.pardir))
    NER_MODEL_DIR = os.path.join(_project_root, _NER_REL_PATH)

# ─── Upadte time  ────────────────────────────────────────────────
Hours = int(os.getenv("PT_UPDATE_HOURS", 22))
Minutes = int(os.getenv("PT_UPDATE_MINUTES", 30))