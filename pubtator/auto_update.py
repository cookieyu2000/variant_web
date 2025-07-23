import os
import json
import logging
from apscheduler.schedulers.background import BackgroundScheduler
from .config          import FULLTEXT_DIR, PMID_LIST_FILE, Hours, Minutes, DATA_DIR
from .pub_inference   import do_inference_for_variant
from .parser_utils    import sanitize_filename
from .file_utils      import load_all_pmids

logger = logging.getLogger(__name__)

# Where we persist our admin’s auto-update settings
CONFIG_PATH = os.path.join(DATA_DIR, "auto_update_config.json")

def load_auto_config():
    """
    Load auto-update settings from disk (enabled/hour/minute).
    If missing, return defaults from config.py.
    """
    defaults = {"enabled": True, "hour": Hours, "minute": Minutes}
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, "r") as f:
                cfg = json.load(f)
                return { **defaults, **cfg }
        except Exception:
            logger.exception("Failed to load auto_update_config.json, using defaults")
    return defaults

def save_auto_config(cfg: dict):
    """Persist auto-update settings to disk."""
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump(cfg, f)

def auto_update_variants():
    """
    Iterate over all variants in pmid_list.json, and re‐run
    do_inference_for_variant for each cached full_text JSON.
    """
    cfg = load_auto_config()
    if not cfg["enabled"]:
        logger.info("Auto-update is disabled; skipping")
        return

    logger.info("🔄 Auto-update: scanning local full_text…")
    try:
        files = os.listdir(FULLTEXT_DIR)
    except FileNotFoundError:
        logger.warning(f"{FULLTEXT_DIR} does not exist, skipping auto-update")
        return

    cached = { os.path.splitext(fn)[0] for fn in files if fn.endswith(".json") }
    all_pmids = load_all_pmids(PMID_LIST_FILE)

    for variant in all_pmids:
        san = sanitize_filename(variant)
        if san in cached:
            try:
                logger.info(f"  ▶ updating {variant}")
                do_inference_for_variant(
                    variant,
                    base_output_dir=FULLTEXT_DIR,
                    pmid_list_file=PMID_LIST_FILE
                )
            except Exception:
                logger.exception(f"  ❌ failed to update {variant}")
    logger.info("🔄 Automatic Update: Complete")

def start_scheduler():
    """
    Configure and start the APScheduler job that runs
    auto_update_variants daily at the configured time.
    """
    cfg = load_auto_config()
    scheduler = BackgroundScheduler(timezone="Asia/Taipei")
    scheduler.add_job(
        auto_update_variants,
        trigger="cron",
        hour=cfg["hour"],
        minute=cfg["minute"],
        id="auto_update_variants",
        replace_existing=True,
        misfire_grace_time=60 * 60 * 3,  # 3 hours grace time
    )
    scheduler.start()
    logger.info(f"Scheduler started: auto-update at {cfg['hour']:02d}:{cfg['minute']:02d} daily (enabled={cfg['enabled']})")
    return scheduler
