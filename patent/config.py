import atexit
import os
from pathlib import Path
import shutil

from dotenv import load_dotenv
from loguru import logger

# ── Data-chunking constants ─────────────────────────────────────────────────
# Every pipeline stage that processes data in batches reads this value.
# Lower it for memory-constrained VMs (e.g. 50_000 for 4 GiB RAM);
# raise it for throughput on large machines (e.g. 200_000+).
# The peak RSS per scoring chunk scales at ~130 bytes/row, so:
#    50_000 rows  →  ~0.6 GiB    100_000 rows →  ~1.2 GiB
#   150_000 rows  →  ~1.8 GiB    200_000 rows →  ~2.4 GiB
CHUNK_SIZE = 100_000

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# ── Project-local temporary directory ─────────────────────────────────────
# All pipeline scratch work (memmaps, intermediate arrays, etc.) lives
# inside this directory so it never touches the system /tmp.
TMP_DIR = PROJ_ROOT / ".tmp"


def project_tempdir() -> Path:
    """Create a unique scratch directory inside the project-local ``.tmp/``.

    Returns a :class:`pathlib.Path` to a freshly-created directory whose
    name includes a timestamp + random hex suffix for uniqueness.
    Callers are responsible for removing the directory when done
    (usually via ``shutil.rmtree`` in a ``finally`` block).

    This completely avoids :func:`tempfile.mkdtemp` collisions with the
    system-wide temporary directory.
    """
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    ts = _timestamp_hex()
    rand = os.urandom(4).hex()
    tmpdir = TMP_DIR / f"{ts}_{rand}"
    tmpdir.mkdir(parents=False, exist_ok=False)
    return tmpdir


def cleanup_project_temp() -> None:
    """Remove the entire project ``.tmp/`` directory and all contents.

    Safe to call at any time — is a no-op if the directory does not
    exist.  Registered as an :mod:`atexit` handler so the project temp
    area is automatically cleaned when the Python process exits normally.
    """
    if TMP_DIR.exists():
        shutil.rmtree(TMP_DIR, ignore_errors=True)
        logger.debug(f"Cleaned up project temp directory: {TMP_DIR}")


def _timestamp_hex() -> str:
    """Compact timestamp for directory naming (avoids colons and spaces)."""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%dT%H%M%S")


# Clean up any orphaned temp directories from a previous crashed run
cleanup_project_temp()

# Register cleanup to run at normal process exit
atexit.register(cleanup_project_temp)

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
