from patent.config import RAW_DATA_DIR

LAST_UPDATE_FILE = RAW_DATA_DIR / "last_update.txt"


def get_last_update_date() -> str:
    if LAST_UPDATE_FILE.exists():
        with open(LAST_UPDATE_FILE, "r") as f:
            return f.read().strip()
    return None


def set_last_update_date(date_str: str) -> None:
    LAST_UPDATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LAST_UPDATE_FILE, "w") as f:
        f.write(date_str)
