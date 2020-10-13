from pathlib import Path

RUN_LOADER = True
ROOT_DIR = Path(__file__).parent.parent
INPUT_CHAT_TEXT = ROOT_DIR / "data" / "01_raw" / "jm.txt"
LOADED_CHAT_PARQUET = ROOT_DIR / "data" / "02_loaded" / "jm.parquet"