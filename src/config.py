from pathlib import Path
from dataclasses import dataclass

ROOT_DIR = Path(__file__).parent.parent


@dataclass
class PipelineConfig:
    """Pipeline configuration setup"""

    input_chat_text_path: Path = ROOT_DIR / "data" / "01_raw" / "jm.txt"

    run_loader: bool = False
    loaded_chat_parquet_path: Path = ROOT_DIR / "data" / "02_loaded" / "jm.parquet"

    run_cleaner: bool = False
    cleaned_chat_parquet_path = ROOT_DIR / "data" / "03_cleaned" / "jm.parquet"

    run_features: bool = True
    features_chat_parquet_path = ROOT_DIR / "data" / "04_featured" / "jm.parquet"


PIPELINE_CONFIG = PipelineConfig()
