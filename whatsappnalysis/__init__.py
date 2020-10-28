import sys
import datetime
from pathlib import Path

import nltk
from loguru import logger


nltk.download("vader_lexicon")
nltk.download("punkt")

# Configure loguru
log_file_name = f"log_{datetime.datetime.now().isoformat(sep='_')}.log"
log_file_path = Path(__file__).parent.parent / log_file_name
logger.add(sys.stderr)
logger.add(sys.stdout)
logger.add(log_file_path, rotation="500 MB")
