from loguru import logger

from src import config
from src.nodes.chat_loader import ChatLoader


def run():

    if config.RUN_LOADER:
        logger.info(f"Running ChatLoader node. Input: {config.INPUT_CHAT_TEXT}; output: {config.LOADED_CHAT_PARQUET}")
        ChatLoader().run(input_file=config.INPUT_CHAT_TEXT, output_file=config.LOADED_CHAT_PARQUET)


if __name__ == '__main__':
    run()
