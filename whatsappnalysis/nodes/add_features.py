from typing import List
from enum import Enum, auto

import pandas as pd
from loguru import logger
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from whatsappnalysis.dataset import ChatDataset
from whatsappnalysis.schema import Schema


class _Column(Enum):
    CHAT_NAME = auto()
    TIMESTAMP = auto()
    AUTHOR = auto()
    MESSAGE = auto()
    HAS_MEDIA = auto()
    TOKENS = auto()
    OVERALL_POLARITY = auto()
    NEGATIVE_POLARITY = auto()
    NEUTRAL_POLARITY = auto()
    POSITIVE_POLARITY = auto()
    TOKEN_POLARITIES = auto()
    WORD_COUNT = auto()


_column_to_dtype = {
    _Column.CHAT_NAME.name: str,
    _Column.TIMESTAMP.name: np.dtype("datetime64[ns]"),
    _Column.AUTHOR.name: str,
    _Column.MESSAGE.name: str,
    _Column.HAS_MEDIA.name: bool,

    _Column.OVERALL_POLARITY.name: float,
    _Column.NEGATIVE_POLARITY.name: float,
    _Column.NEUTRAL_POLARITY.name: float,
    _Column.POSITIVE_POLARITY.name: float,

    _Column.WORD_COUNT.name: int,
}

schema = Schema(_Column, _column_to_dtype)


def run(input_dataset: ChatDataset) -> ChatDataset:
    """ Add features to dataset

    Args:
        input_dataset: input chat dataset

    Returns:
        cleaned chat dataset
    """
    logger.info(f"Running node: Add features")
    data = input_dataset.data

    # Overall Sentiment polarity scores
    logger.info(f"Adding sentiment polarity scores")
    data = _add_overall_sentiment_polarity(data)

    # Tokenize
    logger.info(f"Adding tokens")
    data = _add_sentence_tokens(data)

    # Token polarities
    logger.info(f"Adding token polarity")
    data = _add_token_sentiment_polarities(data)

    # Word count
    logger.info(f"Adding word count")
    data = _add_word_count(data)

    output_dataset = ChatDataset(schema=schema).load_from_pandas(data)
    return output_dataset


def _add_overall_sentiment_polarity(data: pd.DataFrame) -> pd.DataFrame:
    """ Add columns measuring overall message polarity """
    sid = SentimentIntensityAnalyzer()
    polarity_data = data.apply(
        _get_message_sentiment_scores,
        result_type='expand',
        axis='columns',
        sid=sid,
    )
    polarity_data.rename(
        {
            0: _Column.OVERALL_POLARITY.name,
            1: _Column.NEGATIVE_POLARITY.name,
            2: _Column.NEUTRAL_POLARITY.name,
            3: _Column.POSITIVE_POLARITY.name
        },
        axis='columns',
        inplace=True
    )
    data = data.join(
        polarity_data,
        how='outer'
    )
    return data


def _get_message_sentiment_scores(row_data: pd.Series, sid: SentimentIntensityAnalyzer):
    """ Analyze sentiment scores

    Args:
        sid: SentimentIntensityAnalyzer
        row_data: row of dataframe from which to get message text

    Returns: List of floats of polrity scores for...
        compound
        negative
        neutral
        positive
    """
    scores = sid.polarity_scores(row_data[_Column.MESSAGE.name])
    return [
        scores['compound'],
        scores['neg'],
        scores['neu'],
        scores['pos'],
    ]


def _add_sentence_tokens(data: pd.DataFrame) -> pd.DataFrame:
    """Add column for sentence tokens"""
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    data[_Column.TOKENS.name] = data[_Column.MESSAGE.name].apply(
        tokenizer.tokenize
    )
    return data


def _add_token_sentiment_polarities(data: pd.DataFrame) -> pd.DataFrame:
    """ Add columns measuring overall message polarity """
    sid = SentimentIntensityAnalyzer()
    data[_Column.TOKEN_POLARITIES.name] = data[_Column.TOKENS.name].apply(
        _get_token_sentiment_scores,
        sid=sid
    )
    return data


def _get_token_sentiment_scores(tokens: List[str], sid: SentimentIntensityAnalyzer):
    """ Analyze overall sentiment scores for each token in the message

    Args:
        tokens: list of string tokens
        sid: SentimentIntensityAnalyzer

    Returns:
        List of floats of overall polarity scores, one for each token
    """
    scores = []
    for token in tokens:
        token_scores = sid.polarity_scores(token)
        scores.append(token_scores["compound"])
    return scores


def _add_word_count(data: pd.DataFrame) -> pd.DataFrame:
    """Add message word count (number of words in message)"""
    data[_Column.WORD_COUNT.name] = data[_Column.MESSAGE.name].apply(
        _count_words
    )
    return data


def _count_words(message: str) -> int:
    """Count the number of words in the given string"""
    return len(message.split(" "))
