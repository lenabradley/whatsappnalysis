from pytest_mock import MockFixture
from whatsappnalysis.nodes.add_features import run


def test_run(mocker: MockFixture):
    """ Test running add features """
    # Arrange
    mock_dataset = mocker.patch(
        "whatsappnalysis.nodes.add_features.ChatDataset"
    ).return_value
    mock_features = [
        mocker.patch(
            "whatsappnalysis.nodes.add_features._add_overall_sentiment_polarity"
        ),
        mocker.patch("whatsappnalysis.nodes.add_features._add_sentence_tokens"),
        mocker.patch(
            "whatsappnalysis.nodes.add_features._add_token_sentiment_polarities"
        ),
        mocker.patch("whatsappnalysis.nodes.add_features._add_word_count"),
    ]

    # Act
    run(mock_dataset)

    # Assert
    for mock_feature in mock_features:
        assert mock_feature.call_count == 1
