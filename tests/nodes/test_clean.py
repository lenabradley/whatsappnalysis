from pytest_mock import MockFixture

from whatsappnalysis.nodes.clean import run


def test_run(mocker: MockFixture):
    """ Test running cleaning """
    # Arrange
    mock_dataset = mocker.patch("whatsappnalysis.nodes.clean.ChatDataset").return_value
    mock_clean = mocker.patch("whatsappnalysis.nodes.clean._clean_dataset")

    # Act
    run(mock_dataset)

    # Assert
    assert mock_clean.call_count == 1
