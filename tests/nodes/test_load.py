from pathlib import Path
from pytest_mock import MockFixture

from whatsappnalysis.nodes.load import run


def test_run(mocker: MockFixture):
    """ Test running loader """
    # Arrange
    MockChatDataset = mocker.patch("whatsappnalysis.nodes.load.ChatDataset", autospec=True)
    input_path = Path('input/path/chat.txt')

    # Act
    run(input_path)

    # Assert
    assert MockChatDataset.return_value.load_from_txt.call_count == 1