"""Tests for src.utils.__init__."""

import sys
import pytest
from src.utils import setup_project_path

class TestInit:
    def test_setup_project_path(self, mocker):
        """Test project path setup."""
        # Mock sys.path
        mock_path = ["/some/path"]
        mocker.patch("sys.path", mock_path)

        setup_project_path()

        # Should add project root to path
        # Since I cannot easily know what __file__ resolves to in test env vs function env without running it
        # But I can check if something was added
        assert len(mock_path) > 1

        # Check idempotency
        # If I run it again, it shouldn't add it again if it's already there?
        # The code checks `if str(project_root) not in sys.path`.

        # Let's mock the path resolution
        # Path(__file__).parent.parent -> src/
        # .parent -> root

        current_len = len(mock_path)
        setup_project_path()
        assert len(mock_path) == current_len # Should be idempotent if path was added
