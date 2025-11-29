"""Tests for src.utils.user_input."""

import pytest
from src.utils.user_input import ask_yes_no, ask_integer, ask_float, ask_choice

class TestUserInput:
    def test_ask_yes_no(self, mocker):
        """Test yes/no input."""
        mock_input = mocker.patch("builtins.input", side_effect=["y", "n", "", "invalid", "y"])

        assert ask_yes_no("q1") is True
        assert ask_yes_no("q2") is False
        assert ask_yes_no("q3", default=True) is True
        # "invalid" -> print error -> loop -> "y"
        assert ask_yes_no("q4") is True

    def test_ask_integer(self, mocker):
        """Test integer input."""
        mock_input = mocker.patch("builtins.input", side_effect=["10", "", "5", "15", "invalid", "10"])

        assert ask_integer("q1", 5) == 10
        assert ask_integer("q2", 5) == 5

        # Test bounds
        # min 8, max 12
        # "5" -> < min -> loop
        # "15" -> > max -> loop
        # "invalid" -> error -> loop
        # "10" -> ok
        assert ask_integer("q3", 10, min_val=8, max_val=12) == 10

    def test_ask_float(self, mocker):
        """Test float input."""
        mock_input = mocker.patch("builtins.input", side_effect=["10.5", "", "5.0", "15.0", "invalid", "10.0"])

        assert ask_float("q1", 5.0) == 10.5
        assert ask_float("q2", 5.0) == 5.0

        # Test bounds
        # min 8, max 12
        # "5.0" -> < min -> loop
        # "15.0" -> > max -> loop
        # "invalid" -> error -> loop
        # "10.0" -> ok
        assert ask_float("q3", 10.0, min_val=8.0, max_val=12.0) == 10.0

    def test_ask_choice(self, mocker):
        """Test choice input."""
        choices = ["apple", "banana", "cherry"]

        # 1. Select by number
        mocker.patch("builtins.input", side_effect=["1"])
        assert ask_choice(choices, "Choose") == "apple"

        # 2. Select by name
        mocker.patch("builtins.input", side_effect=["banana"])
        assert ask_choice(choices, "Choose") == "banana"

        # 3. Default
        mocker.patch("builtins.input", side_effect=[""])
        assert ask_choice(choices, "Choose", default="cherry") == "cherry"

        # 4. Partial match
        mocker.patch("builtins.input", side_effect=["che"])
        assert ask_choice(choices, "Choose") == "cherry"

        # 5. Invalid/Ambiguous handling
        # "z" -> unknown
        # "10" -> invalid number
        # "apple" -> valid
        mocker.patch("builtins.input", side_effect=["z", "10", "apple"])
        assert ask_choice(choices, "Choose") == "apple"
