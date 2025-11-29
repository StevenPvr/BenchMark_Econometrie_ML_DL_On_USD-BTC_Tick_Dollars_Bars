"""User input utilities."""

from __future__ import annotations

from typing import TypeVar, Sequence

T = TypeVar("T")

def ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Ask user for a yes/no answer.

    Args:
        prompt: Question to ask.
        default: Default value if input is empty.

    Returns:
        True for yes, False for no.
    """
    default_str = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_str}]: ").strip().lower()
        if not answer:
            return default
        if answer in ("y", "yes", "oui", "o"):
            return True
        if answer in ("n", "no", "non"):
            return False
        print("Please answer 'y' or 'n'")


def ask_integer(prompt: str, default: int, min_val: int | None = None, max_val: int | None = None) -> int:
    """Ask for an integer value.

    Args:
        prompt: Prompt message.
        default: Default value.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).

    Returns:
        Selected integer.
    """
    range_str = ""
    if min_val is not None and max_val is not None:
        range_str = f" ({min_val}-{max_val})"

    while True:
        answer = input(f"{prompt}{range_str} [{default}]: ").strip()
        if not answer:
            return default
        try:
            val = int(answer)
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid integer")


def ask_float(prompt: str, default: float, min_val: float | None = None, max_val: float | None = None) -> float:
    """Ask for a float value."""
    while True:
        answer = input(f"{prompt} [{default}]: ").strip()
        if not answer:
            return default
        try:
            val = float(answer)
            if min_val is not None and val < min_val:
                print(f"Value must be >= {min_val}")
                continue
            if max_val is not None and val > max_val:
                print(f"Value must be <= {max_val}")
                continue
            return val
        except ValueError:
            print("Please enter a valid number")


def ask_choice(choices: Sequence[str], prompt: str = "Choose an option", default: str | None = None) -> str:
    """Ask user to choose from a list of options.

    Args:
        choices: List of available choices.
        prompt: Prompt message.
        default: Default choice (must be in choices).

    Returns:
        Selected choice string.
    """
    if not choices:
        raise ValueError("No choices provided")

    default = default or choices[0]

    print(f"\n{prompt}:")
    for i, choice in enumerate(choices, 1):
        marker = " (default)" if choice == default else ""
        print(f"  {i}. {choice}{marker}")

    while True:
        answer = input(f"\nSelect option [1-{len(choices)}] or name [{default}]: ").strip().lower()

        if not answer:
            return default

        # Try as number
        try:
            idx = int(answer)
            if 1 <= idx <= len(choices):
                return choices[idx - 1]
            else:
                print(f"Please enter a number between 1 and {len(choices)}")
                continue
        except ValueError:
            pass

        # Try as name
        if answer in choices:
            return answer

        # Try partial match
        matches = [c for c in choices if c.startswith(answer)]
        if len(matches) == 1:
            return matches[0]
        elif len(matches) > 1:
            print(f"Ambiguous: {matches}. Please be more specific.")
        else:
            print(f"Unknown option: {answer}. Available: {choices}")
