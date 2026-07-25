"""Load versioned prompts from the installed package."""

from functools import lru_cache
from importlib.resources import files


@lru_cache
def load_prompt(name: str) -> str:
    """Load and cache a UTF-8 prompt bundled with the EventPilot package."""
    return files("eventpilot.prompts").joinpath(name).read_text(encoding="utf-8").strip()
