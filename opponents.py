"""Load the hand-coded example_ai bots as opponents for evaluation and training.

Each example_ai/<name>_ai.py is a self-contained module exposing:
  * my_deck: list[int]      -- its 60-card decklist
  * agent(obs_dict)->list[int]

They load their deck CSV via a RELATIVE path at import time and keep per-game state
in MODULE-LEVEL globals (reset when state.turn == 0). Consequences we must respect:
  * import them with cwd = example_ai/ so the relative CSV path resolves;
  * a given module is a SINGLE stateful instance -- never run two games that use the
    same bot concurrently in one process (training here is sequential, so fine).
"""

import contextlib
import importlib
import os
import sys

_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "example_ai")

# module name -> short label
_BOTS = {
    "dragapult_example_ai": "dragapult",
    "abomasnow_ai": "abomasnow",
    "iono_ai": "iono",
}


@contextlib.contextmanager
def _in_dir(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


class Opponent:
    """Wraps a scripted bot: its decklist + its agent callable."""

    def __init__(self, label: str, deck: list[int], agent_fn):
        self.label = label
        self.deck = deck
        self.agent = agent_fn


def load_opponents() -> list[Opponent]:
    if _DIR not in sys.path:
        sys.path.insert(0, _DIR)
    opps = []
    with _in_dir(_DIR):  # so each bot's relative CSV load succeeds at import
        for mod_name, label in _BOTS.items():
            mod = importlib.import_module(mod_name)
            opps.append(Opponent(label, list(mod.my_deck), mod.agent))
    return opps
