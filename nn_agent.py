"""Submission-side agent: load weights and pick an action from an observation.

main.py imports load_model, policy_step (and mcts_agent only if USE_SEARCH).
At inference the policy is greedy over legal candidate actions -- no forward search,
so no hidden opponent state ever needs to be guessed.
"""

import os

import torch

from cg.api import to_observation_class
from features import (
    MyModel,
    enumerate_actions,
    evaluate,
    get_decoder_input,
    get_encoder_input,
)

# Inference-time search budget (used only when main.py sets USE_SEARCH = True).
# Larger = stronger but slower per move; tune to the competition's per-move time limit.
SEARCH_SIMS = 24            # PUCT simulations per determinization
SEARCH_DETERMINIZATIONS = 4  # belief samples averaged (PIMC width)

_archetypes = None  # lazily-loaded opponent-deck prior (see belief.load_archetypes)
_rng = None

# Architecture must match what ppo_training.py saved. Kept in the checkpoint so we
# can rebuild the right shape regardless of hyperparameter changes.
_DEFAULT_ARCH = dict(d_model=128, num_heads=2, d_feedforward=256,
                     num_layers_encoder=2, num_layers_decoder=2)


def load_model(weights_path: str, device: torch.device) -> MyModel:
    ckpt = torch.load(weights_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        arch = ckpt.get("arch", _DEFAULT_ARCH)
        state = ckpt["state_dict"]
    else:
        arch = _DEFAULT_ARCH
        state = ckpt
    model = MyModel(**arch).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


@torch.inference_mode()
def policy_step(obs_dict: dict, your_deck: list[int], model: MyModel) -> list[int]:
    """Greedy action selection: highest-logit legal candidate."""
    obs = to_observation_class(obs_dict)
    device = next(model.parameters()).device
    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, your_deck)
    sv_dec = get_decoder_input(obs, actions)
    _value, logits = evaluate(model, sv_enc, sv_dec, device)
    best = int(torch.argmax(logits[: len(actions)]).item())
    return actions[best]


def _ensure_belief(your_deck: list[int]):
    """Lazily build the opponent-archetype prior. Returns None if the archetype decks
    aren't shipped alongside the agent (so we can fall back to the model-free policy)."""
    global _archetypes, _rng
    if _archetypes is None:
        import random

        from belief import Archetype, load_archetypes
        try:
            deck_csv = "deck.csv" if os.path.exists("deck.csv") else \
                "/kaggle_simulations/agent/deck.csv"
            _archetypes = load_archetypes(
                include_own_deck=deck_csv if os.path.exists(deck_csv) else None)
        except Exception:
            # No archetype pack available -> mark as "empty" so we stop retrying.
            _archetypes = []
        # Always include our own deck as a possible (mirror) archetype.
        if _archetypes is not None and not any(
                a.label == "__mirror__" for a in _archetypes):
            _archetypes = [Archetype("__mirror__", your_deck)] + list(_archetypes)
        _rng = random.Random(0)
    return _archetypes or None


def mcts_agent(obs_dict: dict, your_deck: list[int], model: MyModel) -> list[int]:
    """Determinized PUCT at inference. Falls back to the greedy policy on the initial
    deck-selection observation, when no archetype prior is available, or on any search
    error -- the submission must never crash."""
    if obs_dict.get("select") is None:
        return your_deck  # initial selection: return the 60-card deck
    archetypes = _ensure_belief(your_deck)
    if archetypes is None:
        return policy_step(obs_dict, your_deck, model)
    try:
        from mcts import run_mcts
        action, _pi, _actions = run_mcts(
            obs_dict, your_deck, model, archetypes,
            n_sims=SEARCH_SIMS, n_determinizations=SEARCH_DETERMINIZATIONS,
            rng=_rng, add_noise=False, temperature=0.0)
        return action
    except Exception:
        return policy_step(obs_dict, your_deck, model)
