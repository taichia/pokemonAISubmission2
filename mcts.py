"""Determinized PUCT search guided by MyModel (the policy/value net).

This is the policy-improvement operator at the heart of the from-scratch Expert-Iteration
plan: it turns the raw net into a much stronger policy by looking ahead through the real
engine (`cg.api.search_begin/search_step`), using the net's value head to evaluate leaves
and its per-candidate-action policy head as the PUCT prior.

Imperfect information is handled by *determinized UCT* (a.k.a. PIMC): we sample K full
states consistent with the observation (`belief.make_determinization`), run an independent
PUCT tree in each, and aggregate the root visit counts. The aggregated visit distribution
is both the action to play (argmax / sampled) and the training target `pi` for distillation.

The net's value head outputs a score in [-1, 1] from the *to-move* player's perspective at
each node; we convert everything to the root player's perspective and back up with negamax
sign handling keyed on `obs.current.yourIndex` (turns are not strictly alternating in this
game -- a player makes many sub-decisions before passing -- so we cannot infer perspective
from tree depth).
"""

import math
import random

import torch

from cg.api import search_begin, search_end, search_step, to_observation_class
from features import (
    enumerate_actions,
    evaluate,
    get_decoder_input,
    get_encoder_input,
)

C_PUCT = 1.5            # exploration constant in the PUCT formula
DIRICHLET_ALPHA = 0.3  # root exploration noise (AlphaZero-style)
DIRICHLET_FRAC = 0.25  # weight of the noise mixed into root priors


class _Node:
    __slots__ = ("to_move", "search_id", "terminal", "leaf_value",
                 "actions", "priors", "N", "W", "children", "to_move_is_root")

    def __init__(self, to_move, search_id, terminal, leaf_value,
                 actions, priors):
        self.to_move = to_move          # obs.current.yourIndex at this node
        self.search_id = search_id      # engine handle for stepping from here
        self.terminal = terminal
        self.leaf_value = leaf_value    # root-perspective value used on first backup
        self.actions = actions          # list[list[int]] candidate option-index lists
        self.priors = priors            # list[float], aligned with actions
        n = len(actions)
        self.N = [0] * n
        self.W = [0.0] * n
        self.children = [None] * n
        self.to_move_is_root = False


def _terminal_value(result: int, root_player: int) -> float:
    if result == 2:        # draw
        return 0.0
    return 1.0 if result == root_player else -1.0


def _make_node(search_state, root_player, decks, model, device):
    """Build a node from an engine search-state: evaluate the net (or detect terminal),
    compute the root-perspective leaf value and the PUCT priors."""
    obs = search_state.observation
    cur = obs.current
    if cur.result >= 0:
        v = _terminal_value(cur.result, root_player)
        return _Node(cur.yourIndex, search_state.searchId, True, v, [], [])

    to_move = cur.yourIndex
    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, decks[to_move])
    sv_dec = get_decoder_input(obs, actions)
    value, logits = evaluate(model, sv_enc, sv_dec, device)
    v_tomove = float(value.item())
    leaf_value = v_tomove if to_move == root_player else -v_tomove

    logits = logits[: len(actions)]
    priors = torch.softmax(logits, dim=0).tolist()
    return _Node(to_move, search_state.searchId, False, leaf_value, actions, priors)


def _add_dirichlet_noise(node: _Node, rng: random.Random) -> None:
    n = len(node.priors)
    if n <= 1:
        return
    noise = [rng.gammavariate(DIRICHLET_ALPHA, 1.0) for _ in range(n)]
    s = sum(noise) or 1.0
    node.priors = [(1 - DIRICHLET_FRAC) * p + DIRICHLET_FRAC * (g / s)
                   for p, g in zip(node.priors, noise)]


def _puct_select(node: _Node) -> int:
    total_n = sum(node.N)
    sqrt_total = math.sqrt(total_n) if total_n > 0 else 1.0
    sign = 1.0 if node.to_move_is_root else -1.0  # set on the node before selection
    best_score, best_a = -1e18, 0
    for a in range(len(node.actions)):
        q = (node.W[a] / node.N[a]) if node.N[a] > 0 else 0.0
        u = C_PUCT * node.priors[a] * sqrt_total / (1 + node.N[a])
        score = sign * q + u
        if score > best_score:
            best_score, best_a = score, a
    return best_a


def _simulate(root, root_player, decks, model, device):
    """One PUCT simulation: descend to an unexpanded edge, expand one node, back up."""
    path = []
    node = root
    while True:
        if node.terminal:
            v = node.leaf_value
            break
        a = _puct_select(node)
        path.append((node, a))
        if node.children[a] is None:
            child_state = search_step(node.search_id, node.actions[a])
            child = _make_node(child_state, root_player, decks, model, device)
            child.to_move_is_root = (child.to_move == root_player)
            node.children[a] = child
            v = child.leaf_value
            break
        node = node.children[a]
    for n, a in path:
        n.N[a] += 1
        n.W[a] += v


@torch.inference_mode()
def run_mcts(obs_dict, our_deck, model, archetypes, *, n_sims, n_determinizations,
             rng, add_noise, temperature, device=None):
    """Run determinized PUCT for the current decision.

    Returns (best_action, pi, actions):
      best_action -- list[int] option indices to play
      pi          -- list[float] normalized visit counts over `actions` (training target)
      actions     -- list[list[int]] the candidate actions pi/best_action index into
    """
    from belief import make_determinization

    if device is None:
        device = next(model.parameters()).device
    obs = to_observation_class(obs_dict)

    # Forced / trivial decisions: no search needed.
    actions0 = enumerate_actions(obs)
    if len(actions0) <= 1:
        return (actions0[0] if actions0 else []), [1.0] * len(actions0), actions0

    root_player = obs.current.yourIndex
    total_N = [0] * len(actions0)

    for _ in range(n_determinizations):
        det = make_determinization(obs, our_deck, archetypes, rng)
        decks = [None, None]
        decks[root_player] = our_deck
        decks[1 - root_player] = det.pop("opp_full_deck")
        try:
            root_state = search_begin(obs, **det)
        except (ValueError, RuntimeError):
            continue  # inconsistent determinization; skip this sample
        root = _make_node(root_state, root_player, decks, model, device)
        root.to_move_is_root = (root.to_move == root_player)
        if root.terminal or not root.actions:
            continue
        if add_noise:
            _add_dirichlet_noise(root, rng)
        for _ in range(n_sims):
            _simulate(root, root_player, decks, model, device)
        # Root actions are the public legal moves -> identical across determinizations,
        # so visit counts are index-aligned and can be summed.
        for a in range(len(total_N)):
            total_N[a] += root.N[a]

    search_end()

    if sum(total_N) == 0:
        # Search produced nothing usable (all determinizations failed) -> net prior.
        sv_enc = get_encoder_input(obs, our_deck)
        sv_dec = get_decoder_input(obs, actions0)
        _value, logits = evaluate(model, sv_enc, sv_dec, device)
        best = int(torch.argmax(logits[: len(actions0)]).item())
        pi = [0.0] * len(actions0)
        pi[best] = 1.0
        return actions0[best], pi, actions0

    if temperature and temperature > 0:
        weights = [n ** (1.0 / temperature) for n in total_N]
        s = sum(weights)
        pi = [w / s for w in weights]
        best_idx = rng.choices(range(len(actions0)), weights=weights, k=1)[0]
    else:
        s = sum(total_N)
        pi = [n / s for n in total_N]
        best_idx = max(range(len(total_N)), key=lambda i: total_N[i])

    return actions0[best_idx], pi, actions0
