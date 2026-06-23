"""Belief-based determinization for search.

`cg.api.search_begin` forks the real engine from the agent's point of view but needs
us to *fill in the hidden information*: our own hidden deck/prize ordering, and the
opponent's deck, hand, prizes and (if face-down) active Pokemon. The quality of search
is bounded by the quality of these guesses -- filling the opponent's deck with a single
filler card (as the reference `simple_mcts.py` does) throws away search's whole point.

This module instead maintains a *posterior over which archetype the opponent is piloting*
(the decklists in `training_opponents/`), inferred from the opponent cards we have already
seen, and samples a single consistent full state -- one "determinization" -- per call.
Run search over K independent determinizations and average to get a PIMC / determinized-UCT
estimate that marginalizes over the hidden state.

At training time the opponent really is one of these archetypes, so the posterior is
well-specified; at inference the same machinery degrades gracefully (it falls back to the
best-overlap archetype, padding with basic energy) if the opponent is off-distribution.
"""

import glob
import os
import random
from collections import Counter

from cg.api import AreaType, CardType, Observation
from features import card_table

_OPP_DECK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "training_opponents")

# A guaranteed-legal basic Energy to pad with when an off-archetype opponent leaves us
# short of cards (Basic Psychic Energy). Decks may hold unlimited copies of basic energy.
_PAD_ENERGY_ID = 5


def _is_basic_energy(cid: int) -> bool:
    c = card_table.get(cid)
    return c is not None and c.cardType == CardType.BASIC_ENERGY


def _is_basic_pokemon(cid: int) -> bool:
    c = card_table.get(cid)
    return c is not None and c.cardType == CardType.POKEMON and c.basic


class Archetype:
    """A 60-card opponent decklist plus its card multiset, for posterior matching."""

    def __init__(self, label: str, deck: list[int]):
        self.label = label
        self.deck = list(deck)
        self.counts = Counter(deck)
        self.basic_pokemon = [cid for cid in set(deck) if _is_basic_pokemon(cid)]


def load_archetypes(include_own_deck: str | None = None) -> list[Archetype]:
    """Load `training_opponents/*.csv` (and optionally our own deck for the mirror) as the
    set of decks the opponent might be on. Decks that aren't exactly 60 cards are skipped."""
    archetypes: list[Archetype] = []
    if include_own_deck and os.path.exists(include_own_deck):
        cards = [int(x) for x in open(include_own_deck).read().split("\n") if x.strip()]
        if len(cards) >= 60:
            archetypes.append(Archetype("__mirror__", cards[:60]))
    for path in sorted(glob.glob(os.path.join(_OPP_DECK_DIR, "*.csv"))):
        try:
            cards = [int(x) for x in open(path).read().split("\n") if x.strip()]
        except ValueError:
            continue
        if len(cards) >= 60:
            archetypes.append(Archetype(os.path.basename(path)[:-4], cards[:60]))
    if not archetypes:
        raise RuntimeError(f"no archetype decks found in {_OPP_DECK_DIR}")
    return archetypes


# ---------------------------------------------------------------------------
# Counting cards we can already see (so we don't "deal" them twice).
# ---------------------------------------------------------------------------
def _add_pokemon(counter: Counter, p) -> None:
    """A Pokemon in play accounts for its own card plus everything stacked on it
    (energy, tools, pre-evolution cards) -- all part of that player's 60."""
    if p is None:
        return
    counter[p.id] += 1
    for c in p.energyCards:
        counter[c.id] += 1
    for c in p.tools:
        counter[c.id] += 1
    for c in p.preEvolution:
        counter[c.id] += 1


def _visible_counter(obs: Observation, player_index: int, include_hand: bool) -> Counter:
    """Multiset of that player's cards we can currently see. For the opponent the hand is
    None (hidden) and prizes are face-down, so only board + discard (+ owned stadium) count."""
    state = obs.current
    ps = state.players[player_index]
    counter: Counter = Counter()
    for a in ps.active:
        _add_pokemon(counter, a)
    for b in ps.bench:
        _add_pokemon(counter, b)
    for c in ps.discard:
        counter[c.id] += 1
    if include_hand and ps.hand is not None:
        for c in ps.hand:
            counter[c.id] += 1
    for c in state.stadium:
        if c.playerIndex == player_index:
            counter[c.id] += 1
    return counter


def _opp_active_facedown(obs: Observation, opp_index: int) -> bool:
    active = obs.current.players[opp_index].active
    return len(active) > 0 and active[0] is None


# ---------------------------------------------------------------------------
# Posterior over archetype + sampling a consistent hidden pool.
# ---------------------------------------------------------------------------
def _match_score(seen: Counter, arch: Archetype) -> tuple[bool, int]:
    """Is `arch` consistent with the opponent cards we've seen (every seen card fits,
    counting basic energy as unbounded)? Also return overlap size as a fallback tiebreak."""
    consistent = True
    overlap = 0
    for cid, k in seen.items():
        have = arch.counts.get(cid, 0)
        overlap += min(k, have)
        if k > have and not _is_basic_energy(cid):
            consistent = False
    return consistent, overlap


def _choose_archetype(obs: Observation, opp_index: int,
                      archetypes: list[Archetype], rng: random.Random) -> Archetype:
    seen = _visible_counter(obs, opp_index, include_hand=False)
    consistent = [a for a in archetypes if _match_score(seen, a)[0]]
    if consistent:
        return rng.choice(consistent)
    # Off-distribution opponent: fall back to the best-overlap archetype.
    return max(archetypes, key=lambda a: _match_score(seen, a)[1])


def _hidden_pool(arch: Archetype, seen: Counter) -> list[int]:
    """Cards of the chosen archetype not yet visible -- the pool to deal the opponent's
    hidden deck/hand/prizes from."""
    remaining = arch.counts.copy()
    for cid, k in seen.items():
        remaining[cid] -= k  # may go negative for over-seen basics; floored below
    pool: list[int] = []
    for cid, k in remaining.items():
        pool.extend([cid] * max(0, k))
    return pool


def _fit_length(pool: list[int], n: int, rng: random.Random) -> list[int]:
    """Force `pool` to exactly length n: pad with basic energy, or trim at random."""
    pool = list(pool)
    if len(pool) < n:
        pool.extend([_PAD_ENERGY_ID] * (n - len(pool)))
    elif len(pool) > n:
        rng.shuffle(pool)
        pool = pool[:n]
    return pool


def make_determinization(obs: Observation, our_deck: list[int],
                         archetypes: list[Archetype], rng: random.Random) -> dict:
    """Sample one consistent full state for `search_begin`.

    Returns a dict with the keyword arguments search_begin expects, plus `opp_full_deck`
    (the sampled opponent decklist, used as the deck-identity feature when the encoder runs
    from the opponent's point of view inside search).
    """
    state = obs.current
    my_index = state.yourIndex
    opp_index = 1 - my_index
    my_ps = state.players[my_index]
    opp_ps = state.players[opp_index]

    # --- our own hidden cards: 60 minus everything we can see = deck + face-down prizes ---
    my_seen = _visible_counter(obs, my_index, include_hand=True)
    my_hidden = Counter(our_deck)
    my_hidden.subtract(my_seen)
    my_pool = []
    for cid, k in my_hidden.items():
        my_pool.extend([cid] * max(0, k))
    rng.shuffle(my_pool)
    my_prize_n = len(my_ps.prize)
    my_deck_n = my_ps.deckCount
    my_pool = _fit_length(my_pool, my_prize_n + my_deck_n, rng)
    your_prize = my_pool[:my_prize_n]
    your_deck = my_pool[my_prize_n:my_prize_n + my_deck_n]

    # --- opponent: pick an archetype from the posterior, deal its hidden cards ---
    arch = _choose_archetype(obs, opp_index, archetypes, rng)
    opp_seen = _visible_counter(obs, opp_index, include_hand=False)
    facedown_active = _opp_active_facedown(obs, opp_index)
    opp_deck_n = opp_ps.deckCount
    opp_hand_n = opp_ps.handCount
    opp_prize_n = len(opp_ps.prize)
    need = opp_deck_n + opp_hand_n + opp_prize_n + (1 if facedown_active else 0)

    pool = _fit_length(_hidden_pool(arch, opp_seen), need, rng)
    rng.shuffle(pool)

    opponent_active: list[int] = []
    if facedown_active:
        # Active must be a basic Pokemon: pull one out of the pool (or from the archetype).
        bi = next((i for i, c in enumerate(pool) if _is_basic_pokemon(c)), None)
        if bi is not None:
            opponent_active = [pool.pop(bi)]
        elif arch.basic_pokemon:
            opponent_active = [rng.choice(arch.basic_pokemon)]
            pool.pop()  # keep total accounting right

    opponent_hand = pool[:opp_hand_n]
    opponent_prize = pool[opp_hand_n:opp_hand_n + opp_prize_n]
    opponent_deck = pool[opp_hand_n + opp_prize_n:]

    # Engine requires >=1 Basic Pokemon in the opponent deck (for setup/draws).
    if opp_deck_n > 0 and not any(_is_basic_pokemon(c) for c in opponent_deck):
        if arch.basic_pokemon:
            opponent_deck[rng.randrange(len(opponent_deck))] = rng.choice(arch.basic_pokemon)

    return {
        "your_deck": your_deck,
        "your_prize": your_prize,
        "opponent_deck": opponent_deck,
        "opponent_prize": opponent_prize,
        "opponent_hand": opponent_hand,
        "opponent_active": opponent_active,
        "opp_full_deck": arch.deck,
    }
