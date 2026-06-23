"""Self-play throughput benchmark.

From-scratch ExIt needs a lot of self-play, and each move runs (sims x determinizations)
engine steps + net forward passes. This script measures the rates that decide whether a
from-scratch run converges in days or in months, at a given search budget, so you can size
`--iters`/`--games`/`--sims` before committing.

Run:  python benchmark.py --games 6 --sims 32 --determinizations 4
"""

import argparse
import random
import time

import torch

from belief import load_archetypes
from cg.game import battle_finish, battle_select, battle_start
from features import MyModel
from mcts import run_mcts
from nn_agent import policy_step

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_deck(path="deck.csv"):
    return [int(x) for x in open(path).read().split("\n") if x.strip()][:60]


def bench(model, deck, archetypes, pool, n_games, sims, dets, search, rng):
    """Play n_games (both sides same policy); return (games, total_moves, searched_moves, secs)."""
    moves = searched = 0
    t0 = time.time()
    for g in range(n_games):
        opp = rng.choice(pool)
        decks = [deck, opp] if g % 2 == 0 else [opp, deck]
        obs, sd = battle_start(decks[0], decks[1])
        if sd.errorPlayer >= 0:
            raise ValueError(f"deck error {sd.errorType}")
        while obs["current"]["result"] < 0:
            me = obs["current"]["yourIndex"]
            deck_me = decks[me]
            if search:
                action, _, actions = run_mcts(
                    obs, deck_me, model, archetypes, n_sims=sims,
                    n_determinizations=dets, rng=rng, add_noise=True, temperature=1.0)
                searched += (len(actions) > 1)
            else:
                action = policy_step(obs, deck_me, model)
            obs = battle_select(action)
            moves += 1
        battle_finish()
    return n_games, moves, searched, time.time() - t0


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=6)
    ap.add_argument("--sims", type=int, default=32)
    ap.add_argument("--determinizations", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    deck = load_deck()
    archetypes = load_archetypes(include_own_deck="deck.csv")
    pool = [a.deck for a in archetypes]
    model = MyModel().to(DEVICE)
    model.eval()
    print(f"device={DEVICE}  sims={args.sims}  determinizations={args.determinizations}")

    # Warmup (first call pays import/JIT/caching costs).
    bench(model, deck, archetypes, pool, 1, args.sims, args.determinizations, True, rng)

    print("\n-- policy-only (no search) --")
    g, m, _, s = bench(model, deck, archetypes, pool, args.games, 0, 0, False, rng)
    print(f"  {g} games, {m} moves, {s:.1f}s  ->  {g/s:.2f} games/s, {m/s:.0f} moves/s")

    print("\n-- determinized MCTS self-play (both sides) --")
    g, m, sr, s = bench(model, deck, archetypes, pool, args.games, args.sims,
                        args.determinizations, True, rng)
    sims_total = sr * args.sims * args.determinizations
    print(f"  {g} games, {m} moves ({sr} searched), {s:.1f}s")
    print(f"  -> {g/s:.3f} games/s, {s/g:.1f} s/game, ~{sims_total/s:.0f} sims/s")

    gps = g / s
    print("\n-- projections at this rate (single process) --")
    for n in (1000, 10000, 50000):
        hrs = n / gps / 3600
        print(f"  {n:>6} self-play games  ~  {hrs:5.1f} h  ({hrs/24:.1f} days)")
    print("\nNote: self-play is embarrassingly parallel across processes (the cg engine is")
    print("single-battle per process); divide wall-clock by ~#cores with multiprocessing.")


if __name__ == "__main__":
    main()
