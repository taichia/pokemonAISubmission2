"""Relative-strength yardstick: round-robin among checkpoints + anchors, with Elo.

Win-rate vs a single fixed strong opponent is a terrible training signal here — the
scripted bots beat everything (including random) ~97% of the time, so every net piles up
at the ~2% noise floor and the metric is uninformative. The right measure (the AlphaZero
standard) is *relative*: play all agents against each other and fit ratings.

Agents:
  * each checkpoint you pass        -> greedy policy net piloting the submission deck
  * `random`                        -> uniform-random legal moves (the floor anchor)
  * each example_ai scripted bot    -> the strong reference, piloting its own deck

Outputs a pairwise win matrix and Bradley-Terry Elo ratings (random anchored to 1000).

Run:  python arena.py --ckpts out_warm/best.pth out_deep/best.pth out/bc.pth --games 30
"""

import argparse
import math
import random

import torch

from cg.game import battle_finish, battle_select, battle_start
from features import MyModel
from nn_agent import policy_step
from opponents import load_opponents


def load_deck(path="deck.csv"):
    return [int(x) for x in open(path).read().split("\n") if x.strip()][:60]


class Agent:
    def __init__(self, name, deck, act):
        self.name = name
        self.deck = deck
        self.act = act  # act(obs_dict) -> list[int]


def _random_act(obs):
    sel = obs["select"]
    n = len(sel["option"])
    k = sel["maxCount"]
    return random.sample(range(n), min(k, n)) if k > 0 else []


def _net_agent(name, path, deck, device):
    ck = torch.load(path, map_location=device)
    arch = ck["arch"] if isinstance(ck, dict) and "arch" in ck else dict(
        d_model=128, num_heads=2, d_feedforward=256, num_layers_encoder=2, num_layers_decoder=2)
    sd = ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck
    model = MyModel(**arch).to(device)
    model.load_state_dict(sd)
    model.eval()
    return Agent(name, deck, lambda obs: policy_step(obs, deck, model))


def play_game(a, b, a_first):
    """One game; return True if `a` wins (draws count as half elsewhere)."""
    learner = 0 if a_first else 1
    decks = [a.deck, b.deck] if a_first else [b.deck, a.deck]
    obs, sd = battle_start(decks[0], decks[1])
    if sd.errorPlayer >= 0:
        raise ValueError(f"deck error type {sd.errorType}")
    while obs["current"]["result"] < 0:
        me = obs["current"]["yourIndex"]
        actor = a if (me == learner) else b
        obs = battle_select(actor.act(obs))
    res = obs["current"]["result"]
    battle_finish()
    return res, learner  # res: winner index (0/1/2-draw); learner: which seat was `a`


def bradley_terry_elo(names, wins, prior=2.0, iters=2000):
    """Fit BT strengths from a win-count matrix; return {name: elo}, random anchored 1000.
    `prior` adds smoothing games vs a unit-strength dummy so 0-win agents stay finite."""
    n = len(names)
    W = [sum(wins[i]) for i in range(n)]
    p = [1.0] * n
    for _ in range(iters):
        newp = []
        for i in range(n):
            num = W[i] + prior * 0.5
            den = prior / (p[i] + 1.0)
            for j in range(n):
                if j == i:
                    continue
                nij = wins[i][j] + wins[j][i]
                if nij > 0:
                    den += nij / (p[i] + p[j])
            newp.append(num / den if den > 0 else p[i])
        gm = math.exp(sum(math.log(max(x, 1e-9)) for x in newp) / n)
        p = [x / gm for x in newp]
    elo = {names[i]: 400.0 * math.log10(max(p[i], 1e-9)) for i in range(n)}
    if "random" in elo:
        shift = 1000.0 - elo["random"]
        elo = {k: v + shift for k, v in elo.items()}
    return elo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="*", default=[], help="checkpoint paths (net agents)")
    ap.add_argument("--games", type=int, default=30, help="games per ordered pair-direction")
    ap.add_argument("--no-bots", action="store_true", help="exclude scripted bots")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    deck = load_deck()
    agents = []
    for path in args.ckpts:
        name = path.replace("\\", "/").split("/")[-2] if "/" in path else path
        agents.append(_net_agent(name, path, deck, device))
    agents.append(Agent("random", deck, _random_act))
    if not args.no_bots:
        for opp in load_opponents():
            agents.append(Agent(opp.label, opp.deck, opp.agent))

    n = len(agents)
    names = [a.name for a in agents]
    wins = [[0.0] * n for _ in range(n)]  # wins[i][j] = games i beat j (draw = 0.5 each)
    print(f"round-robin: {n} agents, {args.games} games/direction "
          f"({n*(n-1)*args.games} total)\n")
    for i in range(n):
        for j in range(n):
            if i >= j:
                continue
            for g in range(args.games):
                res, seat = play_game(agents[i], agents[j], a_first=(g % 2 == 0))
                if res == 2:
                    wins[i][j] += 0.5
                    wins[j][i] += 0.5
                elif res == seat:
                    wins[i][j] += 1
                else:
                    wins[j][i] += 1
            tot = args.games
            print(f"  {names[i]:>14s} {100*wins[i][j]/tot:3.0f}% - "
                  f"{100*wins[j][i]/tot:3.0f}% {names[j]:<14s}")

    elo = bradley_terry_elo(names, wins)
    print("\n===== Elo (Bradley-Terry, random=1000) =====")
    for name in sorted(elo, key=lambda k: -elo[k]):
        w = sum(wins[names.index(name)])
        g = (n - 1) * args.games
        print(f"  {elo[name]:7.0f}  {name:<16s}  ({100*w/g:.0f}% overall)")


if __name__ == "__main__":
    main()
