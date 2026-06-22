"""Behavioral-cloning warmstart for the Dragapult policy.

Two PPO runs from scratch plateaued ~15% vs the expert example_ai bots. Instead of
learning from random, we PRETRAIN the policy to imitate the hand-coded dragapult expert
(example_ai/dragapult_example_ai.py), then RL-finetune from these weights.

Pipeline:
  1. Generate supervised data: the dragapult expert plays deck.csv (its internal
     decklist is overridden to match) against the abomasnow/iono bots. At each of the
     expert's decisions we record the encoder features, the enumerated candidate actions,
     and WHICH candidate the expert picked (the classification label). Game outcome is
     the value target.
  2. Supervised-train MyModel: cross-entropy on the policy head (mimic the expert's
     action) + MSE on the value head (predict win/loss).
  3. Save out/bc.pth ({state_dict, arch}) -- load via nn_agent.load_model, or warmstart
     PPO with: ppo_training.py --resume out/bc.pth

Run:  pytorch_env/bin/python bc_pretrain.py --games 3000 --epochs 4
"""

import argparse
import contextlib
import logging
import os
import random
import sys
import time
from datetime import datetime

import torch
import torch.nn.functional as F

import features  # imports cg first so it is cached before we chdir into example_ai
from cg.api import to_observation_class
from cg.game import battle_finish, battle_select, battle_start
from features import (
    MAX_ACTIONS,
    MyModel,
    enumerate_actions,
    get_decoder_input,
    get_encoder_input,
    num_words_encoder,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCH = dict(d_model=128, num_heads=2, d_feedforward=256,
            num_layers_encoder=2, num_layers_decoder=2)
EXAMPLE_DIR = os.path.abspath("example_ai")
LOG_DIR = "logs"


def load_deck(path="deck.csv"):
    return [int(x) for x in open(path).read().split("\n")[:60]]


@contextlib.contextmanager
def _cd(path):
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)


def load_experts(deck):
    """Import the expert + opponent bots; override the dragapult expert's decklist to
    deck.csv so its play/inference matches the submission deck. Returns (drag_module,
    opponents) where opponents is a list of (deck, agent_fn)."""
    if EXAMPLE_DIR not in sys.path:
        sys.path.insert(0, EXAMPLE_DIR)
    with _cd(EXAMPLE_DIR):  # so each bot's relative CSV load works at import
        import dragapult_example_ai as drag
        import abomasnow_ai as abo
        import iono_ai as iono
    drag.my_deck = list(deck)  # expert plays the submission deck
    opponents = [(list(abo.my_deck), abo.agent), (list(iono.my_deck), iono.agent)]
    return drag, opponents


class Sample:
    __slots__ = ("sv_enc", "sv_dec", "n_actions", "label", "value")

    def __init__(self, sv_enc, sv_dec, n_actions, label):
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
        self.n_actions = n_actions
        self.label = label
        self.value = 0.0


def generate(n_games, deck, drag, opponents, log):
    """Play expert-vs-bot games; return supervised Samples (expert decisions only)."""
    data = []
    wins = decisions = skipped = 0
    t0 = time.time()
    for g in range(n_games):
        opp_deck, opp_agent = opponents[g % len(opponents)]
        learner = g % 2  # alternate first/second player
        decks = [deck, opp_deck] if learner == 0 else [opp_deck, deck]
        obs, sd = battle_start(decks[0], decks[1])
        if sd.errorPlayer >= 0:
            raise ValueError(f"deck error type {sd.errorType}")
        game_samples = []
        while obs["current"]["result"] < 0:
            me = obs["current"]["yourIndex"]
            if me == learner:
                sel = drag.agent(obs)
                o = to_observation_class(obs)
                cands = enumerate_actions(o)
                idx = next((i for i, c in enumerate(cands) if c == sorted(sel)), None)
                decisions += 1
                if idx is None:
                    skipped += 1  # expert chose fewer than maxCount (optional skip); not representable
                else:
                    sv_enc = get_encoder_input(o, deck)
                    sv_dec = get_decoder_input(o, cands)
                    game_samples.append(Sample(sv_enc, sv_dec, len(cands), idx))
                obs = battle_select(sel)
            else:
                obs = battle_select(opp_agent(obs))
        res = obs["current"]["result"]
        battle_finish()
        outcome = 0.0 if res == 2 else (1.0 if res == learner else -1.0)
        wins += (res == learner)
        for s in game_samples:
            s.value = outcome
        data.extend(game_samples)
        if (g + 1) % max(1, n_games // 20) == 0:
            log.info("  gen %d/%d games | %d samples | expert WR %.0f%% | %.0fs",
                     g + 1, n_games, len(data), 100 * wins / (g + 1), time.time() - t0)
    log.info("generated %d samples from %d games (expert WR %.1f%%); "
             "skipped %d non-representable decisions (%.1f%%)",
             len(data), n_games, 100 * wins / n_games, skipped,
             100 * skipped / max(1, decisions))
    return data


def _cat_sparse(svs, words_per):
    index, value, offset = [], [], []
    for sv in svs:
        base = len(index)
        index.extend(sv.index)
        value.extend(sv.value)
        for o in sv.offset:
            offset.append(o + base)
        for _ in range(words_per - len(sv.offset)):
            offset.append(len(index))
    return index, value, offset


def train(model, optimizer, data, epochs, batch_size, value_coef, log):
    n = len(data)
    order = list(range(n))
    for ep in range(epochs):
        random.shuffle(order)
        tot_ce = tot_v = correct = seen = 0.0
        model.train()
        for s in range(0, n, batch_size):
            batch = [data[i] for i in order[s:s + batch_size]]
            ei, ev, eo = _cat_sparse([b.sv_enc for b in batch], num_words_encoder)
            di, dv, do = _cat_sparse([b.sv_dec for b in batch], MAX_ACTIONS)
            value, logits = model(
                torch.tensor(ei, dtype=torch.int32, device=DEVICE),
                torch.tensor(ev, dtype=torch.float32, device=DEVICE),
                torch.tensor(eo, dtype=torch.int32, device=DEVICE),
                torch.tensor(di, dtype=torch.int32, device=DEVICE),
                torch.tensor(dv, dtype=torch.float32, device=DEVICE),
                torch.tensor(do, dtype=torch.int32, device=DEVICE))
            mask = torch.full((len(batch), MAX_ACTIONS), float("-inf"), device=DEVICE)
            for r, b in enumerate(batch):
                mask[r, : b.n_actions] = 0.0
            logits = logits + mask
            labels = torch.tensor([b.label for b in batch], device=DEVICE)
            ce = F.cross_entropy(logits, labels)
            vtgt = torch.tensor([b.value for b in batch], dtype=torch.float32, device=DEVICE)
            vloss = F.mse_loss(value.squeeze(1), vtgt)
            loss = ce + value_coef * vloss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(batch)
            tot_ce += ce.item() * bs
            tot_v += vloss.item() * bs
            correct += (logits.argmax(1) == labels).sum().item()
            seen += bs
        log.info("epoch %d/%d | ce %.4f | vloss %.4f | expert-match acc %.1f%%",
                 ep + 1, epochs, tot_ce / seen, tot_v / seen, 100 * correct / seen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=3000, help="expert games to generate")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--value-coef", type=float, default=0.5)
    ap.add_argument("--eval-games", type=int, default=50,
                    help="after BC, evaluate greedy policy vs the bots (0 to skip)")
    ap.add_argument("--out", default="out/bc.pth")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs("out", exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("bc")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(os.path.join(LOG_DIR, f"bc_{run_tag}.log")),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("BC run %s | device=%s | %s", run_tag, DEVICE, vars(args))

    deck = load_deck()
    drag, opponents = load_experts(deck)
    log.info("expert plays deck.csv; opponents: abomasnow, iono")

    data = generate(args.games, deck, drag, opponents, log)
    if not data:
        log.error("no training samples generated")
        return

    model = MyModel(**ARCH).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    train(model, optimizer, data, args.epochs, args.batch_size, args.value_coef, log)

    torch.save({"state_dict": model.state_dict(), "arch": ARCH}, args.out)
    log.info("saved cloned weights -> %s", args.out)

    if args.eval_games > 0:
        from opponents import load_opponents
        import ppo_training
        ppo_training.DEVICE = DEVICE
        model.eval()
        mean, per = ppo_training.evaluate_suite(deck, model, load_opponents(), args.eval_games)
        per_str = " ".join(f"{k} {v:.0%}" for k, v in per.items())
        log.info("BC policy vs bots | WR mean %.1f%% [%s]", 100 * mean, per_str)
        log.info("(prior from-scratch PPO plateaued ~14-17%%; warmstart PPO with "
                 "ppo_training.py --resume %s)", args.out)


if __name__ == "__main__":
    main()
