"""From-scratch Expert Iteration (ExIt / AlphaZero-style) training.

No behavioral cloning, no PPO. The policy-improvement operator is *search*: each move in
self-play is chosen by determinized PUCT (`mcts.run_mcts`), which looks ahead through the
real engine and is much stronger than the raw net. We then distill the search back into the
net -- policy head learns the MCTS visit distribution `pi`, value head learns the game
outcome `z`. Next iteration's search starts from the improved net, and the loop bootstraps
upward from random weights.

  net_0 = random init
  repeat:
    self-play (both sides = current net + MCTS, opponents pilot pool decks) -> (features, pi, z)
    train net on  cross-entropy(pi)  +  c * MSE(z)
    gate: greedy-policy win-rate vs the held-out example_ai bots; promote best.pth on gain

The learner always pilots the submission deck (deck.csv) and only learner-side decisions are
collected, so all training data is about playing *our* deck well. The example_ai bots are a
HELD-OUT test set here (never trained against), used only to gate checkpoints.

Run:  python exit_training.py --iters 20 --games 200 --sims 32 --determinizations 4
Resume / warmstart:  python exit_training.py --resume out/last.pth
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from belief import load_archetypes
from cg.game import battle_finish, battle_select, battle_start
from features import (
    MAX_ACTIONS,
    MyModel,
    enumerate_actions,
    evaluate,
    get_decoder_input,
    get_encoder_input,
    num_words_encoder,
)
from mcts import run_mcts

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_deck(path="deck.csv"):
    return [int(x) for x in open(path).read().split("\n") if x.strip()][:60]


class Sample:
    __slots__ = ("sv_enc", "sv_dec", "n_actions", "pi", "to_move", "value")

    def __init__(self, sv_enc, sv_dec, n_actions, pi, to_move):
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
        self.n_actions = n_actions
        self.pi = pi
        self.to_move = to_move
        self.value = 0.0


# ---------------------------------------------------------------------------
# Self-play
# ---------------------------------------------------------------------------
def play_one_game(game_idx, deck, model, archetypes, pool, sims, dets, temp_moves,
                  max_moves, rng):
    """Play one net-vs-net game (MCTS both sides). Returns (learner_samples, result, learner).

    The learner always pilots `deck`; the opponent pilots a random pool deck. Only the
    learner's non-trivial decisions are collected (training is about playing our deck).
    A game exceeding `max_moves` is abandoned (returns no samples) so one pathological game
    can't stall the worker pool at an iteration boundary."""
    from cg.api import to_observation_class

    opp_deck = rng.choice(pool)
    learner = game_idx % 2  # alternate first/second player
    decks = [deck, opp_deck] if learner == 0 else [opp_deck, deck]
    obs, sd = battle_start(decks[0], decks[1])
    if sd.errorPlayer >= 0:
        raise ValueError(f"deck error type {sd.errorType}")

    game_samples = []
    move = 0
    while obs["current"]["result"] < 0 and move < max_moves:
        me = obs["current"]["yourIndex"]
        deck_me = deck if me == learner else opp_deck
        temp = 1.0 if move < temp_moves else 0.0  # explore early, exploit late
        action, pi, actions = run_mcts(
            obs, deck_me, model, archetypes, n_sims=sims, n_determinizations=dets,
            rng=rng, add_noise=True, temperature=temp)
        if me == learner and len(actions) > 1:
            o = to_observation_class(obs)
            game_samples.append(Sample(get_encoder_input(o, deck),
                                       get_decoder_input(o, actions), len(actions), pi, me))
        obs = battle_select(action)
        move += 1

    res = obs["current"]["result"]
    battle_finish()
    if res < 0:
        return [], 2, learner  # abandoned (hit move cap): drop, count as a draw
    for s in game_samples:
        s.value = 0.0 if res == 2 else (1.0 if res == s.to_move else -1.0)
    return game_samples, res, learner


def selfplay_games(n_games, deck, model, archetypes, pool, cfg, rng, log):
    """Single-process self-play (used when --workers <= 1)."""
    data = []
    wins = draws = 0
    t0 = time.time()
    for g in range(n_games):
        samples, res, learner = play_one_game(
            g, deck, model, archetypes, pool, cfg.sims, cfg.determinizations,
            cfg.temp_moves, cfg.max_moves, rng)
        data.extend(samples)
        wins += (res == learner)
        draws += (res == 2)
        if (g + 1) % max(1, n_games // 10) == 0:
            log.info("  selfplay %d/%d | %d samples | learner WR %.0f%% | %.0fs",
                     g + 1, n_games, len(data), 100 * wins / (g + 1), time.time() - t0)
    log.info("collected %d samples from %d games (learner WR %.1f%%, draws %d)",
             len(data), n_games, 100 * wins / n_games, draws)
    return data


# ---------------------------------------------------------------------------
# Parallel self-play: the cg engine is single-battle per process but self-play is
# embarrassingly parallel, so a persistent worker pool divides wall-clock by ~#cores.
# Workers run the net on CPU (small net, avoids GPU contention); the main process
# trains on GPU. Each worker caches the model and reloads weights only when the
# iteration version changes, so we never re-pickle the 50 MB net per task.
# ---------------------------------------------------------------------------
_W = {}


def _worker_init(arch_kwargs, deck):
    import torch as _t
    _t.set_num_threads(1)  # prevent BLAS thread oversubscription across processes
    from belief import load_archetypes
    from features import MyModel as _MyModel
    _W["model"] = _MyModel(**arch_kwargs).eval()
    _W["archetypes"] = load_archetypes(include_own_deck="deck.csv")
    _W["pool"] = [a.deck for a in _W["archetypes"]]
    _W["deck"] = deck
    _W["version"] = -1


def _worker_play(task):
    import random as _random

    import torch as _t
    game_idx, weights_path, version, sims, dets, temp_moves, max_moves, seed = task
    if version != _W["version"]:
        ckpt = _t.load(weights_path, map_location="cpu")
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        _W["model"].load_state_dict(state)
        _W["version"] = version
    rng = _random.Random(seed)
    return play_one_game(game_idx, _W["deck"], _W["model"], _W["archetypes"],
                         _W["pool"], sims, dets, temp_moves, max_moves, rng)


def selfplay_parallel(worker_pool, n_games, weights_path, version, cfg, rng, log):
    """Parallel self-play over a persistent worker pool. Same return shape as selfplay_games."""
    tasks = [(g, weights_path, version, cfg.sims, cfg.determinizations, cfg.temp_moves,
              cfg.max_moves, rng.randrange(2 ** 31)) for g in range(n_games)]
    data = []
    wins = draws = done = 0
    t0 = time.time()
    for samples, res, learner in worker_pool.imap_unordered(_worker_play, tasks):
        data.extend(samples)
        wins += (res == learner)
        draws += (res == 2)
        done += 1
        if done % max(1, n_games // 10) == 0:
            log.info("  selfplay %d/%d | %d samples | learner WR %.0f%% | %.0fs",
                     done, n_games, len(data), 100 * wins / done, time.time() - t0)
    log.info("collected %d samples from %d games (learner WR %.1f%%, draws %d, %.0fs)",
             len(data), n_games, 100 * wins / n_games, draws, time.time() - t0)
    return data


# ---------------------------------------------------------------------------
# Distillation (train net on pi + z)
# ---------------------------------------------------------------------------
def _cat_sparse(svs, words_per):
    index, value, offset = [], [], []
    for sv in svs:
        base = len(index)
        index.extend(sv.index)
        value.extend(sv.value)
        for o in sv.offset:
            offset.append(o + base)
        for _ in range(words_per - len(sv.offset)):
            offset.append(len(index))  # pad to a fixed word count per sample
    return index, value, offset


def train(model, optimizer, data, cfg, log):
    n = len(data)
    order = list(range(n))
    for ep in range(cfg.epochs):
        random.shuffle(order)
        tot_p = tot_v = seen = 0.0
        model.train()
        for s in range(0, n, cfg.batch_size):
            batch = [data[i] for i in order[s:s + cfg.batch_size]]
            ei, ev, eo = _cat_sparse([b.sv_enc for b in batch], num_words_encoder)
            di, dv, do = _cat_sparse([b.sv_dec for b in batch], MAX_ACTIONS)
            value, logits = model(
                torch.tensor(ei, dtype=torch.int32, device=DEVICE),
                torch.tensor(ev, dtype=torch.float32, device=DEVICE),
                torch.tensor(eo, dtype=torch.int32, device=DEVICE),
                torch.tensor(di, dtype=torch.int32, device=DEVICE),
                torch.tensor(dv, dtype=torch.float32, device=DEVICE),
                torch.tensor(do, dtype=torch.int32, device=DEVICE))

            # Soft cross-entropy against the MCTS visit distribution, masked to each
            # sample's candidate set.
            # Finite (not -inf) mask: padded positions have target 0, and 0 * -inf = NaN.
            mask = torch.full((len(batch), MAX_ACTIONS), -1e9, device=DEVICE)
            target = torch.zeros((len(batch), MAX_ACTIONS), device=DEVICE)
            for r, b in enumerate(batch):
                mask[r, : b.n_actions] = 0.0
                target[r, : len(b.pi)] = torch.tensor(b.pi, device=DEVICE)
            logp = F.log_softmax(logits + mask, dim=1)
            pol_loss = -(target * logp).sum(dim=1).mean()

            vtgt = torch.tensor([b.value for b in batch], dtype=torch.float32, device=DEVICE)
            vloss = F.mse_loss(value.squeeze(1), vtgt)
            loss = pol_loss + cfg.value_coef * vloss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = len(batch)
            tot_p += pol_loss.item() * bs
            tot_v += vloss.item() * bs
            seen += bs
        log.info("  epoch %d/%d | policy_ce %.4f | value_mse %.4f",
                 ep + 1, cfg.epochs, tot_p / seen, tot_v / seen)


# ---------------------------------------------------------------------------
# Gating: greedy-policy win-rate vs the held-out scripted bots
# ---------------------------------------------------------------------------
@torch.inference_mode()
def _policy_action(obs_dict, deck, model, temp, rng, device):
    """Pick an action from the raw policy (no search). With temp>0 sample from a softened
    softmax over candidate logits; with temp==0 take the argmax (greedy)."""
    from cg.api import to_observation_class

    o = to_observation_class(obs_dict)
    acts = enumerate_actions(o)
    if len(acts) <= 1:
        return acts[0] if acts else []
    sv_enc = get_encoder_input(o, deck)
    sv_dec = get_decoder_input(o, acts)
    _v, logits = evaluate(model, sv_enc, sv_dec, device)
    logits = logits[: len(acts)]
    if temp and temp > 0:
        probs = torch.softmax(logits / temp, dim=0).tolist()
        return acts[rng.choices(range(len(acts)), weights=probs, k=1)[0]]
    return acts[int(torch.argmax(logits).item())]


@torch.inference_mode()
def evaluate_vs_bots(model, deck, n_games, temp, log, seed=12345):
    """Gate: win-rate vs the held-out scripted bots, policy-only (no search, fast).

    The cg engine is deterministic given fixed agents, so a greedy learner produces
    byte-identical games every iteration -> a frozen gate. Sampling the learner at a low
    temperature (and averaging over more games) de-correlates the games and gives the gate
    real resolution while the policy is still weak."""
    from opponents import load_opponents

    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    opps = load_opponents()
    per = {}
    total_w = total_n = 0
    for opp in opps:
        wins = n = 0
        for g in range(n_games):
            learner = g % 2
            decks = [deck, opp.deck] if learner == 0 else [opp.deck, deck]
            obs, sd = battle_start(decks[0], decks[1])
            if sd.errorPlayer >= 0:
                raise ValueError(f"deck error type {sd.errorType}")
            while obs["current"]["result"] < 0:
                me = obs["current"]["yourIndex"]
                if me == learner:
                    obs = battle_select(_policy_action(obs, deck, model, temp, rng, device))
                else:
                    obs = battle_select(opp.agent(obs))
            res = obs["current"]["result"]
            battle_finish()
            wins += (res == learner)
            n += 1
        per[opp.label] = wins / n
        total_w += wins
        total_n += n
    mean = total_w / total_n
    log.info("eval vs bots (temp %.2f) | mean WR %.1f%% | %s", temp, 100 * mean,
             " ".join(f"{k} {v:.0%}" for k, v in per.items()))
    return mean, per


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=20, help="ExIt iterations")
    ap.add_argument("--games", type=int, default=200, help="self-play games per iteration")
    ap.add_argument("--sims", type=int, default=32, help="PUCT sims per determinization")
    ap.add_argument("--determinizations", type=int, default=4, help="belief samples (PIMC)")
    ap.add_argument("--temp-moves", type=int, default=20,
                    help="learner moves played with exploration temperature before argmax")
    ap.add_argument("--max-moves", type=int, default=1500,
                    help="abandon a self-play game exceeding this many decisions (tail guard)")
    ap.add_argument("--epochs", type=int, default=2, help="training epochs per iteration")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--value-coef", type=float, default=1.0)
    ap.add_argument("--eval-games", type=int, default=30,
                    help="games per held-out bot when gating (0 to skip)")
    ap.add_argument("--eval-temp", type=float, default=0.4,
                    help="learner sampling temperature during eval (0 = greedy/deterministic)")
    # Network architecture (stored in the checkpoint, so inference rebuilds the right shape).
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--d-ff", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3,
                    help="encoder and decoder layers (both set to this)")
    ap.add_argument("--replay-iters", type=int, default=2,
                    help="keep samples from this many recent iterations in the buffer")
    ap.add_argument("--resume", default=None, help="checkpoint to warmstart from")
    ap.add_argument("--workers", type=int, default=0,
                    help="parallel self-play workers (0 = auto = cpu_count-1, 1 = single process)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="out")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("exit")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    log.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    for h in (logging.FileHandler(os.path.join("logs", f"exit_{run_tag}.log")),
              logging.StreamHandler()):
        h.setFormatter(fmt)
        log.addHandler(h)
    log.info("ExIt run %s | device=%s | %s", run_tag, DEVICE, vars(args))

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    arch = dict(d_model=args.d_model, num_heads=args.heads, d_feedforward=args.d_ff,
                num_layers_encoder=args.layers, num_layers_decoder=args.layers)

    deck = load_deck()
    archetypes = load_archetypes(include_own_deck="deck.csv")
    pool = [a.deck for a in archetypes]  # opponents pilot any pool deck (incl. mirror)
    log.info("deck loaded; %d archetypes in belief/opponent pool; arch=%s", len(archetypes), arch)

    model = MyModel(**arch).to(DEVICE)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        log.info("resumed weights from %s", args.resume)
    else:
        log.info("training from scratch (random init)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Spin up the persistent self-play worker pool (once for the whole run).
    import multiprocessing as mp
    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    worker_pool = None
    if n_workers > 1:
        ctx = mp.get_context("spawn")  # required on Windows; safe with CUDA in the parent
        worker_pool = ctx.Pool(n_workers, initializer=_worker_init, initargs=(arch, deck))
        log.info("self-play: %d parallel workers", n_workers)
    else:
        log.info("self-play: single process")
    selfplay_weights = os.path.join(args.out, "_selfplay.pth")

    replay = []
    best_wr = -1.0
    try:
        for it in range(args.iters):
            log.info("=== iteration %d/%d ===", it + 1, args.iters)
            model.eval()
            if worker_pool is not None:
                # Publish current weights for workers to pick up (version = iteration).
                torch.save({"state_dict": model.state_dict(), "arch": arch}, selfplay_weights)
                samples = selfplay_parallel(worker_pool, args.games, selfplay_weights, it,
                                            args, rng, log)
            else:
                samples = selfplay_games(args.games, deck, model, archetypes, pool, args, rng, log)
            replay.append(samples)
            replay = replay[-args.replay_iters:]
            buffer = [s for chunk in replay for s in chunk]
            log.info("training on %d samples (%d recent iters)", len(buffer), len(replay))
            train(model, optimizer, buffer, args, log)

            torch.save({"state_dict": model.state_dict(), "arch": arch},
                       os.path.join(args.out, "last.pth"))
            if args.eval_games > 0:
                mean, _ = evaluate_vs_bots(model, deck, args.eval_games, args.eval_temp, log)
                if mean > best_wr:
                    best_wr = mean
                    torch.save({"state_dict": model.state_dict(), "arch": arch},
                               os.path.join(args.out, "best.pth"))
                    log.info("** new best (WR %.1f%%) -> %s/best.pth", 100 * best_wr, args.out)
    finally:
        if worker_pool is not None:
            worker_pool.close()
            worker_pool.join()
    log.info("done. best held-out WR %.1f%%", 100 * best_wr)


if __name__ == "__main__":
    main()
