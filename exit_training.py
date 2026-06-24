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
def _random_legal(obs, rng):
    sel = obs["select"]
    n, k = len(sel["option"]), sel["maxCount"]
    return rng.sample(range(n), min(k, n)) if k > 0 else []


def play_one_game(game_idx, deck, model, archetypes, pool, sims, dets, temp_moves,
                  max_moves, rng, opponents=None, scripted_frac=0.0):
    """Play one game; return (learner_samples, result, learner).

    The learner always pilots `deck` and chooses via MCTS; only its non-trivial decisions
    are collected (training is about playing our deck). The opponent is EITHER a scripted
    bot piloting its own deck (probability `scripted_frac` -- exploitative training against
    the strong reference opponents) OR the current net piloting a random pool deck (mirror-
    style self-play). A game past `max_moves` is abandoned so one pathological game can't
    stall the worker pool."""
    from cg.api import to_observation_class

    use_bot = bool(opponents) and rng.random() < scripted_frac
    if use_bot:
        bot = rng.choice(opponents)
        opp_deck, bot_act = bot.deck, bot.agent
    else:
        opp_deck, bot_act = rng.choice(pool), None

    learner = game_idx % 2  # alternate first/second player
    decks = [deck, opp_deck] if learner == 0 else [opp_deck, deck]
    obs, sd = battle_start(decks[0], decks[1])
    if sd.errorPlayer >= 0:
        raise ValueError(f"deck error type {sd.errorType}")

    game_samples = []
    move = 0
    while obs["current"]["result"] < 0 and move < max_moves:
        me = obs["current"]["yourIndex"]
        if me == learner:
            temp = 1.0 if move < temp_moves else 0.0  # explore early, exploit late
            action, pi, actions = run_mcts(
                obs, deck, model, archetypes, n_sims=sims, n_determinizations=dets,
                rng=rng, add_noise=True, temperature=temp)
            if len(actions) > 1:
                o = to_observation_class(obs)
                game_samples.append(Sample(get_encoder_input(o, deck),
                                           get_decoder_input(o, actions), len(actions), pi, me))
        elif bot_act is not None:
            try:
                action = bot_act(obs)
            except Exception:
                action = _random_legal(obs, rng)  # never let a bot edge-case kill the game
        else:
            temp = 1.0 if move < temp_moves else 0.0
            action, _pi, _a = run_mcts(
                obs, opp_deck, model, archetypes, n_sims=sims, n_determinizations=dets,
                rng=rng, add_noise=True, temperature=temp)
        obs = battle_select(action)
        move += 1

    res = obs["current"]["result"]
    battle_finish()
    if res < 0:
        return [], 2, learner  # abandoned (hit move cap): drop, count as a draw
    for s in game_samples:
        s.value = 0.0 if res == 2 else (1.0 if res == s.to_move else -1.0)
    return game_samples, res, learner


def selfplay_games(n_games, deck, model, archetypes, pool, cfg, rng, log, opponents=None):
    """Single-process self-play (used when --workers <= 1)."""
    data = []
    wins = draws = 0
    t0 = time.time()
    for g in range(n_games):
        samples, res, learner = play_one_game(
            g, deck, model, archetypes, pool, cfg.sims, cfg.determinizations,
            cfg.temp_moves, cfg.max_moves, rng,
            opponents=opponents, scripted_frac=cfg.scripted_frac)
        data.extend(samples)
        wins += (res == learner)
        draws += (res == 2)
        if (g + 1) % max(1, n_games // 10) == 0:
            log.info("  selfplay %d/%d | %d samples | learner WR %.0f%% | %.0fs",
                     g + 1, n_games, len(data), 100 * wins / (g + 1), time.time() - t0)
    log.info("collected %d samples from %d games (learner WR %.1f%%, draws %d)",
             len(data), n_games, 100 * wins / n_games, draws)
    return data, wins / n_games


# ---------------------------------------------------------------------------
# Parallel self-play: the cg engine is single-battle per process but self-play is
# embarrassingly parallel, so a persistent worker pool divides wall-clock by ~#cores.
# Workers run the net on CPU (small net, avoids GPU contention); the main process
# trains on GPU. Each worker caches the model and reloads weights only when the
# iteration version changes, so we never re-pickle the 50 MB net per task.
# ---------------------------------------------------------------------------
_W = {}


def _worker_init(arch_kwargs, deck, scripted_frac):
    import torch as _t
    _t.set_num_threads(1)  # prevent BLAS thread oversubscription across processes
    from belief import load_archetypes
    from features import MyModel as _MyModel
    _W["model"] = _MyModel(**arch_kwargs).eval()
    _W["archetypes"] = load_archetypes(include_own_deck="deck.csv")
    _W["pool"] = [a.deck for a in _W["archetypes"]]
    _W["deck"] = deck
    _W["version"] = -1
    _W["scripted_frac"] = scripted_frac
    if scripted_frac > 0:
        from opponents import load_opponents
        _W["opponents"] = load_opponents()  # per-process bot instances (stateful)
    else:
        _W["opponents"] = None


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
                         _W["pool"], sims, dets, temp_moves, max_moves, rng,
                         opponents=_W["opponents"], scripted_frac=_W["scripted_frac"])


def selfplay_parallel(worker_pool, n_games, weights_path, version, cfg, rng, log):
    """Parallel self-play with a stall watchdog. Returns (samples, learner_wr, pool_dead).

    If a worker deadlocks (e.g. a native hang in the cg engine on one game), the missing
    results would otherwise freeze the whole run forever. So we pull results with a per-game
    timeout: if none arrives within `cfg.game_timeout` seconds, we abandon the remaining
    games for this iteration, terminate the (now-suspect) pool, and signal the caller to
    recreate it. Checkpointing is per-iteration, so the cost of a stall is at most the
    games we drop from one iteration."""
    import multiprocessing as _mp
    HEARTBEAT = 90.0  # log a "still running" line if no game completes within this long
    tasks = [(g, weights_path, version, cfg.sims, cfg.determinizations, cfg.temp_moves,
              cfg.max_moves, rng.randrange(2 ** 31)) for g in range(n_games)]
    data = []
    wins = draws = done = 0
    pool_dead = False
    t0 = last_complete = time.time()
    results = worker_pool.imap_unordered(_worker_play, tasks)
    while done < n_games:
        try:
            # Poll on a short heartbeat so the slow tail of an iteration is visibly alive
            # (the both-sides-search games cluster late and can take minutes each).
            samples, res, learner = results.next(timeout=HEARTBEAT)
        except _mp.TimeoutError:
            idle = time.time() - last_complete
            if idle > cfg.game_timeout:  # a real stall (worker deadlock) -> recycle the pool
                log.warning("  self-play STALL: no game finished in %.0fs; abandoning %d "
                            "remaining games and recycling the pool", idle, n_games - done)
                worker_pool.terminate()
                pool_dead = True
                break
            log.info("  selfplay %d/%d | still running (%.0fs since last game, %.0fs total)",
                     done, n_games, idle, time.time() - t0)
            continue
        last_complete = time.time()
        data.extend(samples)
        wins += (res == learner)
        draws += (res == 2)
        done += 1
        if done % max(1, n_games // 10) == 0:
            log.info("  selfplay %d/%d | %d samples | learner WR %.0f%% | %.0fs",
                     done, n_games, len(data), 100 * wins / done, time.time() - t0)
    log.info("collected %d samples from %d games (learner WR %.1f%%, draws %d, %.0fs)",
             len(data), done, 100 * wins / max(1, done), draws, time.time() - t0)
    return data, wins / max(1, done), pool_dead


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
        last_pce, last_vmse = tot_p / seen, tot_v / seen
        log.info("  epoch %d/%d | policy_ce %.4f | value_mse %.4f",
                 ep + 1, cfg.epochs, last_pce, last_vmse)
    return last_pce, last_vmse


# ---------------------------------------------------------------------------
# Evaluation signals (greedy = the deployment policy)
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


@torch.inference_mode()
def eval_vs_random(model, deck, pool, n_games, log, seed=777):
    """Greedy win-rate vs a uniform-random opponent piloting random pool decks.

    Unlike the vs-bots gate (which saturates at the ~2-3% noise floor because the scripted
    experts beat everything), this has real resolution -- a competent net wins 60-70% here,
    so it tracks whether the net is actually getting stronger between iterations."""
    model.eval()
    device = next(model.parameters()).device
    rng = random.Random(seed)
    wins = n = 0
    for g in range(n_games):
        learner = g % 2
        opp_deck = rng.choice(pool)
        decks = [deck, opp_deck] if learner == 0 else [opp_deck, deck]
        obs, sd = battle_start(decks[0], decks[1])
        if sd.errorPlayer >= 0:
            raise ValueError(f"deck error type {sd.errorType}")
        while obs["current"]["result"] < 0:
            me = obs["current"]["yourIndex"]
            if me == learner:
                obs = battle_select(_policy_action(obs, deck, model, 0.0, rng, device))
            else:
                obs = battle_select(_random_legal(obs, rng))
        res = obs["current"]["result"]
        battle_finish()
        wins += (res == learner)
        n += 1
    wr = wins / n
    log.info("eval vs random | greedy WR %.1f%% (%d games)", 100 * wr, n)
    return wr


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
    ap.add_argument("--scripted-frac", type=float, default=0.5,
                    help="fraction of self-play games whose opponent is a scripted bot "
                         "(exploitative training vs the real opponents; 0 = pure self-play)")
    ap.add_argument("--eval-games", type=int, default=40,
                    help="games per bot when gating vs bots (0 to skip)")
    ap.add_argument("--eval-temp", type=float, default=0.0,
                    help="eval sampling temperature (0 = greedy = the deployment policy)")
    ap.add_argument("--eval-random-games", type=int, default=60,
                    help="games for the high-resolution vs-random progress signal (0 to skip)")
    ap.add_argument("--snapshot-every", type=int, default=5,
                    help="also save out/iterNN.pth every N iterations (for arena ranking)")
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
    ap.add_argument("--game-timeout", type=float, default=900.0,
                    help="seconds with no completed game before a stalled pool is recycled")
    ap.add_argument("--metrics-file", default=None,
                    help="append metrics to this existing CSV and continue its iteration "
                         "numbering (for resumed runs); default writes a new metrics/<tag>.csv")
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

    # Scripted bots as exploitative training opponents (single-process path; workers load
    # their own instances). Also used for the vs-bots gate.
    sp_opponents = None
    if args.scripted_frac > 0:
        from opponents import load_opponents
        sp_opponents = load_opponents()
        log.info("scripted-frac %.2f: %d bots in self-play opponent mix (%s)",
                 args.scripted_frac, len(sp_opponents), ", ".join(o.label for o in sp_opponents))

    model = MyModel(**arch).to(DEVICE)
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=DEVICE)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        log.info("resumed weights from %s", args.resume)
    else:
        log.info("training from scratch (random init)")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # Self-play worker pool, via a factory so the stall-watchdog can recreate it.
    import multiprocessing as mp
    n_workers = args.workers if args.workers > 0 else max(1, (os.cpu_count() or 2) - 1)
    ctx = mp.get_context("spawn")  # required on Windows; safe with CUDA in the parent

    def _make_pool():
        return ctx.Pool(n_workers, initializer=_worker_init,
                        initargs=(arch, deck, args.scripted_frac))

    worker_pool = _make_pool() if n_workers > 1 else None
    log.info("self-play: %s", f"{n_workers} parallel workers" if worker_pool else "single process")
    selfplay_weights = os.path.join(args.out, "_selfplay.pth")

    # Git-trackable, easily-parsed metrics: one CSV row per iteration, for graphing.
    # --metrics-file appends to an existing CSV and continues its iteration numbering and
    # wall-clock (resumed runs); otherwise a fresh metrics/<tag>.csv is started.
    import csv as _csv
    os.makedirs("metrics", exist_ok=True)
    cols = ["iter", "samples", "selfplay_wr", "policy_ce", "value_mse", "vs_random_wr",
            "vs_bots_wr", "vs_dragapult", "vs_abomasnow", "vs_iono", "selfplay_secs", "wall_secs"]
    start_iter, wall_offset = 0, 0.0
    if args.metrics_file and os.path.exists(args.metrics_file):
        metrics_path = args.metrics_file
        with open(metrics_path, newline="") as f:
            prev = [r for r in _csv.DictReader(f) if r.get("iter")]
        if prev:
            start_iter = max(int(r["iter"]) for r in prev)
            wall_offset = max((float(r["wall_secs"]) for r in prev if r.get("wall_secs")),
                              default=0.0)
        log.info("appending metrics to %s; continuing from iter %d", metrics_path, start_iter)
    else:
        metrics_path = args.metrics_file or os.path.join("metrics", f"{run_tag}.csv")
        with open(metrics_path, "w", newline="") as f:
            _csv.writer(f).writerow(cols)
        log.info("metrics -> %s (one row per iteration)", metrics_path)
    total_iters = start_iter + args.iters

    def _r(x):  # round numbers, blank for missing
        return round(x, 4) if isinstance(x, (int, float)) else ""

    run_t0 = time.time()
    replay = []
    best_wr = -1.0
    try:
        for it in range(args.iters):
            it_label = start_iter + it + 1
            log.info("=== iteration %d/%d ===", it_label, total_iters)
            model.eval()
            sp_t0 = time.time()
            if worker_pool is not None:
                # Publish current weights for workers to pick up (version = iteration).
                torch.save({"state_dict": model.state_dict(), "arch": arch}, selfplay_weights)
                samples, sp_wr, pool_dead = selfplay_parallel(
                    worker_pool, args.games, selfplay_weights, it_label, args, rng, log)
                if pool_dead:  # watchdog terminated the stalled pool -> rebuild a fresh one
                    worker_pool.join()
                    worker_pool = _make_pool()
                    log.info("recreated worker pool after stall")
            else:
                samples, sp_wr = selfplay_games(args.games, deck, model, archetypes, pool, args,
                                                rng, log, opponents=sp_opponents)
            selfplay_secs = time.time() - sp_t0
            if not samples:
                log.warning("no samples this iteration (all games abandoned); skipping update")
                continue
            replay.append(samples)
            replay = replay[-args.replay_iters:]
            buffer = [s for chunk in replay for s in chunk]
            log.info("training on %d samples (%d recent iters)", len(buffer), len(replay))
            pce, vmse = train(model, optimizer, buffer, args, log)

            torch.save({"state_dict": model.state_dict(), "arch": arch},
                       os.path.join(args.out, "last.pth"))
            if args.snapshot_every > 0 and it_label % args.snapshot_every == 0:
                snap = os.path.join(args.out, f"iter{it_label:02d}.pth")
                torch.save({"state_dict": model.state_dict(), "arch": arch}, snap)
                log.info("snapshot -> %s (rank checkpoints with arena.py)", snap)

            rnd_wr = eval_vs_random(model, deck, pool, args.eval_random_games, log) \
                if args.eval_random_games > 0 else None
            bot_mean, bot_per = None, {}
            if args.eval_games > 0:
                bot_mean, bot_per = evaluate_vs_bots(model, deck, args.eval_games,
                                                     args.eval_temp, log)
                if bot_mean > best_wr:
                    best_wr = bot_mean
                    torch.save({"state_dict": model.state_dict(), "arch": arch},
                               os.path.join(args.out, "best.pth"))
                    log.info("** new best vs-bots WR %.1f%% -> %s/best.pth", 100 * best_wr, args.out)

            with open(metrics_path, "a", newline="") as f:
                _csv.writer(f).writerow([
                    it_label, len(samples), _r(sp_wr), _r(pce), _r(vmse), _r(rnd_wr),
                    _r(bot_mean), _r(bot_per.get("dragapult")), _r(bot_per.get("abomasnow")),
                    _r(bot_per.get("iono")), _r(selfplay_secs), _r(wall_offset + time.time() - run_t0)])
    finally:
        if worker_pool is not None:
            worker_pool.terminate()  # force-kill workers so none are orphaned
            worker_pool.join()
    log.info("done. best vs-bots WR %.1f%%. metrics: %s", 100 * best_wr, metrics_path)


if __name__ == "__main__":
    main()
