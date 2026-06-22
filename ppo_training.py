"""PPO self-play trainer for the Dragapult deck.

Model-free: the network maps observation -> action directly. We play full games with
the engine (NO forward search, so no hidden opponent state is ever fabricated), record
the learner's decisions, and update with PPO. Writes out/best.pth (gated on evaluation
win-rate) and out/last.pth (every iteration).

Run:  pytorch_env/bin/python ppo_training.py
"""

import argparse
import csv
import logging
import multiprocessing as mp
import os
import random
import time
from datetime import datetime

import torch
import torch.nn.functional as F

from cg.api import to_observation_class
from cg.game import battle_start, battle_select, battle_finish
from features import (
    MAX_ACTIONS,
    MyModel,
    enumerate_actions,
    evaluate,
    get_decoder_input,
    get_encoder_input,
    num_words_encoder,
)
from opponents import load_opponents

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARCH = dict(d_model=128, num_heads=2, d_feedforward=256,
            num_layers_encoder=2, num_layers_decoder=2)

# --- PPO / RL hyperparameters ---
GAMMA = 0.997
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
LR = 3e-4
PPO_EPOCHS = 4
MINIBATCH = 256

GAMES_PER_ITER = 40       # games collected per iteration
EVAL_GAMES = 50           # evaluation games per opponent (more = less noisy best-gate)
SAVE_EVERY = 1
# Prize-differential reward shaping. Terminal reward alone (+/-1) is too sparse across
# ~100-decision games for good credit assignment. We add a dense potential-based reward
# from the prize-card lead: potential Phi = (opp prizes remaining) - (my prizes remaining)
# from the learner's view (taking a prize -> my remaining drops -> Phi rises). The per-step
# shaping reward is PRIZE_SHAPE_COEF * (Phi_next - Phi_now); it telescopes so it doesn't
# distort who-won, just densifies the signal. 0 disables shaping.
PRIZE_SHAPE_COEF = 0.1
# Collection mix (per game): SCRIPTED_FRACTION vs scripted bots, LEAGUE_FRACTION vs
# frozen past selves, remainder = current-policy mirror self-play. The scripted
# curriculum gives a strong stationary signal early; the league prevents self-play from
# collapsing into a single style / cyclically forgetting. Shift toward league+selfplay
# as the net surpasses the bots.
SCRIPTED_FRACTION = 0.34
LEAGUE_FRACTION = 0.33
LEAGUE_SNAPSHOT_EVERY = 10   # add a frozen snapshot of the learner every N iterations
LEAGUE_MAX = 8               # keep at most this many recent snapshots on disk
LEAGUE_DIR = "out/league"


def load_deck(path="deck.csv"):
    return [int(x) for x in open(path).read().split("\n")[:60]]


class Transition:
    __slots__ = ("sv_enc", "sv_dec", "n_actions", "action_idx",
                 "logprob", "value", "player", "my_prize", "opp_prize",
                 "reward", "ret", "adv")

    def __init__(self, sv_enc, sv_dec, n_actions, action_idx, logprob, value, player,
                 my_prize, opp_prize):
        self.sv_enc = sv_enc
        self.sv_dec = sv_dec
        self.n_actions = n_actions
        self.action_idx = action_idx
        self.logprob = logprob
        self.value = value
        self.player = player
        self.my_prize = my_prize    # learner's own prize cards remaining at this state
        self.opp_prize = opp_prize  # opponent's prize cards remaining at this state
        self.reward = 0.0
        self.ret = 0.0
        self.adv = 0.0


@torch.inference_mode()
def act(obs_dict, deck, model, greedy=False):
    """Pick an action; return (selected_list, Transition or None for trivial decisions)."""
    obs = to_observation_class(obs_dict)
    actions = enumerate_actions(obs)
    sv_enc = get_encoder_input(obs, deck)
    sv_dec = get_decoder_input(obs, actions)
    value, logits = evaluate(model, sv_enc, sv_dec, DEVICE)
    logits = logits[: len(actions)]
    if greedy:
        idx = int(torch.argmax(logits).item())
        return actions[idx], None
    dist = torch.distributions.Categorical(logits=logits)
    idx = int(dist.sample().item())
    me = obs.current.yourIndex
    tr = Transition(sv_enc, sv_dec, len(actions), idx,
                    float(dist.log_prob(torch.tensor(idx, device=DEVICE)).item()),
                    float(value.item()), me,
                    len(obs.current.players[me].prize),
                    len(obs.current.players[1 - me].prize))
    return actions[idx], tr


def run_game(controllers, decks, learner_model, collect=True, greedy=False):
    """Play one game with arbitrary per-player controllers.

    controllers[p] is one of:
      ("learner", None)    -> learner_model on decks[p]; transitions collected
      ("frozen", model)    -> a frozen league snapshot model; not collected
      ("script", agent_fn) -> a scripted bot; not collected
    Returns (result, transitions).
    """
    obs, sd = battle_start(decks[0], decks[1])
    if sd.errorPlayer >= 0:
        raise ValueError(f"deck error type {sd.errorType}")
    transitions = []
    while obs["current"]["result"] < 0:
        p = obs["current"]["yourIndex"]
        kind, payload = controllers[p]
        if kind == "learner":
            selected, tr = act(obs, decks[p], learner_model, greedy=greedy)
            if collect and tr is not None:
                transitions.append(tr)
        elif kind == "frozen":
            # Older self plays stochastically (variety), no gradient/collection.
            selected, _ = act(obs, decks[p], payload, greedy=False)
        else:  # script
            try:
                selected = payload(obs)
            except Exception:
                # A scripted-bot crash shouldn't kill a long run; play a legal move.
                o = to_observation_class(obs)
                selected = random.sample(range(len(o.select.option)),
                                         max(o.select.minCount, 1))
        obs = battle_select(selected)
    result = obs["current"]["result"]
    battle_finish()
    return result, transitions


def build_specs(n_games, n_scripted, n_league, n_opponents, league_ids):
    """Describe an iteration's games as picklable specs.

    spec is one of:
      ("selfplay",)                       current learner mirror-match
      ("script", opponent_index, learner) vs a scripted bot
      ("league", snapshot_id, learner)    vs a frozen past self
    """
    specs = []
    for g in range(n_games):
        if g < n_scripted and n_opponents > 0:
            specs.append(("script", g % n_opponents, g % 2))
        elif g < n_scripted + n_league and league_ids:
            specs.append(("league", random.choice(league_ids), g % 2))
        else:
            specs.append(("selfplay",))
    return specs


class FrozenProvider:
    """Lazily loads frozen league snapshots from disk, with a small LRU of live models.

    Snapshots are 51 MB each, so we keep models in memory rather than reloading the
    same one repeatedly, but cap how many to bound RAM. Reads are OS page-cached.
    """

    def __init__(self, arch, league_dir, device, cache_size=3):
        self.arch = arch
        self.league_dir = league_dir
        self.device = device
        self.cache_size = cache_size
        self.cache = {}        # snapshot_id -> model
        self.order = []        # LRU order of ids

    def get(self, snap_id):
        if snap_id in self.cache:
            self.order.remove(snap_id)
            self.order.append(snap_id)
            return self.cache[snap_id]
        model = MyModel(**self.arch).to(self.device)
        state = torch.load(os.path.join(self.league_dir, f"snap_{snap_id}.pth"),
                           map_location=self.device)
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        model.eval()
        self.cache[snap_id] = model
        self.order.append(snap_id)
        if len(self.order) > self.cache_size:
            evict = self.order.pop(0)
            del self.cache[evict]
        return model


def realize(spec, deck, opponents, frozen_provider):
    """Turn a spec into (controllers, decks) for run_game."""
    if spec[0] == "selfplay":
        return [("learner", None), ("learner", None)], [deck, deck]
    if spec[0] == "league":
        _, snap_id, learner = spec
        frozen = frozen_provider.get(snap_id)
        controllers = [None, None]
        controllers[learner] = ("learner", None)
        controllers[1 - learner] = ("frozen", frozen)
        return controllers, [deck, deck]
    # script
    _, opp_idx, learner = spec
    opp = opponents[opp_idx]
    decks = [deck, opp.deck] if learner == 0 else [opp.deck, deck]
    controllers = [None, None]
    controllers[learner] = ("learner", None)
    controllers[1 - learner] = ("script", opp.agent)
    return controllers, decks


def play_specs(specs, deck, opponents, learner_model, frozen_provider,
               shape_coef=0.0, greedy=False):
    """Play a list of game specs sequentially; return (transitions, [w0,w1,draw])."""
    data, results = [], [0, 0, 0]
    for spec in specs:
        controllers, decks = realize(spec, deck, opponents, frozen_provider)
        result, trs = run_game(controllers, decks, learner_model, collect=True, greedy=greedy)
        finalize(trs, result, shape_coef)
        data.extend(trs)
        results[result] += 1
    return data, results


def finalize(transitions, result, shape_coef=0.0):
    """Assign rewards (terminal +/-1 plus optional prize-diff shaping) and compute GAE.

    Shaping: potential Phi_t = opp_prize_t - my_prize_t (learner's view); the reward at
    step t gains shape_coef * (Phi_{t+1} - Phi_t). Potential-based, so it telescopes and
    doesn't change the win/loss objective -- it just gives a dense within-game signal.
    """
    for p in (0, 1):
        traj = [t for t in transitions if t.player == p]
        if not traj:
            continue
        if result == 2:
            outcome = 0.0
        else:
            outcome = 1.0 if result == p else -1.0
        traj[-1].reward = outcome
        if shape_coef:
            for i in range(len(traj) - 1):
                phi_now = traj[i].opp_prize - traj[i].my_prize
                phi_next = traj[i + 1].opp_prize - traj[i + 1].my_prize
                traj[i].reward += shape_coef * (phi_next - phi_now)
        gae = 0.0
        next_value = 0.0
        for t in reversed(traj):
            delta = t.reward + GAMMA * next_value - t.value
            gae = delta + GAMMA * GAE_LAMBDA * gae
            t.adv = gae
            t.ret = gae + t.value
            next_value = t.value
    return transitions


# --- batched PPO update ----------------------------------------------------
def _cat_sparse(svs, words_per):
    """Concatenate encoder/decoder SparseVectors into flat EmbeddingBag tensors,
    padding decoder samples to MAX_ACTIONS words. Returns (index, value, offset)."""
    index, value, offset = [], [], []
    for sv in svs:
        base = len(index)
        index.extend(sv.index)
        value.extend(sv.value)
        for o in sv.offset:
            offset.append(o + base)
        # pad to fixed word count (decoder only)
        for _ in range(words_per - len(sv.offset)):
            offset.append(len(index))
    return index, value, offset


def ppo_update(model, optimizer, transitions):
    advs = torch.tensor([t.adv for t in transitions], dtype=torch.float32, device=DEVICE)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    for i, t in enumerate(transitions):
        t.adv = float(advs[i].item())

    n = len(transitions)
    order = list(range(n))
    for _ in range(PPO_EPOCHS):
        random.shuffle(order)
        for s in range(0, n, MINIBATCH):
            batch = [transitions[i] for i in order[s:s + MINIBATCH]]
            ei, ev, eo = _cat_sparse([t.sv_enc for t in batch], num_words_encoder)
            di, dv, do = _cat_sparse([t.sv_dec for t in batch], MAX_ACTIONS)
            value, logits = model(
                torch.tensor(ei, dtype=torch.int32, device=DEVICE),
                torch.tensor(ev, dtype=torch.float32, device=DEVICE),
                torch.tensor(eo, dtype=torch.int32, device=DEVICE),
                torch.tensor(di, dtype=torch.int32, device=DEVICE),
                torch.tensor(dv, dtype=torch.float32, device=DEVICE),
                torch.tensor(do, dtype=torch.int32, device=DEVICE))
            # mask invalid (padded) candidates
            mask = torch.full((len(batch), MAX_ACTIONS), float("-inf"), device=DEVICE)
            for r, t in enumerate(batch):
                mask[r, : t.n_actions] = 0.0
            logits = logits + mask
            logp_all = F.log_softmax(logits, dim=1)
            idx = torch.tensor([t.action_idx for t in batch], device=DEVICE)
            new_logp = logp_all.gather(1, idx.unsqueeze(1)).squeeze(1)
            old_logp = torch.tensor([t.logprob for t in batch], device=DEVICE)
            adv = torch.tensor([t.adv for t in batch], device=DEVICE)
            ret = torch.tensor([t.ret for t in batch], device=DEVICE)

            ratio = torch.exp(new_logp - old_logp)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(value.squeeze(1), ret)
            probs = logp_all.exp()
            entropy = -(probs * torch.where(torch.isinf(logp_all),
                        torch.zeros_like(logp_all), logp_all)).sum(1).mean()
            loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    return float(policy_loss.item()), float(value_loss.item()), float(entropy.item())


def eval_vs_opponent(deck, model, opp, n_games):
    """Greedy nn (deck) vs a scripted opponent (opp.deck). Win rate excl. draws."""
    wins = losses = 0
    for g in range(n_games):
        learner = g % 2  # alternate first/second player for fairness
        decks = [deck, opp.deck] if learner == 0 else [opp.deck, deck]
        controllers = [None, None]
        controllers[learner] = ("learner", None)
        controllers[1 - learner] = ("script", opp.agent)
        res, _ = run_game(controllers, decks, model, collect=False, greedy=True)
        if res == learner:
            wins += 1
        elif res != 2:
            losses += 1
    return wins / max(1, wins + losses)


def evaluate_suite(deck, model, opponents, n_games):
    """Evaluate vs every scripted bot. Returns (mean_winrate, {label: winrate})."""
    per = {opp.label: eval_vs_opponent(deck, model, opp, n_games) for opp in opponents}
    mean = sum(per.values()) / max(1, len(per))
    return mean, per


# --- parallel rollout collection ------------------------------------------
# Weights (learner + league snapshots) are 51 MB each, so they are exchanged via DISK
# (OS page-cached) rather than pickled through the queues every iteration.
def worker_loop(cfg, task_q, result_q):
    """Child process: play assigned game specs on CPU and return transitions.

    Runs in a 'spawn'-started process, so this module is re-imported fresh (its own
    cg engine instance + opponents). We force CPU inference here and reserve the GPU
    in the parent for the PPO update / evaluation.
    """
    global DEVICE
    DEVICE = torch.device("cpu")
    torch.set_num_threads(1)  # avoid thread oversubscription across many workers
    model = MyModel(**cfg["arch"]).to(DEVICE)
    deck = cfg["deck"]
    opponents = load_opponents()
    frozen = FrozenProvider(cfg["arch"], cfg["league_dir"], DEVICE)
    while True:
        task = task_q.get()
        if task is None:
            break
        learner_path, specs, seed = task
        state = torch.load(learner_path, map_location=DEVICE)
        model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
        model.eval()
        random.seed(seed)
        torch.manual_seed(seed)
        data, results = play_specs(specs, deck, opponents, model, frozen,
                                   shape_coef=cfg["shape_coef"], greedy=False)
        result_q.put((results, data))


class RolloutPool:
    """Persistent pool of CPU rollout workers."""

    def __init__(self, num_workers, arch, deck, league_dir, shape_coef):
        self.ctx = mp.get_context("spawn")
        self.task_q = self.ctx.Queue()
        self.result_q = self.ctx.Queue()
        self.num_workers = num_workers
        cfg = {"arch": arch, "deck": deck, "league_dir": league_dir,
               "shape_coef": shape_coef}
        self.procs = []
        for _ in range(num_workers):
            p = self.ctx.Process(target=worker_loop,
                                 args=(cfg, self.task_q, self.result_q), daemon=True)
            p.start()
            self.procs.append(p)

    def collect(self, learner_path, specs, base_seed):
        """Dispatch specs round-robin across workers; gather transitions + stats."""
        chunks = [specs[i::self.num_workers] for i in range(self.num_workers)]
        dispatched = 0
        for wid, chunk in enumerate(chunks):
            if chunk:
                self.task_q.put((learner_path, chunk, base_seed + wid))
                dispatched += 1
        data, results = [], [0, 0, 0]
        for _ in range(dispatched):
            res, trs = self.result_q.get()
            for i in range(3):
                results[i] += res[i]
            data.extend(trs)
        return data, results

    def close(self):
        for _ in self.procs:
            self.task_q.put(None)
        for p in self.procs:
            p.join()


def cpu_state_dict(model):
    return {k: v.detach().cpu() for k, v in model.state_dict().items()}


LOG_DIR = "logs"


def setup_logging(run_tag):
    """Log to console and to logs/train_<tag>.log. Returns the configured logger."""
    os.makedirs(LOG_DIR, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(os.path.join(LOG_DIR, f"train_{run_tag}.log"))
    fh.setFormatter(fmt)
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2),
                    help="parallel rollout workers (1 = serial, in-process)")
    ap.add_argument("--games", type=int, default=GAMES_PER_ITER)
    ap.add_argument("--eval-games", type=int, default=EVAL_GAMES)
    ap.add_argument("--scripted-frac", type=float, default=SCRIPTED_FRACTION)
    ap.add_argument("--league-frac", type=float, default=LEAGUE_FRACTION)
    ap.add_argument("--snapshot-every", type=int, default=LEAGUE_SNAPSHOT_EVERY)
    ap.add_argument("--eval-every", type=int, default=1,
                    help="run the (costly) bot-suite evaluation + best.pth gate every N iters")
    ap.add_argument("--shape-coef", type=float, default=PRIZE_SHAPE_COEF,
                    help="prize-differential reward shaping coefficient (0 disables)")
    args = ap.parse_args()
    games_per_iter, eval_games = args.games, args.eval_games
    scripted_fraction, league_fraction = args.scripted_frac, args.league_frac
    shape_coef = args.shape_coef

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = setup_logging(run_tag)
    log.info("run %s | device=%s | %s", run_tag, DEVICE, vars(args))
    log.info("hparams | gamma=%s lambda=%s clip=%s lr=%s ppo_epochs=%s minibatch=%s "
             "league_max=%s shape_coef=%s arch=%s", GAMMA, GAE_LAMBDA, CLIP_EPS, LR,
             PPO_EPOCHS, MINIBATCH, LEAGUE_MAX, shape_coef, ARCH)

    os.makedirs("out", exist_ok=True)
    league_dir = os.path.abspath(LEAGUE_DIR)
    # Start the league empty (league_ids/next_snap_id reset each run, even on --resume),
    # so old snapshots can't accumulate as orphaned disk weight across runs.
    if os.path.isdir(league_dir):
        for f in os.listdir(league_dir):
            if f.startswith("snap_") and f.endswith(".pth"):
                os.remove(os.path.join(league_dir, f))
    os.makedirs(league_dir, exist_ok=True)
    learner_path = os.path.abspath("out/_learner.pth")  # weights handed to workers each iter
    deck = load_deck()
    opponents = load_opponents()
    log.info("loaded scripted opponents: %s", [o.label for o in opponents])
    model = MyModel(**ARCH).to(DEVICE)
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location=DEVICE)
        model.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
        log.info("resumed from %s", args.resume)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # League state: frozen snapshots persisted on disk as snap_<id>.pth, FIFO-capped.
    league_ids: list[int] = []
    next_snap_id = 0
    frozen = FrozenProvider(ARCH, league_dir, DEVICE)  # for the serial path

    n_scripted = int(round(games_per_iter * scripted_fraction))
    n_league = int(round(games_per_iter * league_fraction))
    pool = None
    if args.workers >= 2:
        pool = RolloutPool(args.workers, ARCH, deck, league_dir, shape_coef)
        log.info("parallel rollout: %d CPU workers", args.workers)
    else:
        log.info("serial rollout (in-process)")

    # Per-iteration metrics CSV (one row per iter) for easy plotting later.
    csv_path = os.path.join(LOG_DIR, f"metrics_{run_tag}.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    opp_labels = [o.label for o in opponents]
    csv_writer.writerow(
        ["iter", "time_s", "samples", "win0", "win1", "draw", "league",
         "ploss", "vloss", "entropy", "wr_mean"]
        + [f"wr_{lbl}" for lbl in opp_labels] + ["best_wr"])
    csv_file.flush()
    log.info("logging metrics to %s", csv_path)

    best_wr = -1.0
    try:
        for it in range(args.iters):
            t0 = time.time()
            specs = build_specs(games_per_iter, n_scripted, n_league,
                                len(opponents), league_ids)

            # --- data collection: scripted + league + mirror self-play ---
            model.eval()
            if pool is not None:
                torch.save({"state_dict": cpu_state_dict(model)}, learner_path)
                data, results = pool.collect(learner_path, specs, base_seed=it * 1000)
            else:
                data, results = play_specs(specs, deck, opponents, model, frozen,
                                           shape_coef=shape_coef, greedy=False)

            # --- PPO update ---
            model.train()
            pl, vl, ent = ppo_update(model, optimizer, data)

            ckpt = {"state_dict": model.state_dict(), "arch": ARCH}
            if it % SAVE_EVERY == 0:
                torch.save(ckpt, "out/last.pth")

            # --- add a frozen snapshot to the league ---
            if (it + 1) % args.snapshot_every == 0:
                torch.save({"state_dict": cpu_state_dict(model)},
                           os.path.join(league_dir, f"snap_{next_snap_id}.pth"))
                league_ids.append(next_snap_id)
                next_snap_id += 1
                if len(league_ids) > LEAGUE_MAX:
                    old = league_ids.pop(0)
                    try:
                        os.remove(os.path.join(league_dir, f"snap_{old}.pth"))
                    except OSError:
                        pass

            # --- evaluation gate: vs the scripted bot suite (every eval-every iters) ---
            do_eval = (it % args.eval_every == 0) or (it == args.iters - 1)
            tag = ""
            if do_eval:
                model.eval()
                mean_wr, per = evaluate_suite(deck, model, opponents, eval_games)
                if mean_wr >= best_wr:
                    best_wr = mean_wr
                    torch.save(ckpt, "out/best.pth")
                    tag = "  <-- new best"
            dt = time.time() - t0

            if do_eval:
                per_str = " ".join(f"{k} {v:.0%}" for k, v in per.items())
                log.info("iter %4d | %5.1fs | samples %5d | p0/p1/draw %s | league %d | "
                         "ploss %+.3f vloss %.3f ent %.3f | WR mean %.1f%% [%s]%s",
                         it, dt, len(data), results, len(league_ids),
                         pl, vl, ent, 100 * mean_wr, per_str, tag)
                csv_writer.writerow(
                    [it, f"{dt:.1f}", len(data), results[0], results[1], results[2],
                     len(league_ids), f"{pl:.4f}", f"{vl:.4f}", f"{ent:.4f}", f"{mean_wr:.4f}"]
                    + [f"{per[lbl]:.4f}" for lbl in opp_labels] + [f"{best_wr:.4f}"])
            else:
                log.info("iter %4d | %5.1fs | samples %5d | p0/p1/draw %s | league %d | "
                         "ploss %+.3f vloss %.3f ent %.3f",
                         it, dt, len(data), results, len(league_ids), pl, vl, ent)
                csv_writer.writerow(
                    [it, f"{dt:.1f}", len(data), results[0], results[1], results[2],
                     len(league_ids), f"{pl:.4f}", f"{vl:.4f}", f"{ent:.4f}", ""]
                    + ["" for _ in opp_labels] + [f"{best_wr:.4f}"])
            csv_file.flush()
    finally:
        csv_file.close()
        if pool is not None:
            pool.close()


if __name__ == "__main__":
    main()
