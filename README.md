# Pokémon TCG Agent — Search-Based Expert Iteration

A neural agent that pilots a fixed **Dragapult ex** deck in the Pokémon TCG competition
environment (`cg/`). It is trained with **Expert Iteration (ExIt)** — AlphaZero-style
self-play in which the policy-improvement operator is *Monte-Carlo Tree Search through the
real game engine*, adapted for an imperfect-information, stochastic card game.

> Environment setup (Python, PyTorch, the `cg/` engine) is in **[SETUP.md](SETUP.md)**.

---

## 1. Strategy in one paragraph

A single network outputs a **policy** (a prior over the legal moves of a decision) and a
**value** (expected game outcome). At every move during self-play we run **determinized
PUCT search** that looks ahead through the real engine, using the network as its prior and
leaf evaluator; the search is much stronger than the raw network. We then **distill the
search back into the network** — the policy head learns the search's visit distribution,
the value head learns the game outcome. The improved network makes the next iteration's
search stronger, and the loop bootstraps upward. Two adaptations make this work for an
imperfect-information game with strong scripted opponents:

1. **Belief-based determinization** turns the agent's partial observation into consistent
   full states for search, sampled from a posterior over the opponent's archetype.
2. **Mixed opponents** — self-play games (for general strength) are blended with games
   against the scripted reference bots (for *exploitative* strength against the actual
   opponents we must beat).

---

## 2. The network (`features.py`)

One Transformer, `MyModel`, with two heads:

- **Encoder → value head.** The observation becomes 24 "tokens" (both players' active +
  bench Pokémon, per-player aggregates, our hand, our decklist identity, the stadium, turn
  info) via a sparse `EmbeddingBag` over card IDs. The value head outputs the expected
  outcome (`tanh`, from the to-move player's perspective).
- **Decoder → policy head.** The decoder attends over the *candidate actions* of the
  current decision (each encoded by its option type and the cards it touches) and emits one
  raw logit per candidate — a prior over legal moves.

`features.py` is the single source of truth for featurization, imported by both training and
inference. Architecture is CLI-configurable and **stored in every checkpoint**, so inference
rebuilds the correct shape automatically.

---

## 3. The three core components

### 3a. Belief-based determinization (`belief.py`)

Search needs a full state, but the agent sees only its own side. `belief.py`:

- Keeps a **posterior over the opponent's archetype** — observed opponent cards (board,
  discard, owned stadium) are matched against the decklists in `training_opponents/`; only
  consistent archetypes survive.
- **Samples one consistent full state** (`make_determinization`): a sampled opponent
  archetype with its hidden deck/hand/prizes/active dealt from the unseen cards, plus our own
  hidden deck/prizes. Counts are made exact so the engine accepts them; off-distribution
  opponents degrade gracefully.

Running search over **K independent determinizations and averaging** the root visit counts
is determinized UCT (PIMC) — it marginalizes over the hidden state instead of betting on one
guess.

### 3b. Determinized PUCT search (`mcts.py`)

`run_mcts` runs AlphaZero-style PUCT through the engine's `search_begin`/`search_step`:

- **Selection** maximizes `Q + c_puct · P · √ΣN / (1 + N)`. Perspective is keyed on each
  node's `yourIndex` (turns are not strictly alternating — a player makes many sub-decisions
  before passing), and values are kept in the root player's frame (negamax).
- **Leaves** are evaluated by the network (value + policy prior) or scored ±1/0 if terminal.
- **K determinizations** each run an independent tree; their root visit counts are summed.
- Returns the chosen action **and** the normalized visit distribution `π` — the training
  target. Root Dirichlet noise + a temperature schedule drive self-play exploration;
  inference uses argmax over visits.

### 3c. Expert Iteration with mixed opponents (`exit_training.py`)

```
net_0 = random init (or --resume a checkpoint)
repeat for each iteration:
  self-play: every learner move chosen by run_mcts; the opponent is EITHER the current
             net piloting a random pool deck (self-play) OR a scripted bot piloting its
             own deck (probability --scripted-frac)        ->  records (features, π, z)
  distill:   policy head -> cross-entropy to π ;  value head -> MSE to outcome z
  evaluate + snapshot (see §4)
```

Why distillation beats model-free RL here: the target `π` is a low-variance, already-improved
policy (search did the exploration), so learning is stable and sample-efficient — no
advantage estimation, no reward shaping. The learner always pilots the submission deck and
only learner-side decisions are collected, so all training data is about playing *our* deck.
Mixing in the scripted bots as opponents makes the agent learn to beat the specific strong
strategies it will actually face.

Self-play is parallelized across a persistent worker pool (`--workers`); the cg engine is
single-battle per process, so workers each run the net on CPU while the main process trains
on GPU and publishes weights each iteration.

---

## 4. Measuring progress

Win-rate against a single strong opponent is a poor signal (the scripted experts beat almost
everything, so it saturates near the noise floor). We use three complementary measures:

- **`arena.py` — relative Elo (the primary yardstick).** A round-robin among any set of
  checkpoints plus anchors (`random` and the scripted bots), fit to Bradley-Terry Elo with
  `random` anchored to 1000. This measures *relative* strength with real resolution — the
  AlphaZero way to tell whether iteration N+1 is stronger than N.
- **vs-random win-rate (in-loop, high resolution).** A competent net wins ~60–70% here, so
  it tracks improvement smoothly between iterations.
- **vs-bots greedy win-rate (in-loop, the objective).** Greedy = the deployment policy.

Every `--snapshot-every` iterations the run saves `out/iterNN.pth` so `arena.py` can rank
checkpoints after the fact, decoupling "pick the best model" from the noisy in-loop gate.

### Metrics & graphing

Each run writes one CSV row per iteration to **`metrics/<run_tag>.csv`** (git-tracked,
stdlib-parseable). Columns:

```
iter, samples, selfplay_wr, policy_ce, value_mse, vs_random_wr,
vs_bots_wr, vs_dragapult, vs_abomasnow, vs_iono, selfplay_secs, wall_secs
```

`plot_metrics.py` renders the strength curves and distillation losses to a PNG next to the
CSV (`python plot_metrics.py` uses the newest run; pass a path for a specific one). The CSV
is plain — graph it with anything (pandas, Excel, gnuplot).

---

## 5. File map

| File | Role |
|---|---|
| `features.py` | Network (`MyModel`) + state/action featurization — shared by train & inference |
| `belief.py` | Opponent-archetype posterior + determinization sampling for search |
| `mcts.py` | Determinized PUCT search (`run_mcts`) guided by the network |
| `exit_training.py` | ExIt training loop: mixed-opponent self-play → distill → evaluate → snapshot |
| `arena.py` | Round-robin Elo across checkpoints + random/scripted anchors (the yardstick) |
| `plot_metrics.py` | Render strength/loss graphs from a `metrics/*.csv` |
| `opponents.py` | Loads the `example_ai` scripted bots (training opponents + eval) |
| `nn_agent.py` | Inference: `load_model`, greedy `policy_step`, search-based `mcts_agent` |
| `main.py` | Competition entry point — loads `out/best.pth`, returns an action per observation |
| `deck.csv` | The 60-card Dragapult ex submission deck |
| `training_opponents/` | Archetype decklists — the belief prior and self-play opponent pool |
| `example_ai/` | Hand-coded reference bots — exploitative training opponents + Elo anchors |
| `metrics/` | Per-run CSV metrics (git-tracked) |
| `cg/` | The native game engine (`battle_*`, `search_*`) — see SETUP.md |

---

## 6. How to run

**Train from scratch:**

```bash
python exit_training.py --iters 50 --games 128 --sims 128 --determinizations 5 \
    --scripted-frac 0.5 --workers 12 --snapshot-every 5 \
    --d-model 160 --heads 4 --d-ff 320 --layers 4 --out out_fresh
```

Resume / warmstart: add `--resume <checkpoint.pth>` (its arch must match the `--d-model/...`
flags). Size a run for your hardware first with `python benchmark.py`.

**Rank checkpoints (the real strength measure):**

```bash
python arena.py --ckpts out_fresh/best.pth out_fresh/iter25.pth out_fresh/iter50.pth --games 30
```

**Graph the run:**

```bash
python plot_metrics.py            # newest metrics/*.csv -> PNG
```

**Submit.** `main.py` loads `out/best.pth` and exposes `agent(obs) -> list[int]`. Copy the
checkpoint you want into place and choose the inference mode:

```bash
cp out_fresh/best.pth out/best.pth
```

- `USE_SEARCH = False` (default) — greedy over the policy head. Fast, no search.
- `USE_SEARCH = True` — full determinized MCTS at inference (`nn_agent.mcts_agent`),
  stronger but slower; tune `SEARCH_SIMS` / `SEARCH_DETERMINIZATIONS` in `nn_agent.py` to the
  per-move time limit. Falls back to the greedy policy on any error.

---

## 7. Key knobs

| Flag | Meaning |
|---|---|
| `--iters`, `--games` | iterations × self-play games per iteration |
| `--sims`, `--determinizations` | search depth × belief width per move (strength vs. speed) |
| `--scripted-frac` | fraction of self-play games whose opponent is a scripted bot (exploitation) |
| `--temp-moves` | learner moves played with exploration temperature before argmax |
| `--replay-iters` | recent iterations of samples kept in the training buffer |
| `--eval-games`, `--eval-random-games` | gate sizes for vs-bots and vs-random signals |
| `--snapshot-every` | save `out/iterNN.pth` every N iters for arena ranking |
| `--d-model/--heads/--d-ff/--layers`, `--workers`, `--lr` | architecture, parallelism, learning rate |
