# Pokémon TCG Agent — Search-Based Expert Iteration

A neural agent that pilots a fixed **Dragapult ex** deck in the Pokémon TCG competition
environment (`cg/`). The agent is trained **from scratch** (random weights, no behavioral
cloning, no PPO) with **Expert Iteration** — AlphaZero-style self-play where the
policy-improvement operator is *search*, not model-free RL.

> Environment setup (Python, PyTorch, the `cg/` engine) lives in **[SETUP.md](SETUP.md)**.

---

## 1. Why this strategy

Pokémon TCG is a poor fit for vanilla self-play PPO:

- **Reward is terminal and sparse** — one ±1 per game, but a game is hundreds of
  micro-decisions (every play / attach / evolve / target choice is its own observation),
  so credit assignment is brutal.
- **The action space is combinatorial and decision-dependent** — the legal candidate set
  changes shape every step; policy-gradient variance is high.
- **Imperfect information + stochasticity** — hidden hand/deck/prizes, coin flips, shuffles.
- **Self-play non-stationarity** — a moving target that can cycle without getting stronger.

Empirically, from-scratch PPO plateaued at ~15% vs the scripted bots. The fix is not a
better PPO — it is to **stop treating this as model-free RL**. Two assets the environment
hands us make a stronger approach possible:

1. **A forward-search API** (`cg.api.search_begin` / `search_step`) — forks the *real*
   engine from the agent's point of view, letting us look ahead. This is a perfect
   simulator for Monte-Carlo Tree Search.
2. **A known metagame** — the opponent decks in `training_opponents/` are the archetypes
   we will face, so hidden opponent state can be *inferred*, not guessed blindly.

So the agent is **AlphaZero adapted to an imperfect-information card game**: a learned
policy/value net, amplified at every move by determinized MCTS, with the search results
distilled back into the net so the next iteration's search starts stronger. This bootstraps
upward from random weights.

---

## 2. The network (`features.py`)

A single Transformer, `MyModel`, with two heads:

- **Encoder** — turns an `Observation` into 24 "tokens" (both players' active + bench
  Pokémon, per-player aggregates, our hand, our decklist identity, the stadium, turn
  info) via a sparse `EmbeddingBag` over card IDs. → **value head** (`tanh`, the expected
  game outcome for the player to move).
- **Decoder** — attends over the *candidate actions* of the current decision (each action
  encoded by its option type + the cards it touches) and emits **one raw logit per
  candidate** → the **policy head** (a prior over legal moves).

`features.py` is the single source of truth for featurization, imported by both training
and inference so the representation is identical in both. Key functions:
`get_encoder_input`, `enumerate_actions`, `get_decoder_input`, `evaluate`.

Architecture is configurable (`--d-model/--heads/--d-ff/--layers`) and **stored in each
checkpoint**, so inference rebuilds the right shape automatically.

---

## 3. The three pillars

### 3a. Belief-based determinization (`belief.py`)

Search needs a *full* state, but the agent only sees its own side. `belief.py` fills the
gaps:

- Maintains a **posterior over the opponent's archetype** — every opponent card we've seen
  (board, discard, owned stadium) is matched against the `training_opponents/` decklists;
  only archetypes consistent with the observations survive.
- **Samples one consistent full state** (`make_determinization`): a sampled archetype for
  the opponent, its hidden deck/hand/prizes/active dealt from the cards not yet seen, plus
  our own hidden deck/prizes (our 60 minus what we can see). Counts are made exact so the
  engine accepts them; off-distribution opponents degrade gracefully (best-overlap
  archetype, padded with basic energy).

Running search over **K independent determinizations and averaging** is *determinized UCT*
(a.k.a. PIMC) — it marginalizes over the hidden state instead of betting on one guess. This
replaces the reference approach of filling the opponent deck with a single filler card,
which discards search's entire advantage.

### 3b. Determinized PUCT search (`mcts.py`)

`run_mcts` runs AlphaZero-style PUCT through the real engine:

- **Selection** maximizes `Q + c_puct · P · √ΣN / (1 + N)`. Because turns are *not* strictly
  alternating (a player makes many sub-decisions before passing), perspective is keyed on
  each node's `yourIndex`; values are kept in the **root player's** frame and the sign is
  flipped for opponent nodes (negamax).
- **Expansion / evaluation** — a new leaf is evaluated by the net (value = leaf estimate,
  policy logits = priors), or scored ±1/0 if terminal.
- **K determinizations** each run an independent tree; the **root visit counts are summed**
  across them (the root's legal moves are public, hence index-aligned).
- Returns the chosen action **and** the normalized visit distribution `π` — the training
  target. Root **Dirichlet noise** and a **temperature** schedule drive exploration during
  self-play; inference uses argmax over visits.

### 3c. Expert Iteration / distillation (`exit_training.py`)

The loop that turns search into a stronger net:

```
net_0 = random init
repeat:
  self-play: every move chosen by run_mcts (both sides = current net + MCTS,
             opponents pilot pool decks)         ->  records (features, π, z)
  train:     policy head  ->  cross-entropy to π   (the searched, improved policy)
             value head   ->  MSE to z             (the game outcome)
  gate:      win-rate vs the held-out scripted bots; promote best.pth on improvement
```

Why this beats the PPO loop: the training target `π` is a **low-variance, already-improved**
policy (search did the hard exploration), so learning is stable and sample-efficient — no
advantage estimation, no reward shaping, no KL hacks. The net distills the search; next
iteration's search starts from the better net and climbs further.

Design choices:
- The learner **always pilots the submission deck** (`deck.csv`) and only **learner-side
  decisions are collected**, so all training data is about playing *our* deck well.
- The opponent pilots a random deck from the pool (the net is deck-agnostic — it takes the
  decklist as input), giving real archetype diversity instead of pure mirrors.
- **Both sides use the same belief sampling they would at inference**, so there is no
  train/test mismatch (we never feed search the true hidden state).
- The `example_ai` scripted bots are a **held-out test set** used only to gate checkpoints,
  never trained against.

---

## 4. From scratch vs. warmstart, and feasibility

Expert Iteration does **not** require behavioral cloning — AlphaGo Zero is the proof that
search-based self-play reaches top strength from random weights, and BC would only imprint
the scripted expert's biases. The cost of going BC-free is **compute**, not optimality.

`benchmark.py` measures the binding constraint. On a 16-core machine at 24 sims × 4
determinizations:

| | rate |
|---|---|
| policy-only (no search) | ~3.2 games/s |
| determinized MCTS self-play | ~38 s/game single-process |

Search is ~120× slower per game — that is the whole from-scratch tax. Self-play is
**embarrassingly parallel** (the `cg` engine is single-battle per process), so
`exit_training.py` runs a persistent worker pool (`--workers`) that divides wall-clock by
~#cores, turning a multi-day single-process run into an overnight one. Workers run the net
on CPU; the main process trains on GPU and publishes weights each iteration for workers to
reload.

To *warmstart* instead (faster convergence, at the price of inheriting expert bias), point
`--resume` at any compatible checkpoint.

---

## 5. File map

| File | Role |
|---|---|
| `features.py` | Network (`MyModel`) + state/action featurization — shared by train & inference |
| `belief.py` | Opponent-archetype posterior + determinization sampling for search |
| `mcts.py` | Determinized PUCT search (`run_mcts`) guided by the net |
| `exit_training.py` | From-scratch Expert-Iteration training loop (self-play → distill → gate) |
| `benchmark.py` | Self-play throughput benchmark (sizing a run) |
| `opponents.py` | Loads the `example_ai` scripted bots as the held-out eval suite |
| `nn_agent.py` | Inference: `load_model`, greedy `policy_step`, search-based `mcts_agent` |
| `main.py` | Competition entry point — loads `out/best.pth`, returns an action per observation |
| `deck.csv` | The 60-card Dragapult ex submission deck |
| `training_opponents/` | Archetype decklists — the belief prior **and** the self-play opponent pool |
| `example_ai/` | Hand-coded reference bots — held-out evaluation only |
| `cg/` | The native game engine (`battle_*`, `search_*`) — see SETUP.md |

---

## 6. How to run

**Train from scratch** (auto-uses `cpu_count − 1` self-play workers):

```bash
python exit_training.py --iters 50 --games 200 --sims 24 --determinizations 3 \
    --epochs 2 --eval-games 30 --eval-temp 0.4 --workers 12 --out out_exit
```

Checkpoints (`last.pth`, `best.pth`) and per-run logs land in `--out` / `logs/`. Resume or
warmstart with `--resume out_exit/last.pth`.

**Size a run first** (pick `--sims/--determinizations/--workers` against your hardware):

```bash
python benchmark.py --games 6 --sims 24 --determinizations 4
```

**Submit.** `main.py` loads `out/best.pth` and exposes `agent(obs) -> list[int]`. Copy a
trained checkpoint into place and pick the inference mode:

```bash
cp out_exit/best.pth out/best.pth
```

- `USE_SEARCH = False` (default) — greedy over the policy head. Fast, no search.
- `USE_SEARCH = True` — full determinized MCTS at inference (`nn_agent.mcts_agent`),
  stronger but slower; tune `SEARCH_SIMS` / `SEARCH_DETERMINIZATIONS` in `nn_agent.py` to
  the per-move time limit. Falls back to the greedy policy on any error, so it can't crash.

---

## 7. Key knobs

| Flag | Meaning |
|---|---|
| `--iters`, `--games` | iterations × self-play games per iteration (total game budget) |
| `--sims`, `--determinizations` | search depth × belief width per move (strength vs. speed) |
| `--temp-moves` | learner moves played with exploration temperature before switching to argmax |
| `--replay-iters` | how many recent iterations of samples to keep in the training buffer |
| `--eval-games`, `--eval-temp` | held-out gate size and sampling temperature (temp > 0 gives the deterministic engine resolution) |
| `--max-moves` | abandon a pathological self-play game past this many decisions (worker-pool tail guard) |
| `--d-model/--heads/--d-ff/--layers`, `--workers`, `--lr` | architecture, parallelism, learning rate |

The training value to watch is the **held-out win-rate trajectory** in the log
(`eval vs bots`) — from random init it starts near zero and the question is whether it
climbs across iterations.
