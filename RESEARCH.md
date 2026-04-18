# 2048 AI Research Reference

## Current State of the Art (as of April 2026)

### Two Competing Lanes

**1. Best pure learning (no search at test time):**
N-tuple TD learning remains dominant. Deep RL has not come close.

**2. Best overall play:**
Search + precomputed endgame tablebases has overtaken pure learning.

### Results Summary

| System                        | Method                          | Avg Score | 32768% | 65536% |
|-------------------------------|---------------------------------|-----------|--------|--------|
| Deep RL best (CNN+DQN)        | No search                       | ~215,803  | --     | --     |
| Stochastic MuZero             | Learned model + MCTS            | ~510,000  | --     | --     |
| TDL2048+ (Guei, 6-ply)       | N-tuple TD + expectimax         | 625,377   | 72%    | ~0.02% |
| macroxue/2048-ai (depth 8)   | Expectimax + endgame lookup     | 711,769   | 80.5%  | 3.5%   |
| 2048EndgameTablebase          | Expectimax + endgame tablebases | ~772,353  | 86.1%  | 8.4%   |

### Key Papers and Projects

- **Guei (2022)** — "On Reinforcement Learning for the Game of 2048". Best
  learning-only result (625,377 avg, 72% 32768-rate). Optimistic TD learning,
  TC learning, multi-stage training.
- **2048EndgameTablebase (2024–2026)** — github.com/game-difficulty/2048EndgameTablebase.
  Overall SOTA (v10.0.1, April 2026). Compressed tablebases + advanced pruning.
- **macroxue/2048-ai** — github.com/macroxue/2048-ai. Expectimax + endgame
  lookup. 80.5% 32768-rate, 3.5% 65536-rate.
- **moporgic/TDL2048** — github.com/moporgic/TDL2048. Reference SOTA learning
  framework (C++, 102M moves/sec). Implements optimistic TD, TC learning,
  multi-stage, expectimax.
- **Gumbel MuZero for 2048 (TAAI 2023)** — 3 simulations counterintuitively
  outperformed 50.
- **"Systematic Selection of N-Tuples" (2025)** — Uses neural networks to
  select n-tuple patterns.

## N-Tuple TD Implementation Details

### Tuple Patterns

Standard high-performing config: 4–8 base 6-tuple patterns covering
rectangular/L-shaped regions. Through 8-fold symmetry expansion, 4 base
patterns become 32 features.

Well-known 4-pattern set (board positions 0–15, row-major):

| Pattern | Positions          |
|---------|--------------------|
| 1       | 0, 1, 2, 3, 4, 5  |
| 2       | 4, 5, 6, 7, 8, 9  |
| 3       | 0, 1, 2, 4, 5, 6  |
| 4       | 4, 5, 6, 8, 9, 10 |

Stronger config adds:

| Pattern | Positions          |
|---------|--------------------|
| 5       | 0, 1, 5, 6, 7, 10 |
| 6       | 0, 1, 2, 3, 4, 9  |
| 7       | 0, 1, 2, 5, 9, 10 |
| 8       | 0, 1, 5, 6, 9, 13 |

### Symmetry Handling

8 symmetries (dihedral group D4: 4 rotations x 2 reflections). Each base
pattern expanded into 8 variants, each with its own weight table. During
training, all 8 variants updated simultaneously with the same TD error.

V(s) = sum over all expanded tuples of their weight table lookups.

### Training Hyperparameters

- **Learning rate:** α = 0.0025 per tuple (Guei/Wu). Fixed rate, no decay
  needed for basic results.
- **Training games:** 100k decent results; 1M+ for SOTA.
- **Optimistic initialization:** Critical trick. Initialize all weights to a
  positive value (e.g. 380,000). Zero-init converges much more slowly.
- **Multi-stage training:** Separate value functions for different game phases.
  Stage boundaries by max tile on board.
- **TC (Temporal Coherence) learning:** Adaptive per-weight learning rate based
  on sign consistency of updates. Improves convergence.
- **Reward:** Sum of merged tile values per move (score increment).

### Afterstate Value Function

V(s') evaluates the afterstate — after the player slides tiles, before the
random tile spawns.

```
V(s_t) <- V(s_t) + α * [r_{t+1} + V(s_{t+1}) - V(s_t)]
```

Action selection: for each legal move, compute r + V(afterstate), pick max.

### Evaluation Methodology

- **Without search:** Greedy action selection using learned V.
- **With 1-ply expectimax:** Average over all possible tile spawns (each empty
  cell x {2,4} with 90%/10% probability).
- **Deeper search (2–3+ ply):** Expensive but improves results significantly.
  TDL2048+ goes from 412K (1-ply) to 625K (6-ply).
- **Game count:** 1,000 minimum for stable stats; 10,000 for publishable.

### Weight Table Sizes

Each 6-tuple: 16^6 = 16,777,216 entries (tile exponent values 0–15).
Dense arrays are standard.

- 4 base patterns x 8 symmetries = 32 tables x 16^6 x 4 bytes ≈ **2 GB**
- 8 base patterns x 8 symmetries = 64 tables x 16^6 x 4 bytes ≈ **4 GB**

### Implementation Tricks

1. **Board as u64:** 4 bits per cell. Rotations/reflections via table-lookup
   on 16-bit rows.
2. **Row-based move tables:** Precompute results for all 65,536 possible rows.
   Full board move = 4 row lookups.
3. **Tuple index computation:** index = Σ tile_exponent × 16^position_in_tuple.
4. **Terminal states:** V(terminal) = 0.

## Our Results

### TC-Hogwild Optimistic Init Sweep (beta=0.5, 8x6, 1M games, 14 threads)

| K | Final Avg | 8192% | 16384% | Notes |
|---|-----------|-------|--------|-------|
| 0 (baseline) | 232,604 | 89.1 | 53.3 | No optimistic init |
| **100** | **242,928** | **91.7** | **55.6** | **New best** |
| 500 | 1,432 | 0.0 | 0.0 | Diverged |
| 1000 | 972 | 0.0 | 0.0 | Diverged |
| 2000 | 151,516 | 86.6 | 1.7 | Recovered but plateaued low |
| 5000 | 141,453 | 80.7 | 0.4 | Similar to K=2000 |

**Key findings:**
- K=100 gives a modest improvement over the no-init baseline (+4.4% avg score).
- K=500–1000 diverges completely — TC's adaptive rates can't correct the
  large initial bias, and the coherence signal gets poisoned.
- K=2000–5000 recovers but never catches the baseline. The non-monotonic
  behaviour suggests the interaction between optimistic init magnitude and
  TC's per-weight adaptive rates is sensitive to the ratio between K and
  the natural value scale at different training stages.
- Optimistic init helps TC less dramatically than it helps plain TD. This
  makes sense: TC already adapts per-weight learning rates, so the
  exploration benefit of optimistic init is partially redundant.

**Next steps:**
1. Fine sweep around K=100 (K=50, 75, 100, 150, 200) to refine the optimum.
2. Longer training runs (5M, 10M games) with the best K to see if the
   advantage persists or compounds.
3. Investigate the `target-cpu=native` effect on training quality (see FUTURE.md).

## Phase 2 Target (TBD)

"Surpassing SOTA" could mean:
- **(a)** Beat best learning-only result (Guei, 625K avg)
- **(b)** Beat overall SOTA (tablebases, 772K avg)
- **(c)** Beat both with novel combination

Decision deferred to phase 2 entry.
