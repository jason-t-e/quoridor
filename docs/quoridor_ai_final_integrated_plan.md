# Quoridor AI — Final Integrated Technical Plan
### Strategy-Informed Neural Network (SINN) + Full System Design
### Zero-Budget, Cloud-First Architecture

---

## Table of Contents

**Part I — Plan Comparison & Integration**
1. [Document Comparison & Integration Decisions](#1-document-comparison--integration-decisions)
2. [Core Philosophy: Strategy as Advice, Not Law](#2-core-philosophy-strategy-as-advice-not-law)

**Part II — ML Architecture**
3. [Game Engine + Zobrist Hashing](#3-game-engine--zobrist-hashing)
4. [Strategy Guide (Agreement-Based)](#4-strategy-guide-agreement-based)
5. [Neural Network Architecture (SINN)](#5-neural-network-architecture-sinn)
6. [Guided Search](#6-guided-search)
7. [Transposition Tables](#7-transposition-tables)
8. [Rating System (0–5)](#8-rating-system-05)
9. [Loss Function](#9-loss-function)
10. [Post-Game Analysis](#10-post-game-analysis)
11. [Emergent Strategy Detection](#11-emergent-strategy-detection)
12. [Self-Play Training Pipeline](#12-self-play-training-pipeline)

**Part III — System Design**
13. [System Architecture Overview](#13-system-architecture-overview)
14. [Complete File Structure](#14-complete-file-structure)
15. [Frontend Design](#15-frontend-design)
16. [API & Backend Logic](#16-api--backend-logic)
17. [Database & Storage](#17-database--storage)
18. [Auth & Permissions](#18-auth--permissions)
19. [Hosting & Cloud (Free Tier Only)](#19-hosting--cloud-free-tier-only)
20. [CI/CD & Version Control](#20-cicd--version-control)
21. [Security](#21-security)
22. [Rate Limiting](#22-rate-limiting)
23. [Caching & CDN](#23-caching--cdn)
24. [Error Tracking & Logs](#24-error-tracking--logs)
25. [Monitoring & Alerts](#25-monitoring--alerts)
26. [Scaling](#26-scaling)

**Part IV — Implementation**
27. [Training Configuration](#27-training-configuration)
28. [Data Flow Diagram](#28-data-flow-diagram)
29. [Key Algorithms (Pseudocode)](#29-key-algorithms-pseudocode)
30. [Implementation Phases](#30-implementation-phases)
31. [Design Decisions Summary](#31-design-decisions-summary)

---

## 1. Document Comparison & Integration Decisions

Two planning documents existed. This section documents what each contributed and what the integration keeps.

### Side-by-Side Comparison

| Dimension | Document 1 ("Strategy-Informed + Emergent") | Document 2 ("Final Consolidated") | This Plan |
|---|---|---|---|
| **Strategy vector** | 16-dim (adds counter confidence, deviation signal) | 12-dim | **16-dim** — richer context |
| **Emergent strategy detection** | Full module: fingerprinting, DBSCAN clustering, counter building | Not included | **Included** — unique differentiator |
| **Deviation logging** | `deviation_logger.py` — tracks profitable disagreements with guide | Not included | **Included** — essential for rule weight learning |
| **Agreement weight learning** | Explicit: per-rule weights updated each game, slow decay for ignored rules | Implicit via loss anneal | **Explicit** — per-rule weights + loss anneal |
| **Philosophy section** | Embedded in Section 5 | Standalone Section 2 | **Standalone** — Doc 2's clarity |
| **Code comments** | Dense, technical | Clean, explanatory | **Doc 2 style** throughout |
| **Counter planner** | Full closed loop including emergent strategies | Known strategies only | **Full loop** from Doc 1 |
| **Path attention** | `attention_scale` fixed init | `attention_scale` is a learned parameter | **Learned** from Doc 2 |
| **DAG backpropagation** | Mentioned | Explicit warning: "simulation path only" | **Explicit** from Doc 2 |
| **BFS performance note** | Not emphasized | "Must be fast. Use deque + early termination" | **Included** from Doc 2 |
| **Training estimate table** | Detailed, with hours-based milestones | More detailed | **Doc 2 version** |
| **System design** | Neither document | Neither document | **Added in this plan** |

### Integration Principles

1. **Doc 2 as structural base** — its section organization and code comments are cleaner.
2. **Doc 1's unique features are additive** — emergent detection, deviation logging, and 16-dim vector are all purely additive; they don't conflict with Doc 2's design.
3. **Contradictions resolved in favor of the more flexible option** — e.g., learned `attention_scale` beats fixed init because the model can always learn to set it to approximately the fixed value anyway.
4. **No feature removed without rationale** — if a feature from either document is dropped, the reason is stated.

---

## 2. Core Philosophy: Strategy as Advice, Not Law

This is the single most important architectural decision. It propagates into the loss function, the UCB formula, the rating system, and the strategy layer architecture.

### The Problem with Strategy as Law

Encoding strategy rules as hard constraints or strong penalties tells the model that human analysts have already found the best strategy. Every disagreement is treated as an error to fix. Over thousands of training games this pushes the model into the region of strategy space that humans already understand — exactly the wrong place to look for superhuman play.

AlphaZero discovered opening systems no human grandmaster had played. That was only possible because the model was free to deviate from human knowledge without penalty.

### The Solution

> **Agreeing with known good strategy is rewarded. Disagreeing is not penalized. Consistently deviating and winning causes the rule's weight to decay.**

This maps to three specific changes:

```
Old thinking:  violate strategy → penalty          (strategy as law)
New thinking:  agree with strategy → bonus reward  (strategy as advice)
               disagree with strategy → zero bonus, never a penalty
               disagree consistently AND win → rule weight decays toward zero
```

### What This Changes

| Component | Old Behavior | New Behavior |
|---|---|---|
| `strategy_advisor.py` | Computes violation scores | Computes agreement scores ∈ [0,1] — zero means absence of bonus, not a punishment |
| `loss_functions.py` | `L_strategy = mean(violation_scores)` penalized | `L_strategy = -mean(agreement_bonuses)` rewards |
| UCB formula | Agreement bonus sometimes negative | `agreement_bonus ≥ 0` always — never pushes away from a move |
| `strategy_layer.py` | Constraint injector | Input feature — model decides how much to weight it |
| `agreement_weight_updater.py` | Fixed weights | Rules model ignores (while winning) decay; rules that predict wins strengthen |

---

## 3. Game Engine + Zobrist Hashing

### `board.py` — Board State

```python
@dataclass
class BoardState:
    pawn_positions:   dict  # {0: (row, col), 1: (row, col)}
    h_walls:          set   # horizontal wall anchors: (row, col)
    v_walls:          set   # vertical wall anchors: (row, col)
    walls_remaining:  dict  # {0: int, 1: int}; 10 each at start
    current_player:   int   # 0 or 1
    move_history:     list
    turn:             int
    current_hash:     int   # Zobrist hash — updated O(1) inside apply_move()
```

Player 0 starts at row 0, goal row 8. Player 1 starts at row 8, goal row 0. The board is always presented from the current player's perspective (flipped so "my goal" is always row 8).

**Board as Tensor — 10 channels, 9×9 each:**

```
Channel 0:  Own pawn binary map
Channel 1:  Opponent pawn binary map
Channel 2:  Horizontal wall map
Channel 3:  Vertical wall map
Channel 4:  Own BFS distance map          (cell value / max_distance)
Channel 5:  Opponent BFS distance map
Channel 6:  Own walls remaining           (scalar broadcast: walls / 10)
Channel 7:  Opponent walls remaining      (scalar broadcast: walls / 10)
Channel 8:  Game phase                    (0=opening, 0.5=mid, 1=end, broadcast)
Channel 9:  Current player                (0 or 1, broadcast)
```

### `pathfinder.py`

BFS is the most performance-critical non-ML component. It is called inside MCTS simulations, inside the strategy advisor, inside the rating system, and inside the endgame solver. Use `collections.deque` with early termination on goal-row hit.

```python
def bfs_distance_map(board, player) -> np.ndarray:
    """9×9 array: min steps from pawn to each cell, respecting walls."""

def shortest_path_to_goal(board, player) -> int:
    """Min moves to reach any cell in the goal row."""

def path_differential(board) -> int:
    """opponent_path_length - own_path_length. Positive = we are ahead in the race."""
```

### `rules.py` — Wall Legality

Three checks in order (fail-fast):
1. Wall fits within 8×8 placement grid
2. No overlap with existing walls (4 overlap cases per orientation)
3. Anti-blockade: BFS for both players confirms paths still exist after placement

Check 3 is expensive. Cache results for recently evaluated positions using the Zobrist hash as the cache key.

### `zobrist_hash.py`

Computed from scratch once at game start; updated in O(1) via XOR inside every `apply_move()`. No other code touches the hash.

```python
class ZobristHasher:
    def __init__(self):
        rng = np.random.default_rng(seed=42)          # Fixed seed — reproducible
        self.pawn_table   = rng.integers(1, 2**64-1, size=(2, 9, 9),  dtype=np.uint64)
        self.hwall_table  = rng.integers(1, 2**64-1, size=(8, 8),     dtype=np.uint64)
        self.vwall_table  = rng.integers(1, 2**64-1, size=(8, 8),     dtype=np.uint64)
        self.walls_table  = rng.integers(1, 2**64-1, size=(2, 11),    dtype=np.uint64)
        self.player_table = rng.integers(1, 2**64-1, size=(2,),       dtype=np.uint64)

    def full_hash(self, board) -> int:
        """Called once per game at initialization."""
        h = np.uint64(0)
        for p in [0, 1]:
            r, c = board.pawn_positions[p]
            h ^= self.pawn_table[p, r, c]
            h ^= self.walls_table[p, board.walls_remaining[p]]
        for (r, c) in board.h_walls:  h ^= self.hwall_table[r, c]
        for (r, c) in board.v_walls:  h ^= self.vwall_table[r, c]
        h ^= self.player_table[board.current_player]
        return int(h)

    def incremental_update(self, current_hash, board_before, move) -> int:
        """O(1). Called inside apply_move() automatically."""
        h = np.uint64(current_hash)
        player = board_before.current_player
        if isinstance(move, PawnMove):
            r_old, c_old = board_before.pawn_positions[player]
            h ^= self.pawn_table[player, r_old, c_old]
            h ^= self.pawn_table[player, move.to_row, move.to_col]
        elif isinstance(move, WallMove):
            w_old = board_before.walls_remaining[player]
            h ^= self.walls_table[player, w_old]
            h ^= self.walls_table[player, w_old - 1]
            table = self.hwall_table if move.orientation == 'h' else self.vwall_table
            h ^= table[move.row, move.col]
        h ^= self.player_table[board_before.current_player]
        h ^= self.player_table[1 - board_before.current_player]
        return int(h)
```

---

## 4. Strategy Guide (Agreement-Based)

### `strategy_guide.py` — Rules as Agreement Functions

Every rule returns a value in `[0.0, 1.0]`. There is no floor below zero. The model is never told it did something wrong by the strategy system — only told when it did something the strategy system recognizes as good.

```python
@dataclass
class StrategyRule:
    name:      str
    phase:     GamePhase
    condition: Callable[[BoardState], bool]
    agreement: Callable[[BoardState, Move], float]  # ALWAYS in [0, 1]
    weight:    float = 1.0                          # Learned — can decay to zero

STRATEGY_RULES = [
    StrategyRule(
        name="wall_efficiency",
        phase=GamePhase.ALL,
        condition=lambda b: True,
        agreement=lambda b, a: (
            0.0 if not is_wall(a) else
            min(1.0, path_extension_from_wall(b, a) / 2.0)
        ),
    ),
    StrategyRule(
        name="path_race_advance",
        phase=GamePhase.ALL,
        condition=lambda b: path_differential(b) < 0,
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and shortens_own_path(b, a) else
            0.5 if is_wall(a) and path_extension_from_wall(b, a) >= 2 else
            0.0
            # Zero = no bonus. NOT a penalty. Model may have found something better.
        ),
    ),
    StrategyRule(
        name="standard_opening_center",
        phase=GamePhase.OPENING,
        condition=lambda b: b.turn < 6,
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and is_central_column(a) else
            0.3 if is_pawn_move(a) else 0.0
        ),
    ),
    StrategyRule(
        name="anti_rush_counter_wall",
        phase=GamePhase.OPENING,
        condition=lambda b: opponent_strategy(b) == "Rush",
        agreement=lambda b, a: (
            1.0 if is_wall(a) and intercepts_opponent_path(b, a) else
            0.0
        ),
    ),
    StrategyRule(
        name="protective_wall_when_ahead",
        phase=GamePhase.MID,
        condition=lambda b: path_differential(b) > 1,
        agreement=lambda b, a: (
            0.8 if is_wall(a) and placed_behind_own_pawn(b, a) else
            0.4 if is_pawn_move(a) and shortens_own_path(b, a) else
            0.0
        ),
    ),
    StrategyRule(
        name="jump_tempo",
        phase=GamePhase.ALL,
        condition=lambda b: adjacent_to_opponent(b),
        agreement=lambda b, a: (
            0.9 if is_jump_move(a) and shortens_own_path(b, a) else 0.0
        ),
    ),
    # additional rules: 3x3 trap, tunnel building, Shiller response, etc.
]
```

### `strategy_advisor.py`

```python
def compute_agreement_scores(board, action, rules=STRATEGY_RULES):
    """
    Returns list of (rule_name, score) for applicable rules.
    Score ∈ [0, 1]. Zero = no agreement = no bonus. Never negative.
    """
    return [
        (rule.name, rule.agreement(board, action))
        for rule in rules if rule.condition(board)
    ]

def total_agreement_bonus(scores, weights):
    """Always >= 0. Adds to reward signal, never to penalty."""
    return sum(weights.get(name, 0.1) * score for name, score in scores)
```

### `opening_recognizer.py`

Identifies opponent's strategy by turn 4. Also matches against registered emergent strategies.

```python
def recognize_strategy(move_history, board) -> tuple[StrategyLabel, float]:
    features = {
        "advance_rate":         compute_advance_rate(move_history),
        "wall_rate":            compute_wall_rate(move_history),
        "first_pawn_direction": get_first_pawn_move_direction(move_history),
        "wall_behind_pawn":     check_wall_behind_pawn(board, move_history),
        "lateral_first_move":   is_lateral_first_move(move_history),
        "tunnel_pattern":       detect_tunnel_walls(board, move_history),
        "central_preference":   compute_column_preference(move_history),
        "move_fingerprint":     compute_move_fingerprint(move_history),
    }
    if features["wall_rate"] == 0 and features["advance_rate"] > 0.8:
        return StrategyLabel.RUSH, 0.9
    if features["lateral_first_move"] and features["tunnel_pattern"]:
        return StrategyLabel.SHILLER, 0.85
    if features["wall_rate"] > 0 and features["wall_behind_pawn"] \
            and features["advance_rate"] > 0.5:
        return StrategyLabel.STANDARD, 0.8
    if features["wall_rate"] > 0 and not features["wall_behind_pawn"] \
            and features["advance_rate"] < 0.4:
        return StrategyLabel.SIDEWALL, 0.7

    emergent_match = EmergentStrategyDetector.match_fingerprint(
        features["move_fingerprint"]
    )
    if emergent_match:
        return StrategyLabel.NOVEL(emergent_match.id), emergent_match.confidence
    return StrategyLabel.UNKNOWN, 0.3
```

### `agreement_weight_updater.py`

```python
def update_rule_weights(game_record, outcome, agreement_weights):
    """
    Rules that predict winning moves strengthen.
    Rules the model consistently overrides (while winning) decay toward zero.
    """
    for turn_data in game_record.turn_data:
        for rule_name, agreement_score in turn_data.agreements:
            if agreement_score > 0.5:
                delta = 0.01 * (outcome - 0.5) * agreement_score
                agreement_weights[rule_name] = clip(
                    agreement_weights[rule_name] + delta, 0.0, 2.0
                )
    # Decay weights for rules the model has moved past
    for rule_name in agreement_weights:
        avg_agreement = game_record.avg_agreement_for_rule(rule_name)
        if avg_agreement < 0.05:
            agreement_weights[rule_name] *= 0.999  # Very slow — intentional
```

---

## 5. Neural Network Architecture (SINN)

### Strategy Vector — 16 Dimensions

```
[0:5]   Opponent strategy one-hot: [Standard, Rush, Shiller, Sidewall, Novel/Unknown]
[5]     Path differential (normalized): (opp_path - own_path) / 9
[6]     Wall efficiency: mean path extension per wall (last 3 walls)
[7:10]  Game phase one-hot: [Opening (turns 0–10), Mid (11–30), End (31+)]
[10:13] Counter action type one-hot: [Advance, Place_wall, Jump]
[13]    Counter confidence: win_rate of best known counter ∈ [0, 1]
[14]    Strategy agreement of last move taken ∈ [0, 1]
[15]    Novel strategy similarity to nearest known strategy ∈ [0, 1]
```

The counter confidence at [13] is important: it tells the network *how much to trust* the counter recommendation. A 80%-win-rate counter warrants stronger following than a 52%-win-rate one.

### `quoridor_net.py` — Master Architecture

```
INPUT:   Board Tensor [10 × 9 × 9]  +  Strategy Vector [16]

STEM:    Conv(10→128, kernel=3, pad=1) → BatchNorm → ReLU
         Output: [128 × 9 × 9]

BLOCK GROUP 1 (5 Guided Residual Blocks):
         ResBlock(128→128) × 5
         PathAttentionGate on output
         Output: [128 × 9 × 9]

STRATEGY INJECTION:
         StrategyEmbedding(16 → 16×9×9) → reshape [16 × 9 × 9]
         Concatenate: [144 × 9 × 9]
         PointwiseConv(144→128) → BatchNorm → ReLU
         Output: [128 × 9 × 9]
         ← Strategy vector is an input feature. The network decides how much to weight it.

BLOCK GROUP 2 (5 Guided Residual Blocks)
BLOCK GROUP 3 (5 Guided Residual Blocks)

TOTAL: 15 Residual Blocks, 128 channels (~3.2M parameters)

FLATTEN: Global Average Pool → [128]

THREE HEADS (parallel):

    POLICY HEAD:
         Linear(128→256) → ReLU → Linear(256→136)
         Softmax (invalid actions masked to -inf before softmax)
         Output: probability distribution [136]

    VALUE HEAD:
         Linear(128→64) → ReLU → Linear(64→1) → Tanh
         Output: scalar ∈ [-1, +1]

    RATING HEAD:
         Concat [128-dim, 8-dim move_context_vector]
         Linear(136→64) → ReLU → Linear(64→1) → Sigmoid × 5
         Output: scalar ∈ [0, 5]
```

### `path_attention.py` — The Guide

```python
class PathAttentionGate(nn.Module):
    """
    BFS distance maps gate spatial features. attention_scale is a LEARNED parameter
    — the network can reduce it toward zero if path attention isn't useful.
    """
    def __init__(self, attention_scale_init=0.5):
        super().__init__()
        self.attention_scale = nn.Parameter(torch.tensor(attention_scale_init))

    def forward(self, features, path_map_own, path_map_opp):
        path_diff_map = path_map_opp - path_map_own           # [9 × 9]
        attention = torch.softmax(
            path_diff_map.flatten(), dim=0
        ).reshape(9, 9).unsqueeze(0).unsqueeze(0)             # [1 × 1 × 9 × 9]
        gated = features * (1 + self.attention_scale * attention)
        return gated + features   # Residual: no cell is ever fully silenced
```

### `residual_block.py` — Pre-Activation ResBlock

```
Input x
→ BatchNorm → ReLU → Conv(128→128, 3×3, pad=1)
→ BatchNorm → ReLU → Conv(128→128, 3×3, pad=1)
→ + x
```

---

## 6. Guided Search

### `move_selector.py` — Phase-Aware Dispatch

```python
ENDGAME_WALLS_HARD = 4   # Total walls ≤ 4: always MTD(f)
ENDGAME_WALLS_SOFT = 6   # Total walls ≤ 6 AND max_path ≤ 5: MTD(f)
ENDGAME_PATH_SOFT  = 5

def select_move(board, network, strategy_guide, mcts_tt, endgame_tt, time_budget_ms):
    total_walls = sum(board.walls_remaining.values())
    max_path = max(shortest_path_to_goal(board, 0),
                   shortest_path_to_goal(board, 1))

    use_solver = (
        total_walls <= ENDGAME_WALLS_HARD or
        (total_walls <= ENDGAME_WALLS_SOFT and max_path <= ENDGAME_PATH_SOFT)
    )

    if use_solver:
        return endgame_solver.solve(board, endgame_tt)

    # Normal phase: DAG-MCTS + strategy context + 1-step lookahead
    counter_rec  = counter_planner.get_in_game_recommendation(board)
    strat_vector = strategy_guide.compute_strategy_vector(board, counter_rec)

    mcts_result = mcts.search(
        board, network, strategy_guide, mcts_tt,
        time_budget_ms=time_budget_ms * 0.85
    )

    top_k = mcts_result.top_k(k=5)
    evaluated = []
    for move in top_k:
        next_state = apply_move(board, move)
        opp_move   = opponent_predictor.predict(next_state, network)
        post_state = apply_move(next_state, opp_move)
        value      = network.value(board_to_tensor(post_state))
        evaluated.append((move, value))

    return max(evaluated, key=lambda x: x[1])[0]
```

### `mcts.py` — DAG-MCTS with Agreement-Guided UCB

```
UCB_strategy(s, a) = Q(s, a)
                   + c_puct * P(s, a) / (1 + N(s, a))
                   + λ_guide * path_guide_bonus(s, a)
                   + λ_agree * strategy_agreement_bonus(s, a)

where:
    path_guide_bonus         = delta_path_differential(s, a) / 9   ∈ [-1, +1]
    strategy_agreement_bonus = total_agreement_bonus(s, a)         ∈ [0, +∞)

Critical: agreement_bonus ≥ 0 always. Moves the strategy doesn't recognize
          get bonus = 0. They are never penalized.
```

**Node (DAG structure):**

```python
class MCTSNode:
    children: dict   # action_idx -> child_board_hash
    Q: float = 0.0   # Mean value: pooled across ALL paths to this node
    W: float = 0.0
    N: int   = 0     # Visit count pooled across all paths
    prior: float = 0.0
```

**Backpropagation rule:** update only nodes on the simulation path taken, not all DAG ancestors. Q/N sharing across paths happens implicitly through shared node references.

**Time-based stopping:**

```python
def search(self, root_board, network, strategy_guide, mcts_tt, time_budget_ms):
    start = time.monotonic_ns()
    while (time.monotonic_ns() - start) / 1e6 < time_budget_ms:
        self._run_simulation(root_board, network, strategy_guide)
    return self.root.visit_count_distribution()
```

### `endgame_solver.py` — MTD(f) Exact Solver

Activated when walls are nearly exhausted. Provides mathematically exact play — no probability, no approximation.

```python
class EndgameSolver:
    def solve(self, board, solver_tt) -> Move:
        f, best_move = 0, None
        for depth in range(1, self._estimate_max_depth(board) + 1):
            value, move = self._mtdf(board, f, depth, solver_tt)
            f, best_move = value, move
            if abs(value) >= WIN_VALUE * 0.9:
                break   # Forced win/loss found
        return best_move

    def _ordered_moves(self, board):
        """
        Priority 0: Winning pawn move (goal row) — almost always causes immediate cutoff
        Priority 1: Pawn advance (reduces own path)
        Priority 2: Wall blocking opponent's next move (extends opp path by 2+)
        Priority 3: Lateral pawn move
        Priority 4: Other wall placements
        Priority 5: Retreat
        """
```

**TT Lifecycle** — one `MCTSTranspositionTable` and one `SolverTranspositionTable` are created at game start, persist across all moves, and are cleared when the game ends. This cross-move persistence is where the main benefit comes from.

---

## 7. Transposition Tables

```python
class MCTSTranspositionTable:
    """Converts MCTS tree into a DAG. Same position, pooled Q/N statistics."""
    def get_or_create(self, board_hash, board_state) -> tuple[MCTSNode, bool]:
        if board_hash in self.table:
            return self.table[board_hash], True   # Inherit existing Q/N
        node = MCTSNode(board_state)
        self.table[board_hash] = node
        return node, False   # New node


class SolverTranspositionTable:
    """Stores alpha-beta bounds for MTD(f)."""
    def probe(self, board_hash, alpha, beta, depth):
        entry = self.table.get(board_hash)
        if entry is None or entry.depth < depth:
            return None, None
        if entry.bound == BoundType.EXACT:
            return entry.value, entry.best_action
        if entry.bound == BoundType.LOWER and entry.value >= beta:
            return entry.value, entry.best_action
        if entry.bound == BoundType.UPPER and entry.value <= alpha:
            return entry.value, entry.best_action
        return None, None

    def store(self, board_hash, value, best_action, alpha_orig, beta, depth):
        bound = (BoundType.UPPER if value <= alpha_orig else
                 BoundType.LOWER if value >= beta else
                 BoundType.EXACT)
        self.table[board_hash] = TTEntry(value, best_action, bound, depth)
```

**Why Quoridor benefits:** Wall A at turn 5 then Wall B at turn 7 reaches the same board as Wall B at turn 5 then Wall A at turn 7. Without a TT, MCTS treats these as two separate nodes. With a TT they collapse into one. Estimated gain: +40–50% effective simulations per time budget, largest during early training when positions repeat most.

---

## 8. Rating System (0–5)

Computed after the game when the full record is available.

```python
def compute_rating(state_before, action, state_after_opp_response) -> float:
    delta = path_differential(state_after_opp_response) - path_differential(state_before)

    if   delta >= 3:  base = 5.0
    elif delta >= 2:  base = 4.5
    elif delta >= 1:  base = 4.0
    elif delta >  0:  base = 3.5
    elif delta == 0:  base = 3.0
    elif delta > -1:  base = 2.0
    elif delta > -2:  base = 1.0
    else:             base = 0.0

    if is_true_blunder(state_after_opp_response):
        return 0.0

    # Agreement adds at most +0.5. Path delta is always the primary signal.
    # The model is NEVER rated lower for deviating from strategy advice.
    agreements  = compute_agreement_scores(state_before, action)
    strat_bonus = min(0.5, total_agreement_bonus(agreements, agreement_weights) * 0.1)

    return max(0.0, min(5.0, base + strat_bonus))


def is_true_blunder(board) -> bool:
    """Even using all remaining walls optimally, we cannot close the deficit."""
    deficit         = path_differential(board)   # Negative = we are behind
    max_recoverable = board.walls_remaining[board.current_player] * 2
    return deficit < -max_recoverable
```

---

## 9. Loss Function

```python
def total_loss(batch, model_output, agreement_data, config):
    # 1. POLICY — cross-entropy vs MCTS visit distribution
    L_policy = -torch.sum(
        batch.policy_targets * torch.log(model_output.policy_pred + 1e-8), dim=1
    ).mean()

    # 2. VALUE — MSE vs game outcome
    L_value = F.mse_loss(model_output.value_pred, batch.value_targets)

    # 3. RATING — MSE vs actual computed ratings
    L_rating     = F.mse_loss(model_output.rating_pred, batch.rating_targets)
    L_rating_max = F.relu(3.0 - model_output.rating_pred.mean())

    # 4. STRATEGY — AGREEMENT REWARD (not violation penalty)
    #    agreement_bonuses ∈ [0, +∞)
    #    L_strategy = -mean(bonuses)  → negative = reduces total loss = reward
    #    Gradient: ∂L/∂action = 0 when no agreement → no punishment for deviation
    agreement_bonuses = torch.tensor(
        [d.bonus_score for d in agreement_data], dtype=torch.float32
    )
    L_strategy = -agreement_bonuses.mean()

    # 5. DEFEAT — extra weight on losing-game samples
    defeat_mask = (batch.outcomes == -1).float()
    L_defeat = (defeat_mask * F.mse_loss(
        model_output.value_pred, batch.value_targets, reduction='none'
    )).mean()

    total = (config.w_policy    * L_policy   +
             config.w_value     * L_value    +
             config.w_rating    * L_rating   +
             config.w_rating_max* L_rating_max +
             config.w_strategy  * L_strategy +  # Negative = reward
             config.w_defeat    * L_defeat)

    return total, {"policy": L_policy.item(), "value": L_value.item(),
                   "rating": L_rating.item(), "strategy": L_strategy.item(),
                   "defeat": L_defeat.item()}
```

**Loss Weight Schedule:**

| Phase | Steps | `w_strategy` | `w_value` | Notes |
|---|---|---|---|---|
| Early | 0–50k | 0.40 | 1.0 | Strategy guides exploration; agreement bonus is meaningful |
| Mid | 50k–150k | 0.15 (linear) | 1.0 | Model learns outcome-based play; strategy fades |
| Late | 150k+ | 0.05 (held) | 1.0 | Near-zero regularization only; model plays how it wants |

---

## 10. Post-Game Analysis

### `backward_analyzer.py`

Scans the losing player's moves last-to-first. Finds the earliest unrecoverable mistake. Generates corrective experiences at elevated replay priority.

```python
def backward_analyze(game_record, losing_player, network, strategy_guide):
    mistakes = []
    moves = game_record.moves_for(losing_player)
    for i in range(len(moves) - 1, -1, -1):
        state        = game_record.states_before[losing_player][i]
        action_taken = moves[i]
        actual_rating  = game_record.actual_ratings[losing_player][i]
        pred_rating    = game_record.predicted_ratings[losing_player][i]
        best_move, _   = find_best_move_in_hindsight(state, network)
        best_rating    = simulate_rating(state, best_move, game_record)
        rating_gap     = pred_rating - actual_rating   # Confident but wrong
        missed_gain    = best_rating - actual_rating   # How much better was the best

        if rating_gap > 1.5 or missed_gain > 2.0:
            mistakes.append(MistakeRecord(
                turn=i, state=state, action_taken=action_taken,
                best_action=best_move, actual_rating=actual_rating,
                predicted_rating=pred_rating, missed_gain=missed_gain,
                severity=missed_gain + rating_gap,
            ))
        if was_game_first_lost_at(game_record, i, losing_player):
            mistakes[-1].is_decisive = True
            break   # Root cause found — stop scanning

    return sorted(mistakes, key=lambda m: -m.severity)
```

### `deviation_logger.py` — Profitable Deviations

```python
def log_profitable_deviations(game_record, winner):
    """
    After each WIN, find moves where:
      - Strategy agreement was near zero (model deviated)
      - Actual rating was high (the deviation worked)
    These reveal places where the model has discovered something
    better than the encoded strategy advice.
    Logged to deviation_log.json and fed to EmergentStrategyDetector.
    """
    for turn, (state, action) in enumerate(game_record.moves_for(winner)):
        agreements    = compute_agreement_scores(state, action)
        max_agreement = max((s for _, s in agreements), default=0.0)
        actual_rating = game_record.actual_ratings[winner][turn]

        if max_agreement < 0.2 and actual_rating >= 4.0:
            DeviationLog.record(state=state, action=action,
                                agreement=max_agreement, rating=actual_rating,
                                game_id=game_record.game_id, turn=turn)
```

---

## 11. Emergent Strategy Detection

```python
class EmergentStrategyDetector:
    """
    Pipeline:
    1. Extract move fingerprint (first FINGERPRINT_DEPTH moves) after each game
    2. Every CLUSTER_INTERVAL games, cluster all fingerprints from winning games
    3. Clusters with win_rate > WIN_RATE_THRESHOLD AND size > MIN_CLUSTER_SIZE
       AND similarity < MAX_KNOWN_SIMILARITY → candidate emergent strategy
    4. Register as EmergentStrategy_XXX in strategy_stats.json
    5. Build counters organically: scan games where the emergent strategy LOST,
       find what the winner did differently → register as counter pattern
    """
    FINGERPRINT_DEPTH    = 8
    MIN_CLUSTER_SIZE     = 15
    WIN_RATE_THRESHOLD   = 0.62
    MAX_KNOWN_SIMILARITY = 0.65
    CLUSTER_INTERVAL     = 500   # Games between cluster runs

    def build_counter_from_losses(self, emergent_strategy_id):
        """
        Find the common winning response among all games where this
        emergent strategy was DEFEATED. Register as counter candidate.
        No human involvement needed — counters emerge from data.
        """
        games_vs = self.loss_log[emergent_strategy_id]
        winner_moves = [game.moves_for(game.winner) for game in games_vs]
        common_patterns = extract_common_patterns(winner_moves, min_frequency=0.6)
        for pattern in common_patterns:
            strategy_tracker.register_counter(
                against=emergent_strategy_id,
                counter=CounterPattern(
                    action_type=pattern.action_type,
                    turn_range=pattern.turn_range,
                    win_rate=pattern.win_rate,
                    sample_size=len(games_vs),
                )
            )
```

**The full counter-strategy loop in operation:**

```
Turn 3:  recognize_strategy → "Emergent_007" (confidence 0.72)
         strategy_tracker: best counter → "early left-wall + advance" (win_rate 0.61)
         strategy_vector[10:13] = [0, 1, 0], strategy_vector[13] = 0.61

Turn 4:  network sees counter recommendation in strategy_vector
         MCTS: agreement_bonus on walls in counter zone
         Model places counter wall

Turn 8:  path_differential = +2 (winning the race)
         strategy_tracker: counter_win_rate updated 0.61 → 0.63

After game (loss scenario):
         backward_analyzer → turn 4 was decisive
         counter_planner → "missed counter Y, +2.3 rating gain"
         High-priority corrective experience injected to replay buffer
         EmergentStrategyDetector → game logged for counter refinement
```

---

## 12. Self-Play Training Pipeline

```python
# Per-game loop inside each worker:
def play_self_play_game(network, strategy_guide, recorder):
    board      = Board()
    mcts_tt    = MCTSTranspositionTable()
    endgame_tt = SolverTranspositionTable()

    while not board.is_terminal():
        counter_rec  = counter_planner.get_in_game_recommendation(board)
        strat_vector = strategy_guide.compute_strategy_vector(board, counter_rec)
        move         = move_selector.select_move(
            board, network, strategy_guide, mcts_tt, endgame_tt, TIME_BUDGET_TRAIN
        )
        board.apply_inplace(move)
        recorder.record(move, strat_vector, ...)

    # Post-game
    actual_ratings = compute_all_actual_ratings(recorder.game_record)
    recorder.set_actual_ratings(actual_ratings)
    loser  = 1 - board.winner
    winner = board.winner

    backward_analyzer.analyze(recorder.game_record, loser)
    counter_planner.plan(recorder.game_record, loser)
    deviation_logger.log_profitable_deviations(recorder.game_record, winner)
    agreement_weight_updater.update(recorder.game_record)
    emergent_detector.analyze_game(recorder.game_record)

    mcts_tt.clear()
    endgame_tt.clear()
```

**Experience priority tiers:**

```python
class PrioritizedExperienceBuffer:
    PRIORITY_CORRECTIVE = 5.0   # Backward-analysis mistakes
    PRIORITY_DEVIATION  = 4.0   # Profitable strategy deviations
    PRIORITY_STANDARD   = 1.0   # Regular self-play
```

**Champion mechanism:** workers always play the current model against the frozen champion. The champion updates only when the current model beats it at >55% over 50 evaluation games. This prevents policy oscillation.

---

---

# Part III — System Design

---

## 13. System Architecture Overview

The project has three distinct runtime environments that must integrate cleanly:

```
┌─────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT 1: Google Colab / Kaggle (Training)                │
│  - Runs self-play loop + trainer                                │
│  - GPU: T4 (Colab free) or P100 (Kaggle free 30hr/wk)          │
│  - Saves checkpoints → Hugging Face Hub (free model storage)   │
│  - Saves game records → Google Drive (15GB free)               │
└──────────────────────────┬──────────────────────────────────────┘
                           │ model weights (via HF Hub API)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT 2: Backend API (Render.com free tier)              │
│  - FastAPI app serving move predictions                         │
│  - Loads model weights from Hugging Face Hub on startup         │
│  - Handles game state management                                │
│  - Rate limiting, auth, game persistence                        │
│  - Spins down after 15min idle (free tier limitation)          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ REST API
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ENVIRONMENT 3: Frontend (Cloudflare Pages — free)              │
│  - Static React app — the game board + UI                       │
│  - Served globally via Cloudflare CDN                           │
│  - Communicates with backend via REST                           │
└─────────────────────────────────────────────────────────────────┘
```

**Full Tech Stack (zero-cost):**

| Layer | Tool | Free Tier Limit |
|---|---|---|
| Training compute | Google Colab | ~12hr GPU/session |
| Training compute (alt) | Kaggle Notebooks | 30hr GPU/week |
| Model storage | Hugging Face Hub | Unlimited public repos |
| Game record storage | Google Drive | 15GB |
| Frontend hosting | Cloudflare Pages | Unlimited bandwidth |
| Backend API | Render.com | 512MB RAM, spins down after 15min idle |
| Database | Supabase | 500MB PostgreSQL + 50MB file storage |
| Auth | Supabase Auth | Up to 50k MAU |
| CDN | Cloudflare (included with Pages) | Unlimited |
| CI/CD | GitHub Actions | 2000 min/month (public repo: unlimited) |
| Error tracking | Sentry | 5k errors/month |
| Uptime monitoring | UptimeRobot | 50 monitors, 5min checks |
| Version control | GitHub | Unlimited public repos |
| Secrets management | GitHub Secrets + Render Env Vars | Free |

---

## 14. Complete File Structure

```
quoridor_ai/
│
├── game_engine/
│   ├── __init__.py
│   ├── board.py                          # BoardState dataclass, tensor conversion
│   ├── moves.py                          # Move generation, validation, application
│   ├── pathfinder.py                     # BFS distance maps (fast: deque + early exit)
│   ├── rules.py                          # Wall legality, anti-blockade DFS check
│   ├── game.py                           # Game loop, turn management, termination
│   └── zobrist_hash.py                   # Full hash + O(1) incremental update
│
├── strategy/
│   ├── __init__.py
│   ├── strategy_guide.py                 # Rules as agreement functions (never negative)
│   ├── strategy_advisor.py               # compute_agreement_scores(), total_agreement_bonus()
│   ├── opening_recognizer.py             # Classify opponent strategy by turn 4
│   ├── strategy_tracker.py               # W/L/rating per strategy + counter tracking
│   ├── counter_planner.py                # Known + emergent counter recommendation
│   ├── agreement_weight_updater.py       # Per-rule weight learning + slow decay
│   ├── emergent_strategy_detector.py     # Fingerprinting, clustering, counter building
│   └── exception_logger.py              # Cases where model outperformed the guide
│
├── models/
│   ├── __init__.py
│   ├── board_encoder.py                  # Input tensor [10×9×9] construction
│   ├── residual_block.py                 # Pre-activation ResBlock
│   ├── path_attention.py                 # PathAttentionGate (learned attention_scale)
│   ├── strategy_layer.py                 # Strategy vector injected as input feature
│   ├── policy_head.py                    # → [136] action probabilities
│   ├── value_head.py                     # → scalar ∈ [-1, +1]
│   ├── rating_head.py                    # → scalar ∈ [0, 5]
│   └── quoridor_net.py                   # Master network: composes all submodules
│
├── search/
│   ├── __init__.py
│   ├── transposition_table.py            # MCTSTranspositionTable + SolverTranspositionTable
│   ├── mcts.py                           # DAG-MCTS: time-budgeted, agreement UCB
│   ├── endgame_solver.py                 # MTD(f) + alpha-beta + SolverTT
│   ├── opponent_predictor.py             # 1-step opponent move prediction
│   └── move_selector.py                  # Phase detection + dispatch to MCTS or solver
│
├── training/
│   ├── __init__.py
│   ├── self_play_worker.py               # Single game: play → rate → analyze → log
│   ├── self_play_manager.py              # Orchestrate N parallel workers
│   ├── experience_buffer.py              # Prioritized replay: corrective > deviation > standard
│   ├── loss_functions.py                 # 5-component loss (strategy = reward bonus)
│   ├── trainer.py                        # Training loop, optimizer, LR scheduler
│   └── evaluator.py                      # Win rate vs champion, avg rating
│
├── analysis/
│   ├── __init__.py
│   ├── game_recorder.py                  # Record all moves + metadata per game
│   ├── move_rater.py                     # Path differential delta → 0–5 rating
│   ├── backward_analyzer.py              # Last-to-first error attribution
│   ├── deviation_logger.py               # Log profitable strategy disagreements
│   └── strategy_updater.py              # Update tracker + feed corrective experiences
│
├── api/                                  # ← NEW: Backend API layer
│   ├── __init__.py
│   ├── main.py                           # FastAPI app, CORS, startup/shutdown
│   ├── routes/
│   │   ├── game.py                       # POST /game/new, POST /game/move, GET /game/{id}
│   │   ├── auth.py                       # POST /auth/login, POST /auth/register
│   │   ├── leaderboard.py               # GET /leaderboard
│   │   └── health.py                     # GET /health (uptime check)
│   ├── models/
│   │   ├── game_schema.py                # Pydantic models for request/response
│   │   └── user_schema.py
│   ├── services/
│   │   ├── ai_service.py                 # Loads model, exposes predict_move()
│   │   ├── game_service.py               # Game state management
│   │   └── auth_service.py              # JWT validation, Supabase auth bridge
│   ├── middleware/
│   │   ├── rate_limiter.py              # In-memory rate limiting
│   │   ├── auth_middleware.py           # JWT verification
│   │   └── error_handler.py             # Global exception → structured response
│   └── db/
│       ├── connection.py                 # Supabase client setup
│       └── queries.py                    # SQL queries (parameterized, never raw)
│
├── frontend/                             # ← NEW: React frontend
│   ├── public/
│   │   ├── index.html
│   │   └── favicon.ico
│   ├── src/
│   │   ├── components/
│   │   │   ├── Board/
│   │   │   │   ├── Board.tsx             # 9×9 game board
│   │   │   │   ├── Cell.tsx              # Individual cell with wall rendering
│   │   │   │   ├── Pawn.tsx
│   │   │   │   └── Wall.tsx
│   │   │   ├── GamePanel/
│   │   │   │   ├── MoveHistory.tsx
│   │   │   │   ├── WallCounter.tsx
│   │   │   │   └── AIThinking.tsx        # Shows AI "thinking" animation
│   │   │   ├── Auth/
│   │   │   │   ├── LoginModal.tsx
│   │   │   │   └── RegisterModal.tsx
│   │   │   └── UI/
│   │   │       ├── Button.tsx
│   │   │       └── Modal.tsx
│   │   ├── hooks/
│   │   │   ├── useGame.ts                # Game state, move submission
│   │   │   └── useAuth.ts                # Supabase auth state
│   │   ├── services/
│   │   │   └── api.ts                    # Axios client with retry + error handling
│   │   ├── store/
│   │   │   └── gameStore.ts             # Zustand game state store
│   │   ├── App.tsx
│   │   └── main.tsx
│   ├── package.json
│   └── vite.config.ts
│
├── data/
│   ├── games/                            # Compressed JSON game records (local dev)
│   ├── checkpoints/                      # Model weight snapshots (local dev)
│   ├── strategy_stats.json
│   ├── agreement_weights.json            # Per-rule learned weights
│   ├── emergent_strategies.json          # Discovered novel strategies
│   ├── deviation_log.json                # Profitable deviations log
│   └── exception_log.json
│
├── scripts/
│   ├── upload_checkpoint.py             # Push model weights to Hugging Face Hub
│   ├── download_checkpoint.py           # Pull latest weights for inference
│   ├── colab_setup.sh                   # One-command Colab environment setup
│   └── run_training.py                  # Entry: launches self-play on Colab
│
├── notebooks/
│   ├── train.ipynb                      # Colab training notebook
│   └── eval.ipynb                       # Evaluation + visualization notebook
│
├── tests/
│   ├── test_game_engine.py
│   ├── test_strategy_guide.py
│   ├── test_mcts.py
│   ├── test_endgame_solver.py
│   ├── test_api.py
│   └── fixtures/
│       └── test_positions.json          # Known positions with expected outputs
│
├── .github/
│   └── workflows/
│       ├── test.yml                     # Run pytest on push/PR
│       ├── deploy_frontend.yml          # Deploy to Cloudflare Pages on main merge
│       └── deploy_backend.yml           # Deploy to Render on main merge
│
├── config.py                            # All hyperparameters, paths, flags
├── train.py                             # Entry: launches self-play training
├── play.py                              # Entry: human vs bot (CLI)
├── evaluate.py                          # Entry: bot vs bot evaluation
├── requirements.txt                     # Python deps (ML + API)
├── requirements-dev.txt                 # Dev deps (pytest, black, ruff)
├── .env.example                         # Environment variable template
├── .gitignore
├── render.yaml                          # Render.com deployment config
└── README.md
```

---

## 15. Frontend Design

### Design Direction

**Subject:** A strategic board game AI. The audience is people who want to test themselves against a strong opponent. The page's single job: present the board clearly and make the AI feel like a real opponent, not a utility.

**Palette:**
- `#0D0D0D` — near-black background (the "table")
- `#F0EDE4` — off-white board grid (aged wood texture via CSS grain)
- `#C94F30` — player red (your pawn)
- `#2E5FA3` — AI blue (opponent pawn)
- `#8A7E6A` — wall color (dark walnut)
- `#E8C84A` — accent gold (highlights, active states)

**Typography:**
- Display: `DM Serif Display` (Google Fonts, free) — for the game title and score
- Body: `Inter` (Google Fonts, free) — clean, neutral for UI chrome
- Monospace: `JetBrains Mono` (Google Fonts, free) — move history notation

**Signature element:** When the AI is computing its move, the BFS path from each pawn to its goal flashes subtly across the board — a translucent heatmap showing "what the AI sees." This is both visually distinctive and genuinely informative. It disappears the moment the move is made.

### Board Component Architecture

```tsx
// src/components/Board/Board.tsx
// 9×9 grid. Walls render as thick borders between cells, not as separate elements.
// Click on a cell → pawn move. Click on a border gap → wall placement.

interface BoardProps {
    state: GameState;
    validMoves: ValidMove[];
    onMove: (move: Move) => void;
    showPathHeatmap: boolean;   // AI thinking mode
}
```

### Game State Flow

```
User clicks cell/border
    → useGame.makeMove(move)
    → api.postMove(gameId, move)           ← REST call to backend
    → Backend: validate move, apply move, call AI engine
    → Backend returns: {newState, aiMove, thinkingPath}
    → useGame updates local state
    → Board re-renders with AI's move
    → If showThinking: render heatmap for 800ms, then clear
```

### Pages

```
/            → Landing: game hero + "Play vs AI" CTA
/game/:id    → Active game: board + panels
/leaderboard → Win rates by registered user
/about       → System explanation (strategy guide, SINN architecture overview)
```

### Accessibility & Performance

- Keyboard navigation for moves (arrow keys for pawn, W key to enter wall-placement mode)
- ARIA labels on all board cells
- `prefers-reduced-motion` respected — heatmap animation disabled
- Lazy-load the game module: landing page stays under 50kB
- Vite build + tree shaking ensures production bundle < 200kB

---

## 16. API & Backend Logic

### `api/main.py` — FastAPI App

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from api.services.ai_service import AIService

ai_service = AIService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model weights from Hugging Face Hub on startup
    ai_service.load_model_from_hub(
        repo_id=os.getenv("HF_REPO_ID"),
        filename="latest_champion.pt"
    )
    yield
    # Cleanup on shutdown

app = FastAPI(title="Quoridor AI", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_URL", "https://quoridor-ai.pages.dev")],
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
)
```

### Routes

```python
# POST /game/new
# Body: { difficulty: "easy" | "medium" | "hard" }
# Returns: { gameId, boardState, currentPlayer }

# POST /game/{game_id}/move
# Body: { move: MoveDTO }   (pawn or wall)
# Returns: { boardState, aiMove, rating, isGameOver, winner }

# GET /game/{game_id}
# Returns: full GameRecord (for resume / spectate)

# GET /health
# Returns: { status: "ok", modelLoaded: bool, uptime_ms: int }

# GET /leaderboard
# Returns: top 20 users by win rate (min 10 games)
```

### `api/services/ai_service.py`

```python
class AIService:
    def load_model_from_hub(self, repo_id, filename):
        """
        Downloads weights from Hugging Face Hub.
        Cached locally after first download (Render ephemeral disk ~512MB).
        If disk is cleared on redeploy, re-downloads on startup.
        """
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.model = QuoridorNet()
        self.model.load_state_dict(torch.load(path, map_location='cpu'))
        self.model.eval()

    def predict_move(self, board_state: BoardState, difficulty: str) -> Move:
        """
        difficulty controls time budget:
          easy   → 100ms MCTS
          medium → 500ms MCTS
          hard   → 2000ms MCTS (or MTD(f) in endgame)
        """
        time_budget = {"easy": 100, "medium": 500, "hard": 2000}[difficulty]
        mcts_tt    = MCTSTranspositionTable()
        endgame_tt = SolverTranspositionTable()
        with torch.no_grad():
            return move_selector.select_move(
                board_state, self.model, strategy_guide,
                mcts_tt, endgame_tt, time_budget
            )
```

### Response Schema

```python
class MoveResponse(BaseModel):
    board_state:  BoardStateDTO
    ai_move:      MoveDTO
    move_rating:  float             # The AI's self-assessed rating for its move
    thinking_path: list[list[int]]  # BFS path cells — sent to frontend for heatmap
    is_game_over: bool
    winner:       Optional[int]
    thinking_ms:  int               # How long the AI took (transparency)

class ErrorResponse(BaseModel):
    error:   str
    code:    str   # "INVALID_MOVE", "GAME_NOT_FOUND", "RATE_LIMITED", etc.
    detail:  str
```

---

## 17. Database & Storage

### Supabase Schema (PostgreSQL, free 500MB)

```sql
-- Users
CREATE TABLE users (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email       TEXT UNIQUE NOT NULL,
    username    TEXT UNIQUE NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);

-- Games
CREATE TABLE games (
    id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    player_id     UUID REFERENCES users(id) ON DELETE SET NULL,
    difficulty    TEXT NOT NULL CHECK (difficulty IN ('easy', 'medium', 'hard')),
    winner        SMALLINT CHECK (winner IN (0, 1)),   -- 0=human, 1=AI, NULL=ongoing
    total_turns   INT,
    avg_rating    FLOAT,                               -- average move rating for AI
    created_at    TIMESTAMPTZ DEFAULT NOW(),
    completed_at  TIMESTAMPTZ
);

-- Individual moves (for replay + analysis)
CREATE TABLE moves (
    id            BIGSERIAL PRIMARY KEY,
    game_id       UUID REFERENCES games(id) ON DELETE CASCADE,
    turn_number   INT NOT NULL,
    player        SMALLINT NOT NULL,    -- 0=human, 1=AI
    move_type     TEXT NOT NULL,        -- 'pawn' | 'wall'
    move_data     JSONB NOT NULL,       -- {to_row, to_col} or {row, col, orientation}
    move_rating   FLOAT,               -- 0–5 for AI moves; null for human moves
    path_diff_before FLOAT,
    path_diff_after  FLOAT
);

-- Leaderboard view
CREATE VIEW leaderboard AS
SELECT
    u.username,
    COUNT(g.id) AS total_games,
    SUM(CASE WHEN g.winner = 0 THEN 1 ELSE 0 END) AS wins,
    ROUND(AVG(CASE WHEN g.winner = 0 THEN 1.0 ELSE 0.0 END) * 100, 1) AS win_rate_pct
FROM users u
JOIN games g ON g.player_id = u.id
WHERE g.winner IS NOT NULL
GROUP BY u.username
HAVING COUNT(g.id) >= 10
ORDER BY win_rate_pct DESC;

-- Indexes
CREATE INDEX idx_games_player_id ON games(player_id);
CREATE INDEX idx_moves_game_id   ON moves(game_id);
CREATE INDEX idx_games_completed ON games(completed_at);
```

### Model + Training Data Storage (Hugging Face Hub)

```
hf.co/{your-username}/quoridor-ai-sinn/
├── latest_champion.pt         ← always the current best model
├── checkpoint_1000.pt
├── checkpoint_2000.pt
├── ...
├── strategy_stats.json        ← live strategy effectiveness data
├── agreement_weights.json     ← per-rule learned weights
├── emergent_strategies.json   ← discovered novel strategies
└── README.md                  ← model card (required for HF Hub)
```

Uploads from Colab via `huggingface_hub`:

```python
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="checkpoints/champion.pt",
    path_in_repo="latest_champion.pt",
    repo_id="your-username/quoridor-ai-sinn",
    token=os.environ["HF_TOKEN"],  # stored as Colab secret
)
```

### Game Records (Google Drive, training session outputs)

```python
# From Colab training loop — after each session:
from google.colab import drive
drive.mount('/content/drive')

# Save compressed game records
with gzip.open('/content/drive/MyDrive/quoridor_ai/games/session_{ts}.jsonl.gz', 'wt') as f:
    for record in session_records:
        f.write(json.dumps(record) + '\n')
```

### Local Dev Storage

```
data/
├── games/          ← JSONL.GZ game records (gitignored)
├── checkpoints/    ← .pt files (gitignored — use .gitignore rule: data/checkpoints/*.pt)
├── strategy_stats.json    ← committed (small, human-readable)
└── agreement_weights.json ← committed
```

**What goes in Git:** code, configs, small JSON metadata, schema migrations.  
**What does not go in Git:** model weights, game records, large data files.

---

## 18. Auth & Permissions

### Supabase Auth (Free up to 50k MAU)

Auth is optional for the game (anonymous play is allowed) but required for the leaderboard.

```
Anonymous users:
  - Can start and play games
  - Game is saved with player_id = NULL
  - Cannot appear on leaderboard
  - Rate limited by IP

Registered users:
  - Same as anonymous + leaderboard eligibility
  - Can view own game history
  - Rate limited by user_id (more generous than IP)
```

**JWT Flow:**

```
Frontend login → Supabase Auth → JWT token (signed with Supabase secret)
Backend receives token in Authorization: Bearer header
api/middleware/auth_middleware.py verifies JWT signature using SUPABASE_JWT_SECRET
Decoded payload contains user_id → used for all DB writes
```

```python
# api/middleware/auth_middleware.py
import jwt
from fastapi import Request, HTTPException

async def verify_optional_auth(request: Request):
    """
    Returns user_id if valid JWT present, None for anonymous requests.
    Does NOT reject anonymous requests — they are allowed for gameplay.
    """
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return None
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(
            token,
            os.getenv("SUPABASE_JWT_SECRET"),
            algorithms=["HS256"],
            audience="authenticated"
        )
        return payload.get("sub")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except jwt.InvalidTokenError:
        return None    # Treat invalid token as anonymous
```

### Permission Matrix

| Action | Anonymous | Registered |
|---|---|---|
| Start new game | ✅ | ✅ |
| Make move | ✅ | ✅ |
| View game by ID | ✅ (own session) | ✅ |
| Appear on leaderboard | ❌ | ✅ (≥10 games) |
| View own game history | ❌ | ✅ |
| Access training endpoints | ❌ | ❌ (internal only) |

---

## 19. Hosting & Cloud (Free Tier Only)

### Frontend — Cloudflare Pages

**Why Cloudflare over GitHub Pages:** Cloudflare Pages builds from Git automatically, has no bandwidth cap, deploys globally across 300+ PoPs, and handles the CDN layer for free.

```yaml
# Cloudflare Pages build settings (configured in dashboard):
Build command:   cd frontend && npm run build
Build output:    frontend/dist
Environment variables:
  VITE_API_URL:  https://quoridor-ai.onrender.com
  VITE_SUPABASE_URL: https://xxx.supabase.co
  VITE_SUPABASE_ANON_KEY: eyJh...
```

### Backend API — Render.com

**Free tier:** 512MB RAM, 0.1 CPU, spins down after 15 min of inactivity (first request after spin-down takes ~30s to cold-start).

**Mitigating cold starts:**
1. UptimeRobot pings `/health` every 14 minutes → keeps the service warm
2. Frontend shows a "Waking up the AI..." banner when response time > 3s
3. `/health` endpoint returns instantly without loading the model (model loads on first actual request)

```yaml
# render.yaml
services:
  - type: web
    name: quoridor-ai-api
    runtime: python
    plan: free
    buildCommand: "pip install -r requirements.txt"
    startCommand: "uvicorn api.main:app --host 0.0.0.0 --port $PORT"
    envVars:
      - key: HF_REPO_ID
        sync: false        # Set in Render dashboard (not in repo)
      - key: HF_TOKEN
        sync: false
      - key: SUPABASE_URL
        sync: false
      - key: SUPABASE_KEY
        sync: false
      - key: SUPABASE_JWT_SECRET
        sync: false
      - key: FRONTEND_URL
        value: https://quoridor-ai.pages.dev
```

### Training — Google Colab + Kaggle

```bash
# scripts/colab_setup.sh — run once per Colab session
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install huggingface_hub numpy scipy scikit-learn
git clone https://github.com/your-username/quoridor-ai.git
cd quoridor-ai
python scripts/download_checkpoint.py  # Pull latest weights from HF Hub
```

**Colab session management strategy:**
- Training saves a checkpoint to HF Hub every 2000 games (roughly every 30–45 min)
- If session ends, resume from last HF Hub checkpoint in the next session
- Session state is never lost because HF Hub is the source of truth for weights

**Kaggle as backup/supplement:**
- Kaggle gives 30hr/week GPU time (P100 — slightly better than Colab's T4)
- Same training script works on both
- Use Kaggle Secrets to store `HF_TOKEN`

---

## 20. CI/CD & Version Control

### Branch Strategy

```
main          ← production-ready; triggers frontend + backend deploy
dev           ← integration branch; all feature branches merge here
feature/xxx   ← individual features
fix/xxx       ← bug fixes
```

### GitHub Actions Workflows

**`.github/workflows/test.yml`** — runs on every push and PR:

```yaml
name: Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v4
        with: { python-version: '3.11' }
      - run: pip install -r requirements-dev.txt
      - run: pytest tests/ -v --tb=short
      - run: ruff check .           # Linting
      - run: black --check .        # Formatting
```

**`.github/workflows/deploy_frontend.yml`** — on push to `main`:

```yaml
name: Deploy Frontend
on:
  push:
    branches: [main]
    paths: ['frontend/**']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: '20' }
      - run: cd frontend && npm ci && npm run build
      - uses: cloudflare/pages-action@v1
        with:
          apiToken: ${{ secrets.CF_API_TOKEN }}
          accountId: ${{ secrets.CF_ACCOUNT_ID }}
          projectName: quoridor-ai
          directory: frontend/dist
```

**`.github/workflows/deploy_backend.yml`** — on push to `main`:

```yaml
name: Deploy Backend
on:
  push:
    branches: [main]
    paths: ['api/**', 'game_engine/**', 'models/**', 'search/**']
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Trigger Render Deploy
        run: |
          curl -X POST "${{ secrets.RENDER_DEPLOY_HOOK_URL }}"
        # Render's deploy hook URL triggers a redeploy via webhook
```

### Git Configuration

```gitignore
# .gitignore
data/checkpoints/*.pt
data/games/*.jsonl.gz
data/games/*.json
__pycache__/
*.pyc
.env
node_modules/
frontend/dist/
.pytest_cache/
.ruff_cache/
```

**Model weights are never committed to Git.** They live on Hugging Face Hub and are pulled at deploy time. This keeps the repo lean and the training history separate from production artifacts.

---

## 21. Security

### Principles

1. **No secrets in code or Git** — all secrets in environment variables
2. **All API inputs validated** — Pydantic models reject malformed requests before they reach game logic
3. **SQL via parameterized queries only** — the `db/queries.py` module never formats strings into SQL
4. **CORS locked to known origin** — backend only accepts requests from the Cloudflare Pages domain
5. **HTTPS enforced everywhere** — Cloudflare and Render both terminate TLS; no plaintext traffic

### Input Validation (Pydantic)

```python
class MoveDTO(BaseModel):
    move_type: Literal["pawn", "wall"]
    # Pawn move fields
    to_row: Optional[int] = Field(None, ge=0, le=8)
    to_col: Optional[int] = Field(None, ge=0, le=8)
    # Wall move fields
    row: Optional[int]    = Field(None, ge=0, le=7)
    col: Optional[int]    = Field(None, ge=0, le=7)
    orientation: Optional[Literal["h", "v"]] = None

    @model_validator(mode='after')
    def check_fields_for_type(self):
        if self.move_type == "pawn":
            assert self.to_row is not None and self.to_col is not None
        else:
            assert self.row is not None and self.col is not None \
                   and self.orientation is not None
        return self
```

Game logic then independently validates the move against board rules — Pydantic only validates shape and range, not legality.

### Secrets Management

```
Local dev:   .env file (gitignored). Copy from .env.example and fill in values.
Colab:       Colab Secrets (sidebar > key icon) — accessed via userdata.get('KEY')
Render:      Environment Variables in Render dashboard
GitHub CI:   GitHub Secrets — injected as environment variables in workflows
```

```bash
# .env.example (committed — no real values)
SUPABASE_URL=https://xxx.supabase.co
SUPABASE_KEY=your-supabase-anon-key
SUPABASE_JWT_SECRET=your-jwt-secret
HF_REPO_ID=your-username/quoridor-ai-sinn
HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx
FRONTEND_URL=http://localhost:5173
```

---

## 22. Rate Limiting

### Strategy

Two tiers — anonymous (IP-based, strict) and authenticated (user-based, generous):

```python
# api/middleware/rate_limiter.py
from collections import defaultdict
import time

class InMemoryRateLimiter:
    """
    Simple sliding window counter.
    No Redis needed for this scale — in-memory is fine for a free-tier service
    with one server instance.
    """
    def __init__(self):
        self.counters = defaultdict(list)  # key -> [timestamp, ...]

    def is_allowed(self, key: str, max_requests: int, window_seconds: int) -> bool:
        now = time.time()
        window_start = now - window_seconds
        # Remove old entries
        self.counters[key] = [t for t in self.counters[key] if t > window_start]
        if len(self.counters[key]) >= max_requests:
            return False
        self.counters[key].append(now)
        return True

rate_limiter = InMemoryRateLimiter()

# Rate limit tiers:
LIMITS = {
    "anonymous_move":      (30, 60),    # 30 moves per minute per IP
    "anonymous_new_game":  (5,  60),    # 5 new games per minute per IP
    "authenticated_move":  (120, 60),   # 120 moves per minute per user_id
    "authenticated_new":   (20, 60),    # 20 new games per minute per user_id
}
```

### Applied in Routes

```python
@router.post("/game/{game_id}/move")
async def make_move(game_id: UUID, move: MoveDTO, request: Request,
                    user_id: Optional[str] = Depends(verify_optional_auth)):
    key  = user_id if user_id else request.client.host
    tier = "authenticated_move" if user_id else "anonymous_move"
    max_req, window = LIMITS[tier]
    if not rate_limiter.is_allowed(key, max_req, window):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please slow down.",
            headers={"Retry-After": "60"}
        )
    # ... rest of handler
```

---

## 23. Caching & CDN

### Cloudflare CDN (Frontend)

All static assets (JS, CSS, fonts, images) are cached at Cloudflare's edge globally. Cache headers set by Vite build:

```javascript
// vite.config.ts
export default defineConfig({
    build: {
        rollupOptions: {
            output: {
                // Content-hash filenames → infinite cache TTL for assets
                entryFileNames:  'assets/[name]-[hash].js',
                chunkFileNames:  'assets/[name]-[hash].js',
                assetFileNames:  'assets/[name]-[hash].[ext]',
            }
        }
    }
})
```

### API Response Caching

```python
# api/services/game_service.py
# Cache active game states in memory (ephemeral, per-process)
# This avoids a Supabase round-trip on every move

from functools import lru_cache
import time

_game_cache: dict[str, tuple[BoardState, float]] = {}  # game_id -> (state, timestamp)
GAME_CACHE_TTL = 300  # 5 minutes

def get_game_state(game_id: str) -> Optional[BoardState]:
    if game_id in _game_cache:
        state, ts = _game_cache[game_id]
        if time.time() - ts < GAME_CACHE_TTL:
            return state
    # Cache miss — fetch from Supabase
    state = db.queries.get_game(game_id)
    if state:
        _game_cache[game_id] = (state, time.time())
    return state
```

### Static Data Caching (Strategy Stats)

```python
# Strategy stats, agreement weights, and emergent strategies are loaded
# once at startup and refreshed every 6 hours. These change slowly — no need
# to hit Supabase on every request.

@asynccontextmanager
async def lifespan(app: FastAPI):
    strategy_guide.load_stats()          # Load from HF Hub JSON on startup
    # Refresh task every 6 hours
    asyncio.create_task(refresh_strategy_stats_loop())
    yield
```

### What to Cache vs. Not Cache

| Resource | Cache? | Why |
|---|---|---|
| Static frontend assets | Yes — indefinitely (via hash) | Immutable after build |
| Active game state | Yes — in-memory, 5min TTL | Avoids DB round-trip per move |
| Strategy stats JSON | Yes — 6hr refresh | Changes only after training runs |
| Model weights | Yes — disk cache (Render ephemeral) | 100MB+; expensive to re-download |
| User game history | No | Must always be fresh |
| Leaderboard | Yes — 60s TTL | Acceptable staleness |

---

## 24. Error Tracking & Logs

### Sentry (Free: 5k errors/month)

```python
# api/main.py
import sentry_sdk
from sentry_sdk.integrations.fastapi import FastApiIntegration

sentry_sdk.init(
    dsn=os.getenv("SENTRY_DSN"),
    integrations=[FastApiIntegration()],
    traces_sample_rate=0.1,     # 10% of requests traced (keeps within free tier)
    environment=os.getenv("ENV", "development"),
    release=os.getenv("GIT_COMMIT_SHA", "unknown"),
)
```

Sentry automatically captures unhandled exceptions. For the game engine:

```python
# Capture a specific error with context
import sentry_sdk
with sentry_sdk.push_scope() as scope:
    scope.set_tag("game_id", game_id)
    scope.set_extra("board_state", board_state.to_dict())
    sentry_sdk.capture_exception(e)
```

### Structured Logging

```python
# api/middleware/error_handler.py — global exception handler
import logging
import json

logger = logging.getLogger("quoridor_ai")
logger.setLevel(logging.INFO)

# JSON format for easy parsing in Render logs
handler = logging.StreamHandler()
handler.setFormatter(
    logging.Formatter('{"time":"%(asctime)s","level":"%(levelname)s","msg":%(message)s}')
)
logger.addHandler(handler)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(json.dumps({
        "error": str(exc),
        "type":  type(exc).__name__,
        "path":  str(request.url),
        "method": request.method,
    }))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            code="INTERNAL_ERROR",
            detail="The AI encountered an unexpected error. Please try again."
        ).dict()
    )
```

### Error Codes

All API errors return a consistent structure:

| HTTP Status | Code | Meaning |
|---|---|---|
| 400 | `INVALID_MOVE` | Move failed game-rule validation |
| 400 | `INVALID_REQUEST` | Malformed request body |
| 401 | `UNAUTHORIZED` | Invalid or expired JWT |
| 404 | `GAME_NOT_FOUND` | No game with this ID |
| 409 | `GAME_OVER` | Game already ended |
| 422 | `VALIDATION_ERROR` | Pydantic model validation failed |
| 429 | `RATE_LIMITED` | Too many requests |
| 500 | `INTERNAL_ERROR` | Unexpected server error (logged + reported to Sentry) |
| 503 | `MODEL_NOT_READY` | Model still loading on cold start |

---

## 25. Monitoring & Alerts

### UptimeRobot (Free: 50 monitors, 5-min checks)

Set up two monitors in UptimeRobot:

```
Monitor 1: Backend API
  URL: https://quoridor-ai.onrender.com/health
  Method: GET
  Check interval: 5 minutes
  Alert: Email if down for 2 consecutive checks
  Secondary purpose: Keeps Render free-tier service warm (pings every 5min < 15min idle threshold)

Monitor 2: Frontend
  URL: https://quoridor-ai.pages.dev
  Method: GET
  Check interval: 5 minutes
  Alert: Email if down for 2 consecutive checks
```

### Render Metrics Dashboard

Render's free tier includes basic metrics:
- Request count and response times
- Memory usage (critical — 512MB cap)
- CPU usage

**Memory alert:** If the model + server overhead approaches 400MB, set Render's alert. Model loading strategies if memory is tight:

```python
# Load model in float16 (half precision) to halve memory usage
# Inference quality is effectively identical
self.model.half()  # ~160MB instead of ~320MB for 3.2M param model
torch.backends.cudnn.benchmark = True
```

### `/health` Endpoint (Detailed)

```python
@router.get("/health")
async def health():
    return {
        "status":        "ok",
        "model_loaded":  ai_service.model is not None,
        "model_version": ai_service.model_version,
        "uptime_ms":     int((time.time() - START_TIME) * 1000),
        "games_served":  metrics.games_served,
        "db_connected":  await db.ping(),
    }
```

### Grafana Cloud (Free: 10k metrics/month, 14-day retention)

Optional: if basic metrics aren't enough, push custom metrics to Grafana Cloud via their free Prometheus remote write endpoint.

```python
# Key metrics worth tracking:
# - ai_inference_duration_ms (histogram)
# - games_started_total (counter)
# - games_completed_total (counter, labeled by winner: human/ai)
# - mcts_simulations_per_move (histogram)
# - endgame_solver_activations_total (counter)
```

---

## 26. Scaling

### Current Architecture Constraints (Free Tier)

| Bottleneck | Limit | Impact |
|---|---|---|
| Render single instance | 512MB RAM, 1 process | One game at a time in MCTS |
| Render no persistence | Ephemeral disk | Model re-downloads on each deploy |
| Supabase free | 500MB DB | ~5M rows before hitting limit |
| Colab GPU | 12hr sessions | Training interrupted; checkpoint saves mitigate |

### Concurrency Within Free Tier

The MCTS runs CPU-bound Python for 100–2000ms per move. To serve concurrent games without blocking:

```python
# api/services/ai_service.py
import asyncio
from concurrent.futures import ProcessPoolExecutor

executor = ProcessPoolExecutor(max_workers=2)  # 2 workers within 512MB

async def predict_move_async(board_state, difficulty) -> Move:
    """
    Runs MCTS in a separate process, freeing the event loop
    to handle other requests (health checks, new game creation).
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        predict_move_sync,  # The actual CPU-bound work
        board_state, difficulty
    )
```

### Scale-Up Path (When Ready to Pay)

The architecture is designed so scaling requires only configuration changes, not code rewrites:

| Scale Need | Free Tier | Paid Solution |
|---|---|---|
| More concurrent users | ProcessPoolExecutor (2 workers) | Render paid → multiple instances + Redis for state |
| Persistent warm API | UptimeRobot pings | Render Starter ($7/mo) — no spin-down |
| More training compute | Colab + Kaggle rotation | Lambda Labs ($0.60/hr A100) |
| More DB | Supabase 500MB | Supabase Pro ($25/mo) → 8GB |
| Better CDN | Cloudflare Pages | Same — Cloudflare is already excellent at free tier |

### Stateless Backend Design

All game state is stored in Supabase, not in server memory (except the 5-min LRU cache). This means:
- Any request can be handled by any server instance
- Render can redeploy at any time without losing in-progress games
- Scaling to multiple instances requires only Redis for the LRU cache

---

## 27. Training Configuration

```python
# config.py — complete hyperparameter reference

# ── Network ──────────────────────────────────────────────────────────────
NUM_RES_BLOCKS           = 15
CHANNELS                 = 128
STRATEGY_DIM             = 16       # 16-dim vector (extended from 12)
STRATEGY_EMBED_DIM       = 16
ATTENTION_SCALE_INIT     = 0.5      # Learned — network can adjust

# ── MCTS ─────────────────────────────────────────────────────────────────
MCTS_TIME_BUDGET_TRAIN   = 300      # ms per move during self-play
MCTS_TIME_BUDGET_EVAL    = 1000     # ms during evaluation
MCTS_TIME_BUDGET_PLAY    = 2000     # ms vs human
C_PUCT                   = 1.5
LAMBDA_GUIDE             = 0.3      # Path-guide bonus weight in UCB
LAMBDA_AGREE             = 0.1      # Agreement bonus weight in UCB (small; never negative)
TOP_K_LOOKAHEAD          = 5

# ── Transposition Tables ─────────────────────────────────────────────────
MCTS_TT_MAX_SIZE         = 500_000
SOLVER_TT_MAX_SIZE       = 2_000_000

# ── Phase Switching ───────────────────────────────────────────────────────
ENDGAME_WALLS_HARD       = 4
ENDGAME_WALLS_SOFT       = 6
ENDGAME_PATH_SOFT        = 5
WIN_VALUE                = 1000

# ── Training ─────────────────────────────────────────────────────────────
BATCH_SIZE               = 512
LEARNING_RATE            = 0.001
LR_DECAY                 = 0.95      # Multiplicative, every 5000 steps
WEIGHT_DECAY             = 1e-4
GRAD_CLIP                = 1.0
BUFFER_SIZE              = 500_000
MIN_BUFFER               = 10_000
UPDATE_FREQ              = 100       # Training steps per game batch
CHAMPION_UPDATE_GAMES    = 1000
CHAMPION_WIN_THRESHOLD   = 0.55
DIRICHLET_ALPHA          = 0.3
DIRICHLET_WEIGHT         = 0.25

# ── Loss Weights ─────────────────────────────────────────────────────────
W_POLICY                 = 1.0
W_VALUE                  = 1.0
W_RATING                 = 0.5
W_RATING_MAX             = 0.3
W_DEFEAT                 = 0.5
W_STRATEGY_INIT          = 0.40     # Agreement REWARD weight at start
W_STRATEGY_MID           = 0.15     # After 50k steps (linear anneal)
W_STRATEGY_FINAL         = 0.05     # After 150k steps (held — never zero)
W_STRATEGY_MID_STEP      = 50_000
W_STRATEGY_FINAL_STEP    = 150_000

# ── Self-Play ─────────────────────────────────────────────────────────────
NUM_WORKERS              = 8
GAMES_PER_EVAL           = 50

# ── Emergent Strategy Detection ───────────────────────────────────────────
EMERGENT_FINGERPRINT_DEPTH     = 8
EMERGENT_MIN_CLUSTER_SIZE      = 15
EMERGENT_WIN_RATE_THRESHOLD    = 0.62
EMERGENT_MAX_KNOWN_SIMILARITY  = 0.65
EMERGENT_CLUSTER_INTERVAL      = 500

# ── Agreement Weight Learning ─────────────────────────────────────────────
AGREEMENT_WEIGHT_LR            = 0.01
AGREEMENT_WEIGHT_DECAY_RATE    = 0.999
```

### Training Throughput Estimate (24hr, Colab T4 + 8 CPU workers)

| Configuration | Games/Day | Quality |
|---|---|---|
| Tree MCTS, fixed 200 sims | ~15,000 | Baseline |
| + Transposition Tables (DAG) | ~21,000–22,500 (+40%) | Pooled Q/N |
| + Time-based MCTS (300ms) | ~22,000–24,000 | Consistent across phases |
| + MTD(f) endgame solver | Marginal game count gain | Better endgame experiences |

### Expected Milestones (Single 24-hour Run)

| Hour | What to Watch |
|---|---|
| 0–2 | Loss components decreasing; no NaN; strategy_bonus is negative (reward active) |
| 2–4 | Opening strategy recognition converges; model learns to advance toward goal |
| 4–8 | Basic wall efficiency emerges; path differential becomes dominant signal |
| 8–16 | Mid-game patterns develop; strategy_bonus plateaus (model diverging from guide — good) |
| 16–24 | Late-game refinement; endgame solver transitions clean; avg_rating > 3.5 |

**Checkpoint recommendation:** save to Hugging Face Hub every 2 hours. Monitor `avg_move_rating`, `strategy_bonus`, `value_loss`. If `value_loss` plateaus before hour 8, reduce LR by 0.5×.

---

## 28. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING SYSTEM (Colab/Kaggle)                    │
│                                                                          │
│  ┌─────────────────┐  game records   ┌────────────────────────────────┐  │
│  │  Self-Play       │──────────────▶ │    Experience Buffer           │  │
│  │  Workers (N=8)   │               │    corrective > deviation       │  │
│  │                  │◀── weights ─── │    > standard                  │  │
│  └───────┬──────────┘               └──────────────┬─────────────────┘  │
│          │                                         │ batches             │
│          │                                         ▼                    │
│          │                          ┌────────────────────────────────┐  │
│          │ reads                    │          TRAINER               │  │
│          │                          │  policy + value + rating       │  │
│  ┌───────▼──────────┐               │  + strategy_agreement_REWARD   │  │
│  │  Strategy Guide   │              │  (not penalty)                 │  │
│  │  - agreement only │              └────────────────────────────────┘  │
│  │  - per-rule weights│                                                  │
│  └───────┬──────────┘                                                   │
│          │ updates                                                       │
│          ▼                                                               │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │  STRATEGY FEEDBACK LOOP                                            │ │
│  │  AgreementWeightUpdater ◄── per-game outcomes                      │ │
│  │  EmergentStrategyDetector ◄── deviation log                        │ │
│  │  StrategyTracker ◄── backward analysis results                     │ │
│  │  CounterPlanner ◄── detected opponent strategies                   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  Checkpoints ──────────────────────────────▶ Hugging Face Hub           │
│  Game records ─────────────────────────────▶ Google Drive               │
└──────────────────────────────────────────────────────────────────────────┘

INFERENCE PIPELINE (per API request):

  BoardState (from DB / request)
      │
      ├── ZobristHasher ──────────────────▶ current_hash (O(1))
      ├── BFS Pathfinder ─────────────────▶ path_map_own, path_map_opp [9×9]
      ├── Opening Recognizer ─────────────▶ strategy_label (known|emergent|unknown)
      ├── Counter Planner ────────────────▶ counter_recommendation + confidence
      └── Strategy Advisor ───────────────▶ agreement_scores + strategy_vector [16]
      │
      ▼
  Board Tensor [10×9×9]  +  Strategy Vector [16]
      │
      ▼
  QuoridorSINN (~3.2M params, fp16)
      │   Stem → Group1 → PathAttentionGate
      │        → StrategyLayer (advice as input feature)
      │        → Group2 → PathAttentionGate
      │        → Group3 → PathAttentionGate
      │        → GlobalAvgPool → [128]
      ├── Policy Head → [136] action priors
      ├── Value Head  → scalar ∈ [-1, +1]
      └── Rating Head → scalar ∈ [0, 5]
      │
      ▼
  Phase Detection
      ├── total_walls ≤ 4            ──▶ MTD(f) + SolverTT  (exact, <20ms)
      ├── total_walls ≤ 6 + path ≤ 5 ──▶ MTD(f) + SolverTT
      └── otherwise                  ──▶ DAG-MCTS (time-budgeted)
                                           ▼
                                      Top-K moves
                                           ▼
                                      1-Step Opponent Lookahead
                                           ▼
                                      Final Move  ──▶ API Response
                                                      (move + rating + path heatmap)

SYSTEM REQUEST FLOW:

  User (browser)
      │ HTTPS
      ▼
  Cloudflare CDN ──────────────▶ Static frontend assets (cached globally)
      │ Origin request (cache miss)
      ▼
  Cloudflare Pages (origin)     ← React app served from here
      │ REST API call (HTTPS)
      ▼
  Render.com (backend API)
      │
      ├── Rate limiter check
      ├── JWT verification (optional)
      ├── Game state fetch (Supabase or in-memory cache)
      ├── Move validation (game engine)
      ├── AI inference (MCTS / MTD(f))
      ├── Game state persist (Supabase)
      └── Response
      │
      ▼
  User browser ← { newBoardState, aiMove, thinkingPath, rating }
```

---

## 29. Key Algorithms (Pseudocode)

### Complete Self-Play Game

```
function play_self_play_game(network, strategy_guide, recorder):
    board = Board()
    board.current_hash = zobrist.full_hash(board)
    mcts_tt    = MCTSTranspositionTable()    # Per-game; persists across all turns
    endgame_tt = SolverTranspositionTable()  # Per-game; persists across all turns

    while not board.is_terminal():
        player = board.current_player
        counter_rec  = counter_planner.get_in_game_recommendation(board)
        strat_vector = strategy_guide.compute_strategy_vector(board, counter_rec)

        # Predict before move (for predicted vs actual comparison)
        _, _, pred_rating = network(board_to_tensor(board), strat_vector)

        move = move_selector.select_move(
            board, network, strategy_guide, mcts_tt, endgame_tt,
            time_budget_ms=MCTS_TIME_BUDGET_TRAIN
        )
        board.apply_inplace(move)   # Hash updated O(1) inside
        recorder.record(player, move, mcts.last_visit_dist(), pred_rating, strat_vector)

    # Post-game: compute actual ratings for ALL moves
    actual_ratings = compute_all_actual_ratings(recorder.game_record)
    recorder.set_actual_ratings(actual_ratings)

    loser  = 1 - board.winner
    winner = board.winner

    backward_analyzer.analyze(recorder.game_record, loser, network)
    counter_planner.plan(recorder.game_record, loser)
    deviation_logger.log_profitable_deviations(recorder.game_record, winner)
    agreement_weight_updater.update(recorder.game_record, board.winner)
    emergent_detector.analyze_game(recorder.game_record)

    mcts_tt.clear()
    endgame_tt.clear()
    return recorder.finalize()
```

### DAG-MCTS Simulation

```
function _run_simulation(root_board, network, strategy_guide):
    node = mcts_tt.get(root_board.current_hash)
    path = []

    # SELECTION
    while node.is_fully_expanded() and not node.is_terminal():
        action = node.select_by_ucb_strategy(strategy_guide)
        path.append((node.hash, action))
        node = mcts_tt.get(node.children[action])

    # EXPANSION
    if not node.is_terminal():
        node.expand(network, mcts_tt)
        child_action = node.select_unvisited()
        path.append((node.hash, child_action))
        node, _ = mcts_tt.get_or_create(node.children[child_action], ...)

    # EVALUATION (value network — no rollout needed)
    _, value, _ = network(board_to_tensor(node.board), compute_strategy_vector(node.board))

    # BACKPROPAGATION (simulation path only — not all DAG ancestors)
    for node_hash, _ in reversed(path):
        n = mcts_tt.get(node_hash)
        n.N += 1
        n.W += value
        n.Q  = n.W / n.N
        value = -value   # Alternating perspective
```

### Training Step

```
function training_step(batch, model, optimizer, strategy_guide, config):
    # Agreement bonuses — always ≥ 0, never negative
    agreement_bonuses = [
        strategy_guide.total_agreement_bonus(batch.agreements[i], agreement_weights)
        for i in range(len(batch))
    ]

    policy, value, rating = model(batch.state_tensors, batch.strategy_vectors)
    loss, breakdown = total_loss(batch, {policy, value, rating}, agreement_bonuses, config)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), breakdown
```

---

## 30. Implementation Phases

### Phase 1 — Game Engine (Days 1–3)
Files: `board.py`, `moves.py`, `pathfinder.py`, `rules.py`, `game.py`, `zobrist_hash.py`

Build `pathfinder.py` first — everything else depends on BFS being correct and fast.

**Validate:** 1000 random-vs-random games. Correct termination, no illegal moves reach the board, both players always have a path to goal, hash is consistent with board state on every move.

### Phase 2 — Strategy Guide (Days 3–5)
Files: `strategy_guide.py`, `strategy_advisor.py`, `opening_recognizer.py`, `strategy_tracker.py`, `counter_planner.py`, `agreement_weight_updater.py`

**Validate:** Feed sample board states. Verify all agreement scores ∈ [0, 1] with no negatives. Opening recognizer classifies known sequences correctly. Tracker updates correctly on synthetic game records.

### Phase 3 — Neural Network (Days 5–7)
Files: `board_encoder.py`, `residual_block.py`, `path_attention.py`, `strategy_layer.py`, `policy_head.py`, `value_head.py`, `rating_head.py`, `quoridor_net.py`

**Validate:** Random input [10×9×9] + [16] → correct output shapes. No NaN/Inf. `attention_scale` is a leaf parameter and `requires_grad=True`. Strategy layer output shape correct after concatenation.

### Phase 4 — Transposition Tables + Search (Days 7–10)
Files: `transposition_table.py`, `mcts.py`, `endgame_solver.py`, `opponent_predictor.py`, `move_selector.py`

**Validate:** Same board reached via two different paths maps to exactly one TT node. MCTS respects time budget (within ±10ms). MTD(f) returns the correct optimal move on trivial hand-crafted positions with known forced wins. Phase switching activates at correct wall counts.

### Phase 5 — Rating + Analysis (Days 10–12)
Files: `move_rater.py`, `game_recorder.py`, `backward_analyzer.py`, `deviation_logger.py`, `strategy_updater.py`, `emergent_strategy_detector.py`

**Validate:** Ratings ∈ [0, 5]. Blunder detection fires on clearly lost positions. Backward analyzer identifies the correct root mistake on reconstructed game records. Deviation logger fires on moves with agreement < 0.2 and rating ≥ 4.0.

### Phase 6 — Training Pipeline (Days 12–15)
Files: `experience_buffer.py`, `loss_functions.py`, `trainer.py`, `self_play_worker.py`, `self_play_manager.py`, `evaluator.py`

**Validate:** Buffer sampling is proportional to priority. Strategy loss term is always ≤ 0 (reward, not penalty). Value loss decreases over 1000 training steps on synthetic data. Workers don't deadlock on startup.

### Phase 7 — API Backend (Days 15–17)
Files: `api/` directory

**Validate:** All routes return correct status codes. Rate limiter fires at correct thresholds. Auth middleware allows anonymous play. Move validation rejects illegal moves. Model loads from HF Hub on startup.

### Phase 8 — Frontend (Days 17–20)
Files: `frontend/` directory

**Validate:** Board renders correct 9×9 grid. Pawn moves and wall placements work. AI thinking heatmap appears and disappears. Anonymous play works without login. Game persists on page refresh (game ID in URL).

### Phase 9 — Deployment + CI/CD (Days 20–21)
Files: `.github/workflows/`, `render.yaml`

**Validate:** Push to `main` triggers frontend deploy to Cloudflare Pages and backend deploy to Render. Tests pass in CI. UptimeRobot monitoring active. Sentry receiving test errors.

### Phase 10 — Full Training Run (Day 22+)
24-hour training run on Colab + Kaggle.

**Monitor every 2 hours:**
- `avg_move_rating` → should trend toward 3.5+ by hour 16
- `strategy_bonus` (negative value in loss) → should become less negative over time (model diverging from guide = good)
- `value_loss` → should decrease; if plateauing before hour 8, halve LR
- `agreement_weights` → some rules should strengthen, others decay
- Emergent strategies detected → check after 5000+ games

---

## 31. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| **Strategy loss type** | Agreement reward (never violation penalty) | Penalties create ceilings. Rewards create floors with no ceiling. Model can transcend any encoded rule. |
| **Strategy vector dimensions** | 16 (vs 12 in Doc 2) | Counter confidence + deviation signal add genuine information. Extra 4 dims are negligible compute cost. |
| **Agreement weight learning** | Per-rule weights updated each game + slow decay | Rules that reliably predict wins strengthen; rules the model consistently overrides decay toward zero. The guide adapts to the model. |
| **Emergent strategy detection** | Fingerprint clustering every 500 games | Self-play invents strategies; we must detect, name, and build counters without human involvement. |
| **Path attention scale** | Learned parameter (Doc 2 approach) | Network can reduce it toward zero if path attention isn't useful — more flexible than fixed init. |
| **Transposition Tables** | Zobrist hash, per-game TTs (both variants) | +40–50% effective simulations per time budget. Benefit largest in early training when many paths converge to bad positions. |
| **Time-based MCTS** | Budget in ms, not fixed sim count | Correct: early-game positions (136 moves) take longer per sim than late-game (8 moves). Fixed count wastes budget in endgame and risks overrun in opening. |
| **MTD(f) Endgame** | Hard trigger ≤4 walls, soft trigger ≤6 walls + path≤5 | Exact play where MCTS is weakest. Eliminates the most common late-game blunder class entirely. |
| **Deviation logging** | Track profitable strategy disagreements | These are the most valuable training signals: they show the model has found something better than the encoded guide. |
| **Backend hosting** | Render free tier + UptimeRobot pings | Render's spin-down is mitigated by the uptime pinger. Zero-cost. |
| **Model storage** | Hugging Face Hub | Free, versioned, accessible from Colab and Render. The right tool for ML artifact storage. |
| **Frontend hosting** | Cloudflare Pages | Better than GitHub Pages: unlimited bandwidth, global CDN, automatic deployments. |
| **Database** | Supabase (PostgreSQL) | Free 500MB, includes auth, Row Level Security, and a dashboard. Postgres enables complex queries for leaderboard and game analysis. |
| **Auth** | Supabase Auth (optional) | Anonymous play is supported. Auth only required for leaderboard. Supabase handles the complexity for free. |
| **No pretrained models** | Built from scratch | As required. Agreement bonus during early training substitutes for transfer learning. |

---

*End of Final Integrated Technical Plan.*
*Strategy is advice. The model earns the right to disagree. The system earns the right to stay free.*
