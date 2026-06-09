# Quoridor AI: Final Technical Plan
### Strategy-Informed Neural Network (SINN) with Guided Search
### *Merged & Updated — Includes Transposition Tables, MTD(f) Endgame Solver, Agreement-Based Strategy Loss, and Emergent Strategy Detection*

---

## Table of Contents
1. [Executive Summary](#1-executive-summary)
2. [Existing Bots: Landscape Analysis](#2-existing-bots-landscape-analysis)
3. [Our Approach vs Existing Approaches](#3-our-approach-vs-existing-approaches)
4. [File Structure](#4-file-structure)
5. [Core Components: Detailed Logic](#5-core-components-detailed-logic)
   - 5.1 Game Engine (+ Zobrist Hashing)
   - 5.2 Strategy Guide — Agreement-Based, Not Violation-Based
   - 5.3 Neural Network Architecture
   - 5.4 Guided Search (MCTS + TT + Time Budget + MTD(f) Endgame Solver)
   - 5.5 Rating System
   - 5.6 Loss Function
   - 5.7 Post-Game Backward Analysis
   - 5.8 Self-Play Training Pipeline
   - 5.9 Emergent Strategy Detection
6. [Data Structures & Representations](#6-data-structures--representations)
7. [Key Algorithms (Pseudocode)](#7-key-algorithms-pseudocode)
8. [Training Configuration](#8-training-configuration)
9. [Data Flow Diagram](#9-data-flow-diagram)
10. [Implementation Priorities & Phases](#10-implementation-priorities--phases)
11. [Design Decisions Summary](#11-design-decisions-summary)

---

## 1. Executive Summary

This document describes a custom Quoridor AI built entirely from scratch. The core engine is a **Strategy-Informed Neural Network (SINN)** — an architecture inspired by Physics-Informed Neural Networks (PINNs), where "physics" is replaced by formalized Quoridor strategy knowledge. The model uses a path-guided attention mechanism (BFS distance maps as the "guide") and a strategy layer that *encourages* known-good moves without *punishing* novel ones.

### The Central Design Philosophy

> **Human strategy is advice. Not law.**

This is the defining departure from naive PINN analogies. In physics, constraints are absolute — you cannot violate conservation of energy. In strategy, constraints are heuristics derived from human experience. They are an excellent starting point, but they represent a *ceiling* if enforced as hard rules.

The SINN therefore uses an **agreement-based** strategy signal:

```
Old thinking:  violate strategy → penalty      (strategy as law)
New thinking:  follow strategy  → bonus reward  (strategy as advice)
               deviate from strategy → zero bonus, never a penalty
```

This means:
- The model is *guided toward* known-good strategy early in training (agreement bonus is high)
- The model is *never punished* for discovering something better (no negative gradient from disagreement)
- When the model consistently deviates from a rule and wins, the rule's weight decays — the model has learned to transcend it
- The system actively watches for *emergent strategies* (novel recurring win patterns) and builds counters to them

### Additional Architectural Features

- **Transposition Tables (Zobrist hashing)**: MCTS becomes a DAG — the same board state reached via different move orders pools statistics, boosting effective simulations by ~40–50%
- **Time-based MCTS**: budget in milliseconds instead of fixed simulation count — consistent quality across all game phases
- **MTD(f) Endgame Solver**: activated when walls are nearly exhausted; finds mathematically forced moves in <20ms, eliminating endgame blunders
- **0–5 continuous move rating**: richer training signal than binary win/loss
- **Post-game backward analysis**: identifies the root cause of defeat, not just symptoms
- **Full counter-strategy loop**: detect opponent strategy → look up or learn a counter → apply it → record effectiveness

---

## 2. Existing Bots: Landscape Analysis

### 2.1 Classical Approaches

| Bot / Method | Core Algorithm | Evaluation Function | Weakness |
|---|---|---|---|
| **Minimax** | Depth-limited tree search | Handcrafted: `path_diff + wall_count` | Exponential branching (~130 moves/turn); shallow depth only |
| **Alpha-Beta Pruning** | Minimax + pruning | Same as above | Still too slow beyond depth 4–5 |
| **MTD(f)** | Iterative-deepening minimax | Same | Hard to tune; brittle endgame |
| **MCTS (pure)** | Random rollout + UCB1 | Terminal result from rollout | Poor signal in early game; needs many simulations |
| **MCTS + Heuristics** | UCB1 with path-heuristic | BFS path differential | Better but no learning |

### 2.2 Learning-Based Approaches

| Approach | Model | What It Learns | Problem |
|---|---|---|---|
| **Q-Learning / DQN** | MLP / small CNN | Q(state, action) | State space too large without good representation |
| **AlphaZero-style** | ResNet Policy+Value | π(a|s), V(s) | No domain knowledge; slow convergence |
| **REINFORCE** | Small MLP | Policy gradient | High variance; ignores strategic structure |

### 2.3 What ALL Existing Bots Are Missing

- Strategy-level *guidance* baked into the loss function (they learn strategy implicitly, or not at all)
- Move quality tracking (they optimize only win/loss)
- Opponent strategy detection and adaptive counter-planning
- Structured post-game analysis feeding back into training
- Path-guided attention as a first-class architectural component
- A mechanism to detect and respond to *novel* opponent strategies

### 2.4 Shared Infrastructure (Everyone Uses)

- BFS/Dijkstra for path length computation — universal; we keep it
- Action space: ~136 possible actions per turn (128 wall placements + ≤8 pawn moves)
- Board as tensor channels

---

## 3. Our Approach vs Existing Approaches

| Feature | AlphaZero-style | Minimax+Heuristic | **Our SINN** |
|---|---|---|---|
| Training signal | Win/loss only | Handcrafted eval | Win/loss + move rating + strategy agreement bonus |
| Strategy knowledge | Implicit (learned) | Hardcoded eval terms | Explicit advice + learned refinement; no hard constraints |
| Strategy ceiling | Implicit ceiling | Hard ceiling | **No ceiling** — model can transcend encoded strategy |
| Path guidance | Via value network | Via eval function | Dedicated path-attention module (the "guide") |
| Opponent modeling | None / self-play proxy | Minimax assumes optimal | Strategy detection + counter-planning + emergent pattern recognition |
| Post-game analysis | None | None | Backward error attribution per move |
| Move quality metric | Implied by MCTS visits | None | 0–5 continuous rating per move |
| Strategy loss | None | None | Agreement bonus (reward agreement; ignore disagreement) |
| Emergent strategies | None | None | Detected, tracked, and countered automatically |
| Endgame | MCTS (probabilistic) | Alpha-beta (exact but slow) | MTD(f) solver with TT (exact, fast) |
| Transposition handling | Sometimes | Rarely | Zobrist + DAG MCTS + Solver TT |

---

## 4. File Structure

```
quoridor_ai/
│
├── game_engine/
│   ├── __init__.py
│   ├── board.py                        # Board state class, tensor conversion
│   ├── moves.py                        # Move generation, validation, application
│   ├── pathfinder.py                   # BFS path calculator, distance maps
│   ├── rules.py                        # Wall legality, anti-blockade check (DFS)
│   ├── game.py                         # Full game loop, turn management, termination
│   └── zobrist_hash.py                 # [NEW] Zobrist hashing for O(1) incremental board hashing
│
├── strategy/
│   ├── __init__.py
│   ├── strategy_guide.py               # Strategy rules encoded as agreement functions (not violations)
│   ├── opening_recognizer.py           # Classifies game phase & opening type (known + novel)
│   ├── agreement_engine.py             # Computes agreement scores (replaces constraint_engine)
│   ├── strategy_tracker.py             # Tracks W/L/rating + counter effectiveness per strategy
│   ├── exception_logger.py             # Logs cases where strategy agreement was high but outcome poor
│   ├── counter_planner.py              # Plans counters to identified strategies (known + emergent)
│   └── emergent_strategy_detector.py   # [NEW] Detects novel win patterns; registers new strategies
│
├── models/
│   ├── __init__.py
│   ├── board_encoder.py                # Input tensor construction (10 × 9 × 9)
│   ├── residual_block.py               # Standard ResBlock with optional path attention
│   ├── path_attention.py               # Path-guided attention module (the "guide")
│   ├── strategy_layer.py               # Strategy embedding injection
│   ├── policy_head.py                  # → 136-dim action probability vector
│   ├── value_head.py                   # → scalar ∈ [-1, +1]
│   ├── rating_head.py                  # → scalar ∈ [0, 5]
│   └── quoridor_net.py                 # Master network: composes all submodules
│
├── search/
│   ├── __init__.py
│   ├── zobrist_hash.py                 # (imported from game_engine — listed here for clarity)
│   ├── transposition_table.py          # [NEW] Two TT variants: MCTS-TT (DAG) and Solver-TT
│   ├── mcts.py                         # [MODIFIED] DAG-based MCTS with time budget + agreement UCB
│   ├── endgame_solver.py               # [NEW] MTD(f) + alpha-beta + Solver-TT
│   ├── opponent_predictor.py           # 1-step opponent move prediction
│   └── move_selector.py                # [MODIFIED] Phase switching + time-based stopping
│
├── training/
│   ├── __init__.py
│   ├── self_play_worker.py             # Single self-play game instance
│   ├── self_play_manager.py            # Orchestrates N parallel workers
│   ├── experience_buffer.py            # Prioritized replay buffer
│   ├── loss_functions.py               # [MODIFIED] Agreement-based strategy reward
│   ├── trainer.py                      # Training loop, optimizer, scheduler
│   └── evaluator.py                    # Periodic eval: win rate, avg rating, etc.
│
├── analysis/
│   ├── __init__.py
│   ├── game_recorder.py                # Records all moves + metadata per game
│   ├── move_rater.py                   # Calculates 0–5 rating for each move
│   ├── backward_analyzer.py            # Post-game: last-to-first error attribution
│   ├── strategy_updater.py             # [MODIFIED] Tracks deviations that worked + updates weights
│   └── deviation_logger.py             # [NEW] Logs moves that deviated from strategy advice and won
│
├── data/
│   ├── games/                          # JSON game records (compressed)
│   ├── checkpoints/                    # Model weight snapshots
│   ├── strategy_stats.json             # Live strategy effectiveness stats
│   ├── agreement_weights.json          # Per-rule agreement bonus weights (learned, not fixed)
│   ├── emergent_strategies.json        # [NEW] Discovered novel strategies + counters
│   ├── deviation_log.json              # [NEW] Profitable deviations from strategy advice
│   ├── exception_log.json              # Cases where agreement was high but outcome poor
│   └── ratings_history.json            # Move rating distributions over training
│
├── config.py                           # All hyperparameters, paths, flags
├── train.py                            # Entry point: launches self-play training
├── play.py                             # Entry point: human vs bot
└── evaluate.py                         # Entry point: bot vs bot evaluation
```

---

## 5. Core Components: Detailed Logic

---

### 5.1 Game Engine

**`board.py` — Board State**

```python
@dataclass
class BoardState:
    pawn_positions:   dict   # {0: (row, col), 1: (row, col)}
    h_walls:          set    # Horizontal walls: set of (row, col)
    v_walls:          set    # Vertical walls: set of (row, col)
    walls_remaining:  dict   # {0: int, 1: int}; default 10 each
    current_player:   int    # 0 or 1
    move_history:     list   # list[Move]
    turn:             int
    current_hash:     int    # [NEW] Zobrist hash, maintained incrementally
```

**Board as Tensor** (for network input):

```
Channel 0:  Own pawn binary map
Channel 1:  Opponent pawn binary map
Channel 2:  Horizontal wall map
Channel 3:  Vertical wall map
Channel 4:  Own BFS distance map          (normalized: distance[i][j] / max_distance)
Channel 5:  Opponent BFS distance map
Channel 6:  Own walls remaining           (scalar broadcast: walls/10)
Channel 7:  Opponent walls remaining
Channel 8:  Game phase                    (0=opening, 0.5=mid, 1=end, broadcast)
Channel 9:  Current player                (0 or 1, broadcast)
```

Board is always presented from the current player's perspective (flipped so "my goal" = row 8).

**`pathfinder.py`**

```python
def bfs_distance_map(board, player) -> np.ndarray:
    """9×9 array: min moves from pawn to each cell, respecting walls."""

def shortest_path_to_goal(board, player) -> int:
    """Min moves to reach goal row."""

def path_differential(board) -> int:
    """opponent_path - own_path. Positive = we are winning the race."""
```

**`rules.py` — Wall Legality**

Three checks in order:
1. Wall within 8×8 placement grid
2. No overlap with existing walls (4 overlap cases per orientation)
3. Anti-blockade: BFS for both players confirms paths still exist after placement

---

### 5.2 Strategy Guide — Agreement-Based, Not Violation-Based

> **This section embodies the core philosophical departure from the original PINN framing.**

#### The Principle

In physics-informed neural networks, the physical constraints are absolute — `F = ma` is always true. You penalize violation because violation is impossible in reality.

In Quoridor strategy, rules like "advance when behind in path race" are *heuristics* — useful generalizations derived from human experience that hold most of the time. Treating them as absolute laws:

1. Creates a **ceiling**: the model can never learn that rule X is suboptimal in context Y
2. Produces a **conflict**: the outcome signal (win/loss) may reward the very move the strategy signal penalizes
3. Slows convergence in the mid-to-late training regime

The solution: **strategy rules generate agreement bonuses, not violation penalties**.

```
Old: move violates rule → gradient pushes AWAY from the move
New: move agrees with rule → gradient pulls TOWARD the move
     move disagrees with rule → gradient is ZERO (the model ignores the rule freely)
```

This means early training still benefits from strategic guidance (agreement bonuses pull the model toward known-good moves). But as the model matures, it can freely deviate from any rule with zero cost — and when it consistently deviates and wins, the rule's weight is reduced automatically.

---

#### `strategy/strategy_guide.py` — Rules as Agreement Functions

```python
@dataclass
class StrategyRule:
    name:        str
    phase:       GamePhase                           # Opening | Mid | End | All
    condition:   Callable[[BoardState], bool]        # When this rule is relevant
    agreement:   Callable[[BoardState, Move], float] # 0.0 = no alignment; 1.0 = full alignment
                                                     # NEVER returns negative values
    weight:      float = 1.0                         # Learned over training; can decay to zero


STRATEGY_RULES = [

    StrategyRule(
        name="wall_efficiency",
        phase=GamePhase.ALL,
        condition=lambda b: True,
        agreement=lambda b, a: (
            0.0 if not is_wall(a) else
            min(1.0, path_extension_from_wall(b, a) / 2.0)
            # Full agreement if wall extends opponent path by 2+
            # Partial agreement for 1-move extension
            # Zero agreement for zero-extension wall (not a penalty, just no bonus)
        ),
    ),

    StrategyRule(
        name="path_race_advance",
        phase=GamePhase.ALL,
        condition=lambda b: path_differential(b) < 0,  # We're losing the race
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and reduces_own_path(b, a) else
            0.5 if is_wall(a) and path_extension_from_wall(b, a) >= 2 else
            0.0   # No bonus for advancing sideways or placing weak walls when behind
                  # But ALSO no penalty — the model may have found something better
        ),
    ),

    StrategyRule(
        name="standard_opening_center",
        phase=GamePhase.OPENING,
        condition=lambda b: b.turn < 6 and detected_strategy(b) == "Standard",
        agreement=lambda b, a: (
            1.0 if is_pawn_move(a) and is_central_column(a) else
            0.3 if is_pawn_move(a) and is_near_central(a) else
            0.0
        ),
    ),

    StrategyRule(
        name="anti_rush_counter_wall",
        phase=GamePhase.OPENING,
        condition=lambda b: opponent_strategy(b) == "Rush",
        agreement=lambda b, a: (
            1.0 if is_wall(a) and blocks_opponent_advance(b, a) else
            0.3 if is_wall(a) else
            0.0   # Pawn move against Rush = zero bonus, not a penalty
                  # Model may discover pawn racing is better in some Rush matchups
        ),
    ),

    StrategyRule(
        name="protective_wall_when_ahead",
        phase=GamePhase.MID,
        condition=lambda b: path_differential(b) > 1,  # We're winning the race
        agreement=lambda b, a: (
            0.8 if is_wall(a) and placed_behind_own_pawn(b, a) else
            0.4 if is_pawn_move(a) and reduces_own_path(b, a) else
            0.0
        ),
    ),

    StrategyRule(
        name="jump_tempo",
        phase=GamePhase.ALL,
        condition=lambda b: adjacent_to_opponent(b),
        agreement=lambda b, a: (
            0.9 if is_jump_move(a) and reduces_own_path(b, a) else
            0.0
        ),
    ),

    # ... additional rules from full strategy document
]
```

**Key property**: every `agreement` function returns a value in `[0.0, 1.0]`. There is no floor below zero. The model is *never told it did something wrong* by the strategy system — only told when it did something the strategy system recognizes as good.

---

#### `strategy/agreement_engine.py` — Compute Agreement Scores

```python
def compute_strategy_agreements(board: BoardState,
                                 action: Move,
                                 rules: list = STRATEGY_RULES) -> list[tuple[str, float]]:
    """
    Returns a list of (rule_name, agreement_score) pairs for all applicable rules.
    
    Agreement score ∈ [0, 1]. Never negative.
    A score of 0 does NOT mean the move is bad — it means the rule doesn't apply
    or the move doesn't match what the rule recommends. No penalty is incurred.
    
    Contrast with old 'compute_strategy_violations': that returned values where
    high score = punishment. This returns values where high score = reward.
    The gradient signal is fundamentally asymmetric:
        agreement > 0  →  small pull toward this move
        agreement = 0  →  no gradient from strategy system
        (there is no agreement < 0 path)
    """
    agreements = []
    for rule in rules:
        if rule.condition(board):
            score = rule.agreement(board, action)
            # score is guaranteed ≥ 0 by the rule's contract
            agreements.append((rule.name, score))
    return agreements


def total_strategy_bonus(agreements: list[tuple[str, float]],
                          weights: dict[str, float]) -> float:
    """
    Returns a non-negative bonus value.
    High bonus = move is well-aligned with multiple strategy rules.
    Zero bonus = move is not aligned with any rule (no penalty).
    
    weights[rule_name] is learned over training:
        - Rule consistently leads to winning moves → weight increases
        - Rule leads to high-agreement but losing moves → weight decreases
        - Rule is consistently overridden by model and model still wins → weight decays
    """
    return sum(weights.get(name, 0.1) * score for name, score in agreements)
```

---

#### `strategy/opening_recognizer.py` — Strategy Detection (Known + Novel)

Identifies opponent's opening strategy by turn 4. Now also flags patterns that don't match known strategies.

```python
def recognize_strategy(move_history, board) -> tuple[StrategyLabel, float]:
    """
    Returns: (StrategyLabel, confidence)
    StrategyLabel: Standard | Rush | Shiller | Sidewall | Novel | Unknown
    
    Novel = recurring pattern in self-play data that doesn't match a known label
            but has an associated ID in emergent_strategies.json.
    """
    features = {
        "advance_rate":            compute_advance_rate(move_history),
        "wall_rate":               compute_wall_rate(move_history),
        "first_pawn_direction":    get_first_pawn_move_direction(move_history),
        "wall_behind_pawn":        check_wall_behind_pawn(board, move_history),
        "lateral_first_move":      is_lateral_first_move(move_history),
        "tunnel_pattern":          detect_tunnel_walls(board, move_history),
        "central_column_pref":     compute_column_preference(move_history),
        "move_fingerprint":        compute_move_fingerprint(move_history),  # [NEW]
    }

    # Rule-based classification for known strategies
    if features["wall_rate"] == 0 and features["advance_rate"] > 0.8:
        return StrategyLabel.RUSH, 0.9

    if features["lateral_first_move"] and features["tunnel_pattern"]:
        return StrategyLabel.SHILLER, 0.85

    if features["wall_rate"] > 0 and features["wall_behind_pawn"] and features["advance_rate"] > 0.5:
        return StrategyLabel.STANDARD, 0.8

    if features["wall_rate"] > 0 and not features["wall_behind_pawn"] and features["advance_rate"] < 0.4:
        return StrategyLabel.SIDEWALL, 0.7

    # [NEW] Check against emergent strategies database
    emergent_match = EmergentStrategyDetector.match_fingerprint(features["move_fingerprint"])
    if emergent_match:
        return StrategyLabel.NOVEL(emergent_match.id), emergent_match.confidence

    return StrategyLabel.UNKNOWN, 0.3
```

---

#### `strategy/strategy_tracker.py` — Effectiveness Database

```
strategy_stats.json:
{
    "Standard": {
        "wins": int,
        "losses": int,
        "total_games": int,
        "avg_move_rating": float,
        "known_counters": {
            "AntiRushWall": {"wins": int, "losses": int, "avg_turns_to_advantage": float},
            ...
        },
        "exceptions": [game_ids where high agreement moves led to losses]
    },
    "Rush":     { ... },
    "Shiller":  { ... },
    "Sidewall": { ... },

    # [NEW] Emergent strategies get the same structure as known ones,
    # populated over time from self-play data:
    "Emergent_001": {
        "description": "auto-generated: early lateral wall + delayed advance",
        "first_seen_game": "game_id",
        "wins": int,
        "losses": int,
        "known_counters": { ... },
        "rule_agreement_history": []   # Was strategy advice relevant? Mostly ignored?
    }
}
```

---

#### `strategy/counter_planner.py` — Counter-Strategy Planning

The full counter-strategy loop, now supporting emergent strategies:

```python
def plan_counter(game_record, losing_player, strategy_tracker) -> list[CounterRecord]:
    """
    After a loss:
    1. Identify what strategy the winner used (known or emergent)
    2. Look up effective counters from strategy_tracker
    3. Find what the model should have done (highest missed counter value)
    4. Inject as high-priority replay experiences
    
    For EMERGENT strategies with no known counter yet:
    5. Log the game for EmergentStrategyDetector analysis
    6. Fall back to base path-race strategy as default counter
    7. Record outcome → over time, effective counters emerge from the data
    """
    opp_strategy = game_record.detected_strategies[1 - losing_player]
    known_counters = strategy_tracker.get_counters(opp_strategy)

    if not known_counters and is_novel_strategy(opp_strategy):
        # No counter known yet — log and learn
        EmergentStrategyDetector.log_game(game_record, opp_strategy)
        return []  # Nothing to inject yet; system will learn over games

    missed_counters = []
    for turn, (state, action_taken) in enumerate(game_record.moves_for(losing_player)):
        for counter_action in known_counters.recommended_moves(state):
            if counter_action != action_taken:
                expected_gain = (evaluate_move(state, counter_action)
                                 - evaluate_move(state, action_taken))
                if expected_gain > 0.5:
                    missed_counters.append(CounterRecord(
                        turn=turn,
                        counter_action=counter_action,
                        expected_gain=expected_gain,
                        opponent_strategy=opp_strategy,
                    ))

    return missed_counters
```

**During a live game** (the use-counter part of the loop):

```python
def get_in_game_counter_recommendation(board, strategy_tracker) -> ActionType | None:
    """
    Called during move selection. If opponent strategy is recognized,
    look up the highest-win-rate counter approach and add it to strategy_vector.
    This is returned as a RECOMMENDATION — the model may ignore it freely.
    """
    opp_strategy = opening_recognizer.current_opponent_strategy
    if opp_strategy == StrategyLabel.UNKNOWN:
        return None

    best_counter = strategy_tracker.get_best_counter(opp_strategy)
    if best_counter and best_counter.win_rate > 0.5:
        return best_counter.action_type
    return None
```

---

#### `strategy/agreement_weight_updater.py` — Rule Weight Learning

The agreement bonus weights for each rule are not fixed — they are learned over training:

```python
def update_rule_weights(game_record, outcome, agreement_weights):
    """
    For each move in the game:
    - If agreement was high AND the outcome was good: increase rule weight slightly
    - If agreement was high AND the outcome was bad: decrease rule weight slightly
    - If agreement was zero (model deviated): record the deviation, don't touch the weight
    
    Additionally: if a rule is CONSISTENTLY being ignored by the model (agreement
    never fires because the model has found better moves in those positions),
    decay the rule's weight toward zero over time. The rule has become obsolete.
    """
    for turn_data in game_record.turn_data:
        for rule_name, agreement_score in turn_data.agreements:
            if agreement_score > 0.5:   # The model followed this rule's advice
                delta = 0.01 * (outcome - 0.5) * agreement_score
                agreement_weights[rule_name] = clip(
                    agreement_weights[rule_name] + delta, 0.0, 2.0
                )

    # Decay weights for rules rarely agreed with (model has moved past them)
    for rule_name in agreement_weights:
        avg_agreement = game_record.avg_agreement_for_rule(rule_name)
        if avg_agreement < 0.05:   # Consistently not following this advice
            agreement_weights[rule_name] *= 0.999  # Very slow decay
```

---

### 5.3 Neural Network Architecture

**Strategy Vector (16-dimensional)** — expanded from original 12:

```
[0:5]   Opponent strategy one-hot:
            [Standard, Rush, Shiller, Sidewall, Novel/Unknown]
[5]     Path differential (normalized): (opp_path - own_path) / 9
[6]     Wall efficiency: avg(path_ext_per_wall) from last 3 walls
[7:10]  Game phase one-hot: [Opening (turns 0-10), Mid (11-30), End (31+)]
[10:13] Recommended counter action type: [Advance, Place_wall, Jump]
        (from counter_planner.get_in_game_counter_recommendation)
[13]    Counter confidence: win_rate of best known counter ∈ [0, 1]
[14]    Strategy agreement of the last move taken: ∈ [0, 1]
        (feedback signal: did the model follow advice on the last turn?)
[15]    Novel strategy similarity to nearest known strategy: ∈ [0, 1]
```

The counter confidence at [13] is important: it tells the network *how much to weight the counter recommendation*. A counter with 80% win rate should be followed more strongly than one with 52% win rate.

**`quoridor_net.py` — Master Architecture (QuoridorSINN)**

```
INPUT:  Tensor [10 × 9 × 9]   (10 channels)
        Strategy Vector [16]   (see above)

STEM:   Conv(10→128, kernel=3, padding=1) → BatchNorm → ReLU
        Output: [128 × 9 × 9]

BLOCK GROUP 1 (5 Guided Residual Blocks):
        ResBlock(128→128) + PathAttentionGate after each block
        PathAttentionGate uses channels 4 & 5 (BFS maps) as soft attention

STRATEGY INJECTION:
        StrategyEmbedding(16 → 9×9×16) → reshaped to [16 × 9 × 9]
        Concatenate: [128+16 = 144 × 9 × 9]
        PointwiseConv(144→128) → BatchNorm → ReLU
        Output: [128 × 9 × 9]

BLOCK GROUP 2 (5 Guided Residual Blocks)
BLOCK GROUP 3 (5 Guided Residual Blocks)

TOTAL: 15 Residual Blocks

FLATTEN:
        Global Average Pool → [128]

THREE HEADS (parallel):

    POLICY HEAD:
        Linear(128→256) → ReLU → Linear(256→136)
        Softmax (after masking invalid actions to -inf)
        Output: probability distribution over valid actions [136]

    VALUE HEAD:
        Linear(128→64) → ReLU → Linear(64→1) → Tanh
        Output: scalar ∈ [-1, +1]

    RATING HEAD:
        Concat [128-dim, 8-dim move_context_vector]
        Linear(136→64) → ReLU → Linear(64→1) → Sigmoid × 5
        Output: scalar ∈ [0, 5]
```

**`path_attention.py` — The Guide**

```python
class PathAttentionGate(nn.Module):
    """
    BFS distance maps gate feature maps spatially.
    Cells where our path advantage is greatest receive higher attention.
    Residual connection ensures no cell is completely ignored.
    """
    def forward(self, features, path_map_own, path_map_opp):
        path_diff_map = path_map_opp - path_map_own       # [9 × 9]
        attention = torch.softmax(path_diff_map.flatten(), dim=0).reshape(9, 9)
        attention = attention.unsqueeze(0).unsqueeze(0)    # [1 × 1 × 9 × 9]
        gated = features * (1 + self.attention_scale * attention)
        return gated + features    # Residual: never zero out any position
```

**Total Parameters**: ~3.2M — trainable on a single consumer GPU.

---

### 5.4 Guided Search

#### `game_engine/zobrist_hash.py`

Generates a unique 64-bit hash for any board state. The hash is maintained *incrementally* — only the changed components are XOR'd in/out on each move, making hash updates O(1).

```python
class ZobristHasher:
    def __init__(self):
        rng = np.random.default_rng(seed=42)
        # Pawn positions: 2 players × 9 × 9 cells
        self.pawn_table  = rng.integers(1, 2**64-1, size=(2, 9, 9), dtype=np.uint64)
        # Horizontal walls: 8×8 placement grid
        self.hwall_table = rng.integers(1, 2**64-1, size=(8, 8),    dtype=np.uint64)
        # Vertical walls
        self.vwall_table = rng.integers(1, 2**64-1, size=(8, 8),    dtype=np.uint64)
        # Walls remaining: 2 players × 11 values (0–10)
        self.walls_table = rng.integers(1, 2**64-1, size=(2, 11),   dtype=np.uint64)
        # Current player
        self.player_table = rng.integers(1, 2**64-1, size=(2,),     dtype=np.uint64)

    def full_hash(self, board) -> int:
        """Compute from scratch once at game initialization."""
        h = np.uint64(0)
        r0, c0 = board.pawn_positions[0]
        r1, c1 = board.pawn_positions[1]
        h ^= self.pawn_table[0, r0, c0]
        h ^= self.pawn_table[1, r1, c1]
        for (r, c) in board.h_walls: h ^= self.hwall_table[r, c]
        for (r, c) in board.v_walls: h ^= self.vwall_table[r, c]
        h ^= self.walls_table[0, board.walls_remaining[0]]
        h ^= self.walls_table[1, board.walls_remaining[1]]
        h ^= self.player_table[board.current_player]
        return int(h)

    def incremental_update(self, current_hash, board_before, move) -> int:
        """O(1) update. Called inside apply_move() — zero overhead at runtime."""
        h = np.uint64(current_hash)
        if isinstance(move, PawnMove):
            p = board_before.current_player
            r_old, c_old = board_before.pawn_positions[p]
            h ^= self.pawn_table[p, r_old, c_old]
            h ^= self.pawn_table[p, move.to_row, move.to_col]
        elif isinstance(move, WallMove):
            p = board_before.current_player
            w_old = board_before.walls_remaining[p]
            h ^= self.walls_table[p, w_old]
            h ^= self.walls_table[p, w_old - 1]
            if move.orientation == 'h':
                h ^= self.hwall_table[move.row, move.col]
            else:
                h ^= self.vwall_table[move.row, move.col]
        h ^= self.player_table[board_before.current_player]
        h ^= self.player_table[1 - board_before.current_player]
        return int(h)
```

**Why Quoridor benefits**: Wall placement order is often interchangeable — Wall A at turn 5 then Wall B at turn 7 reaches the same board as Wall B at turn 5 then Wall A at turn 7. Without a TT, MCTS treats these as separate nodes. With a TT, they collapse into one shared node with pooled Q and N statistics.

---

#### `search/transposition_table.py` — Two TT Variants

**MCTS Transposition Table** — converts MCTS tree into a DAG:

```python
class MCTSTranspositionTable:
    """
    Global store for MCTS nodes keyed by Zobrist hash.
    Same board state reached via different move orders → same node, pooled Q and N.
    """
    def __init__(self, max_size=500_000):
        self.table = {}
        self.max_size = max_size

    def get_or_create(self, board_hash, board_state) -> tuple[MCTSNode, bool]:
        """
        Returns (node, was_existing).
        If was_existing=True: node already has Q/N from a different path — inherit.
        If was_existing=False: new node, initialize prior from policy network.
        """
        existing = self.table.get(board_hash)
        if existing:
            return existing, True
        new_node = MCTSNode(board_state)
        self.table[board_hash] = new_node
        return new_node, False
```

**Solver Transposition Table** — stores alpha-beta bounds:

```python
class BoundType(Enum):
    EXACT = 0    # Stored value is exact
    LOWER = 1    # True value ≥ stored (beta cutoff)
    UPPER = 2    # True value ≤ stored (alpha cutoff)

class SolverTranspositionTable:
    def probe(self, board_hash, alpha, beta, depth):
        """Returns (value, best_action) if TT causes a cutoff, else (None, None)."""
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

---

#### `search/mcts.py` — DAG-Based MCTS with Agreement-Guided UCB

**UCB Formula (Updated)**:

```
UCB_strategy(s, a) = Q(s, a)
                   + c_puct * P(s, a) / (1 + N(s, a))
                   + λ_guide * path_guide_bonus(s, a)
                   + λ_strat * agreement_bonus(s, a)

where:
    path_guide_bonus(s, a) = delta_path_differential(s, a) / 9   [−1, +1]
    agreement_bonus(s, a)  = total_strategy_bonus(agreements)    [0, +∞)

Critical property: agreement_bonus ≥ 0 always.
Moves the strategy doesn't recognize get bonus = 0.
They are NOT penalized with a negative UCB contribution.
The model explores freely; strategy guidance just tips the scales slightly toward known-good moves.
```

**Node Structure (DAG)**:

```python
class MCTSNode:
    # Children store board hashes, not node references
    children: dict[int, int]    # action_idx -> child_board_hash
    Q: float = 0.0
    N: int = 0
    W: float = 0.0
    prior: float = 0.0
    # TT resolves hash -> node on demand
```

**Expansion with TT**:

```python
def expand(node, board, network, agreement_engine, tt):
    policy, _, _ = network(board_to_tensor(board))
    for action in get_valid_actions(board):
        child_board = apply_move(board, action)
        child_hash  = child_board.current_hash
        child_node, was_existing = tt.get_or_create(child_hash, child_board)
        if not was_existing:
            child_node.prior = policy[action].item()
        # was_existing: inherit pooled Q/N — no reset
        node.children[action] = child_hash
```

**Time-Based Search** (replaces fixed simulation count):

```python
def search(self, root_board, network, agreement_engine, mcts_tt, time_budget_ms):
    """
    Runs simulations for time_budget_ms milliseconds.
    Consistent quality across all game phases:
      - Early game: many valid actions → each sim is slower → fewer sims, same time
      - Late game: few valid actions → each sim is faster → more sims, same time
    """
    start = time.monotonic_ns()
    while (time.monotonic_ns() - start) / 1e6 < time_budget_ms:
        self._run_simulation(root_board, network, agreement_engine)
    return self.root.visit_count_distribution()
```

**TT Lifecycle** (per game, not per move):

```python
# In self_play_worker.py / move_selector.py:
game_mcts_tt    = MCTSTranspositionTable()    # Created once at game start
game_endgame_tt = SolverTranspositionTable()  # Created once at game start

for turn in game:
    move = move_selector.select_move(
        board, network, strategy_guide,
        mcts_tt=game_mcts_tt,
        endgame_tt=game_endgame_tt,
        time_budget_ms=300
    )
    board.apply(move)
    # Both TTs accumulate knowledge across all turns within the game

game_mcts_tt.clear()
game_endgame_tt.clear()
```

---

#### `search/endgame_solver.py` — MTD(f) Exact Solver

Activated when the position is shallow enough for exhaustive search.

```python
class EndgameSolver:
    """
    Exact endgame solver using MTD(f) + alpha-beta + SolverTT.
    Finds mathematically forced moves in <20ms for typical endgame positions.
    Completely replaces MCTS and the neural network in the endgame phase —
    perfect play with no approximation.
    """

    def solve(self, board, solver_tt) -> Move:
        """Entry point. Returns the optimal move."""
        max_depth = self._estimate_max_depth(board)
        best_move = None
        f = 0    # MTD(f) initial estimate

        for depth in range(1, max_depth + 1):
            value, move = self._mtdf(board, f, depth, solver_tt)
            f = value
            best_move = move
            if abs(value) >= WIN_VALUE * 0.9:
                break    # Forced win/loss found

        return best_move

    def _mtdf(self, board, f, depth, tt):
        """MTD(f): null-window alpha-beta calls until convergence."""
        g, upper, lower = f, WIN_VALUE, -WIN_VALUE
        best_move = None
        while lower < upper:
            beta = max(g, lower + 1)
            g, move = self._alpha_beta(board, beta - 1, beta, depth, tt)
            best_move = move or best_move
            if g < beta: upper = g
            else:         lower = g
        return g, best_move

    def _alpha_beta(self, board, alpha, beta, depth, tt):
        """Negamax alpha-beta with TT probe and move ordering."""
        if board.is_terminal():
            return self._terminal_value(board), None
        if depth == 0:
            return self._heuristic_eval(board), None

        alpha_orig = alpha
        tt_value, tt_move = tt.probe(board.current_hash, alpha, beta, depth)
        if tt_value is not None:
            return tt_value, tt_move

        best_value, best_move = -WIN_VALUE, None
        for move in self._ordered_moves(board):
            child = apply_move(board, move)
            v, _ = self._alpha_beta(child, -beta, -alpha, depth - 1, tt)
            v = -v
            if v > best_value:
                best_value, best_move = v, move
            alpha = max(alpha, best_value)
            if alpha >= beta:
                break

        tt.store(board.current_hash, best_value, best_move, alpha_orig, beta, depth)
        return best_value, best_move

    def _ordered_moves(self, board):
        """
        Move ordering for maximum alpha-beta cutoffs:
        1. Winning pawn move (goal row)        → almost always cuts immediately
        2. Pawn advance (reduces own path)
        3. Wall blocking opponent's next move  (extends opp path by 2+)
        4. Lateral pawn move
        5. Other wall placements
        6. Retreat
        """
        ...
```

---

#### `search/move_selector.py` — Phase-Aware Selection

```python
ENDGAME_WALLS_HARD = 4    # Total walls ≤ 4: always MTD(f)
ENDGAME_WALLS_SOFT = 6    # Total walls ≤ 6 AND max_path ≤ 5: MTD(f)
ENDGAME_PATH_SOFT  = 5

def select_move(board, network, strategy_guide, mcts_tt, endgame_tt, time_budget_ms):
    total_walls = sum(board.walls_remaining.values())
    max_path    = max(shortest_path_to_goal(board, 0),
                      shortest_path_to_goal(board, 1))

    use_endgame = (total_walls <= ENDGAME_WALLS_HARD or
                   (total_walls <= ENDGAME_WALLS_SOFT and max_path <= ENDGAME_PATH_SOFT))

    if use_endgame:
        return endgame_solver.solve(board, endgame_tt)

    # Normal phase: MCTS + 1-step lookahead
    counter_rec = counter_planner.get_in_game_counter_recommendation(board, strategy_tracker)
    strat_vector = strategy_guide.compute_strategy_vector(board, counter_recommendation=counter_rec)

    mcts_result = mcts.search(board, network, agreement_engine,
                               mcts_tt=mcts_tt,
                               time_budget_ms=time_budget_ms * 0.85)

    top_k = mcts_result.top_k(k=5)
    evaluated = []
    for move in top_k:
        next_state  = apply_move(board, move)
        opp_move    = opponent_predictor.predict(next_state, network)
        post_state  = apply_move(next_state, opp_move)
        evaluation  = network.value(board_to_tensor(post_state))
        evaluated.append((move, evaluation))

    return max(evaluated, key=lambda x: x[1])[0]
```

---

### 5.5 Rating System (0–5 Continuous Scale)

The 0–5 move rating is computed after the fact (once the opponent has responded), based on the change in path differential after both moves.

**Core Formula**:

```python
def compute_rating(state_before, action, state_after_opp_response):
    path_diff_before = path_differential(state_before)
    path_diff_after  = path_differential(state_after_opp_response)
    delta = path_diff_after - path_diff_before

    # Base rating from delta
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

    # Strategy agreement adds a SMALL bonus (not a penalty for disagreement)
    agreements = agreement_engine.compute_strategy_agreements(state_before, action)
    strat_bonus = min(0.5, 0.5 * total_strategy_bonus(agreements, agreement_weights))
    # Cap at 0.5 so path differential is always the primary signal

    return max(0.0, min(5.0, base + strat_bonus))
```

**Note**: strategy agreement adds at most +0.5 to the rating. Path differential is the primary signal. The model is never rated *lower* because it deviated from strategy advice.

**Blunder Detection**:

```python
def is_true_blunder(state):
    """
    True blunder: even with optimal play from this point,
    the opponent can maintain path advantage indefinitely.
    """
    critical_deficit  = path_differential(state)      # Negative = we're behind
    max_recoverable   = state.walls_remaining[current_player] * 2  # Best case
    return critical_deficit < -max_recoverable
```

---

### 5.6 Loss Function

**The Strategy Loss: Reward Signal, Not Penalty**

```python
def total_loss(batch, model_output, strategy_data, config):
    """
    strategy_data: list of (agreements, bonus_score) per sample
                   — not violations, not penalties
    """

    # 1. POLICY LOSS — cross-entropy vs MCTS visit distribution
    L_policy = -torch.sum(
        batch.policy_targets * torch.log(model_output.policy_pred + 1e-8),
        dim=1
    ).mean()

    # 2. VALUE LOSS — MSE vs game outcome
    L_value = F.mse_loss(model_output.value_pred, batch.value_targets)

    # 3. RATING LOSS — MSE vs actual computed rating
    L_rating = F.mse_loss(model_output.rating_pred, batch.rating_targets)
    avg_rating_penalty = F.relu(3.0 - model_output.rating_pred.mean())

    # 4. STRATEGY LOSS — AGREEMENT REWARD, NOT VIOLATION PENALTY
    #
    #    Old: L_strategy = mean(violation_scores)      → minimized by avoiding violations
    #    New: agreement_bonuses ∈ [0, +∞)              → maximized by following advice
    #         L_strategy = -mean(agreement_bonuses)    → negative loss = reward signal
    #
    #    Gradient behavior:
    #         Old: ∂L/∂action < 0 when violating (pushes AWAY from the move)
    #         New: ∂L/∂action < 0 when agreeing (pulls TOWARD the move)
    #              ∂L/∂action = 0 when not agreeing (gradient is ZERO — no punishment)
    #
    #    This asymmetry is the key: agreement is gently rewarded.
    #    Deviation is completely ignored by the strategy signal.
    #    The outcome signal (L_policy, L_value) is the only force on deviating moves.
    
    agreement_bonuses = torch.tensor(
        [data.bonus_score for data in strategy_data], dtype=torch.float32
    )
    L_strategy = -agreement_bonuses.mean()    # Negative = reward for agreement

    # 5. DEFEAT PENALTY — weight loss samples more heavily
    defeat_mask = (batch.outcomes == -1).float()
    L_defeat = (defeat_mask * F.mse_loss(
        model_output.value_pred, batch.value_targets, reduction='none'
    )).mean()

    total = (
        config.w_policy          * L_policy    +
        config.w_value           * L_value     +
        config.w_rating          * L_rating    +
        config.w_rating_max      * avg_rating_penalty +
        config.w_strategy        * L_strategy  +    # Negative = reward
        config.w_defeat          * L_defeat
    )

    return total, {
        "policy":    L_policy.item(),
        "value":     L_value.item(),
        "rating":    L_rating.item(),
        "strategy":  L_strategy.item(),    # Negative = good (model is following advice)
        "defeat":    L_defeat.item(),
    }
```

**Loss Weight Schedule**:

| Phase | `w_strategy` | `w_value` | Notes |
|---|---|---|---|
| Early (0–5k steps) | 0.5 | 0.8 | Strategy guides early exploration; agreement bonus has meaningful pull |
| Mid (5k–30k) | 0.3 | 1.0 | Model learns to win; strategy reward fades in relative importance |
| Late (30k+) | 0.1 | 1.0 | Strategy serves only as minor regularization; model plays how it wants |

The strategy weight decays gradually because as the model improves, it will naturally follow effective strategy (learning it from outcome signal) and the explicit agreement bonus becomes redundant. Keeping it too high late in training would create the very ceiling we're trying to avoid.

---

### 5.7 Post-Game Backward Analysis

**`backward_analyzer.py`** — runs after every loss, scanning last-to-first:

```python
def backward_analyze(game_record, losing_player, network, strategy_guide):
    """
    Identifies: earliest move where position became unrecoverable.
    Returns: list[MistakeRecord] sorted by severity.
    """
    mistakes = []
    moves = game_record.moves_for(losing_player)

    for i in range(len(moves) - 1, -1, -1):
        state          = game_record.states[i * 2]
        action_taken   = moves[i]
        actual_rating  = game_record.move_ratings[losing_player][i]
        pred_rating    = game_record.predicted_ratings[losing_player][i]
        best_move, _   = find_best_move_in_hindsight(state, network)
        best_rating    = simulate_rating(state, best_move, game_record)
        rating_gap     = pred_rating - actual_rating       # Model was confident but wrong
        missed_gain    = best_rating - actual_rating       # How much better was the best move

        if rating_gap > 1.5 or missed_gain > 2.0:
            agreements = agreement_engine.compute_strategy_agreements(state, action_taken)
            mistakes.append(MistakeRecord(
                turn=i,
                state=state,
                action_taken=action_taken,
                best_action=best_move,
                actual_rating=actual_rating,
                predicted_rating=pred_rating,
                missed_gain=missed_gain,
                agreements=agreements,
                severity=missed_gain + rating_gap,
            ))

        if was_game_decisive_at(game_record, i, losing_player):
            mistakes[-1].is_decisive = True
            break    # Root cause found

    return sorted(mistakes, key=lambda m: -m.severity)
```

**`analysis/deviation_logger.py`** — NEW: tracks profitable deviations:

```python
def log_profitable_deviations(game_record, winner):
    """
    [NEW] After each WIN, scan the winner's moves for cases where:
    - Strategy agreement was zero or near zero (model deviated)
    - The move had high actual rating (the deviation worked)
    
    These are the most valuable data points in the whole system:
    they reveal places where the model has discovered something
    better than the encoded strategy advice.
    
    Logged to deviation_log.json and fed to EmergentStrategyDetector.
    """
    for turn, (state, action) in enumerate(game_record.moves_for(winner)):
        agreements = agreement_engine.compute_strategy_agreements(state, action)
        max_agreement = max(score for _, score in agreements) if agreements else 0.0
        actual_rating = game_record.move_ratings[winner][turn]

        if max_agreement < 0.2 and actual_rating >= 4.0:
            # Low agreement + high rating = profitable deviation
            DeviationLog.record(
                state=state,
                action=action,
                agreement=max_agreement,
                rating=actual_rating,
                game_id=game_record.game_id,
                turn=turn,
            )
```

---

### 5.8 Self-Play Training Pipeline

**`self_play_manager.py` — Parallel Architecture**

```
MainProcess
├── NetworkServer (serves weights to all workers via shared memory)
├── StrategyGuide (singleton — strategy rules + agreement engine)
├── AgreementWeightUpdater (updates per-rule weights after each game)
├── EmergentStrategyDetector (watches for novel patterns)
├── ExperienceBuffer (thread-safe prioritized replay)
├── Trainer (reads from buffer, updates weights)
├── Worker_0 (plays games, writes experiences + deviations)
├── Worker_1
├── Worker_2
└── ... (N = CPU_cores - 2)
```

Each worker per game:
1. Request current network weights
2. Create `game_mcts_tt` and `game_endgame_tt` (per-game TTs)
3. Play one complete game
4. Compute actual ratings for all moves
5. Run backward analysis (if lost)
6. Log profitable deviations (if won)
7. Write experiences to shared buffer
8. Update strategy tracker + agreement weights
9. Clear TTs

**Champion evaluation** (every 1000 games): if current model beats frozen champion at >55%, update champion.

**`experience_buffer.py` — Prioritized Replay**:

```python
class PrioritizedExperienceBuffer:
    """
    Experiences from backward analysis (corrective) have high priority.
    Regular self-play experiences have standard priority.
    Deviation log entries (profitable deviations) have elevated priority.
    """
    PRIORITY_CORRECTIVE  = 5.0   # From backward analysis mistakes
    PRIORITY_DEVIATION   = 4.0   # From profitable deviation log
    PRIORITY_STANDARD    = 1.0   # Regular self-play
```

---

### 5.9 Emergent Strategy Detection

> **New module.** Watches for novel recurring win patterns in self-play data that don't match any known strategy label. When found, registers them in the strategy tracker and eventually builds counters.

**`strategy/emergent_strategy_detector.py`**

```python
class EmergentStrategyDetector:
    """
    Pipeline:
    1. After each game, extract a move sequence "fingerprint" (first K moves)
    2. Every N games, cluster all fingerprints from winning games
    3. Clusters with win_rate >> baseline AND size > min_cluster_size
       AND similarity < max_similarity_to_known_strategies
       → Candidate emergent strategy
    4. Register candidate as EmergentStrategy_XXX in strategy_stats.json
    5. Begin tracking win/loss for games where this pattern appears
    6. Counter-building: analyze games where the emergent strategy LOST
       to identify what the opponent did that worked
    """

    FINGERPRINT_DEPTH   = 8     # First 8 moves of the game (4 per player)
    MIN_CLUSTER_SIZE    = 15    # At least 15 games with this pattern before registering
    WIN_RATE_THRESHOLD  = 0.62  # Must win significantly more than baseline (~50%)
    MAX_KNOWN_SIMILARITY = 0.65 # Must be dissimilar enough from all known strategies

    def compute_fingerprint(self, move_history) -> np.ndarray:
        """
        Compact vector representation of opening move sequence.
        Features: relative pawn positions, wall placement zones, advance pattern.
        Designed to be invariant to minor variations while capturing strategic intent.
        """
        ...

    def cluster_and_register(self, fingerprint_db):
        """
        Run every 500 games.
        Uses DBSCAN or k-means on fingerprint vectors.
        Filters clusters by win_rate and dissimilarity from known strategies.
        Registers novel clusters as EmergentStrategy_XXX.
        """
        ...

    def match_fingerprint(self, fingerprint) -> EmergentStrategyMatch | None:
        """
        Called from opening_recognizer.recognize_strategy().
        Returns match if fingerprint is close to a registered emergent strategy.
        Returns None if no match (will be classified as UNKNOWN).
        """
        ...

    def build_counter_from_losses(self, emergent_strategy_id):
        """
        Scan all games where the emergent strategy was used and LOST.
        Find what the opponent (the winner) did that differed from baseline.
        Register the common winning response as a candidate counter.
        
        This is how counters to novel strategies emerge organically:
        not from human encoding but from self-play win/loss data.
        """
        games_vs_strategy = self.loss_log[emergent_strategy_id]
        winner_moves = [game.moves_for(game.winner) for game in games_vs_strategy]
        
        # Find move patterns common among winners
        common_patterns = extract_common_patterns(winner_moves, min_frequency=0.6)
        
        for pattern in common_patterns:
            strategy_tracker.register_counter(
                against=emergent_strategy_id,
                counter=CounterPattern(
                    action_type=pattern.action_type,
                    turn_range=pattern.turn_range,
                    win_rate=pattern.win_rate,
                    sample_size=len(games_vs_strategy),
                )
            )
```

**The full counter-strategy loop** in operation:

```
Turn 3:  opening_recognizer detects "Emergent_007" (confidence 0.72)
         strategy_tracker: EmergentStrategy_007 has win_rate 0.68, 
                           best known counter: "early left-wall + advance" (win_rate 0.61)
         counter_planner: returns recommended_action = WALL + direction hint
         strategy_vector[10:13] = [0, 1, 0] (recommend wall), strategy_vector[13] = 0.61

Turn 4:  network receives strategy_vector with counter recommendation
         network's policy head: 73% weight on wall placements in recommended zone
         MCTS runs with agreement_bonus on wall placements matching counter
         Model places counter wall

Turn 8:  path differential is now +2 (we're winning the race)
         strategy_tracker: record "counter worked at turn 4, game eventually won"
         Counter win_rate updated: 0.61 → 0.63

After game (loss scenario):
         backward_analyzer: finds turn 4 as decisive point
         counter_planner: "missed counter Y at turn 4, would have given +2.3 rating gain"
         High-priority experience injected: state_turn4, correct_counter_action
         EmergentStrategyDetector: logs this game for counter refinement
```

---

## 6. Data Structures & Representations

### 6.1 Move Representation

```python
@dataclass
class PawnMove:
    to_row: int
    to_col: int
    is_jump: bool
    is_diagonal_jump: bool

@dataclass
class WallMove:
    row: int            # 0–7 (top-left anchor)
    col: int            # 0–7
    orientation: str    # 'h' or 'v'

Move = Union[PawnMove, WallMove]
```

**Action Index Mapping**:
```
Indices 0–127:   Wall placements
    index = row * 16 + col * 2 + (0 if 'h' else 1)

Indices 128–135: Pawn moves
    128: forward     129: backward    130: left       131: right
    132: fwd-jump    133: diag-left   134: diag-right  135: back-jump
```

### 6.2 Game Record

```python
@dataclass
class GameRecord:
    game_id:              str
    winner:               int
    total_turns:          int
    states:               list[BoardState]
    actions:              list[Move]
    players:              list[int]
    predicted_ratings:    list[float]
    actual_ratings:       list[float]
    value_estimates:      list[float]
    strategy_vectors:     list[np.ndarray]   # [16] per turn
    agreements:           list[list]         # Per-turn agreement scores
    detected_strategies:  dict[int, StrategyLabel]
    walls_used:           dict[int, int]
    avg_move_ratings:     dict[int, float]
    mistakes:             list[MistakeRecord]         # Post-game
    counter_suggestions:  list[CounterRecord]         # Post-game
    profitable_deviations: list[DeviationRecord]      # [NEW] Post-game
```

### 6.3 Experience (Replay Buffer)

```python
@dataclass
class Experience:
    state_tensor:         np.ndarray    # [10 × 9 × 9]
    strategy_vector:      np.ndarray    # [16]
    action_index:         int           # 0–135
    policy_target:        np.ndarray    # MCTS visit distribution [136]
    value_target:         float         # +1 / -1
    rating_target:        float         # Actual computed rating
    agreement_scores:     list          # For strategy reward computation
    priority:             float         # Replay priority
    game_id:              str
    turn:                 int
    is_corrective:        bool          # True = from backward analysis
    is_deviation:         bool          # [NEW] True = profitable deviation
```

---

## 7. Key Algorithms (Pseudocode)

### 7.1 Complete Self-Play Game

```
function play_self_play_game(model, strategy_guide, recorder):
    board = Board()
    mcts_tt = MCTSTranspositionTable()
    endgame_tt = SolverTranspositionTable()
    recorder.start_game()

    while not board.is_terminal():
        player = board.current_player
        state_before = board.copy()

        # Compute strategy context (includes counter recommendation if strategy detected)
        counter_rec  = counter_planner.get_in_game_recommendation(board)
        strat_vector = strategy_guide.compute_strategy_vector(board, counter_rec)

        # Compute agreement scores for current best candidate moves
        # (used in UCB and later in loss function)
        agreements = agreement_engine.compute_strategy_agreements(board, ...)

        # Select move: MTD(f) in endgame, MCTS+lookahead otherwise
        move, mcts_policy = move_selector.select_move(
            board, model, strategy_guide, mcts_tt, endgame_tt, time_budget_ms=300
        )

        # Predict rating before committing
        _, _, pred_rating = model(board_to_tensor(state_before), strat_vector)

        # Execute
        board.apply_inplace(move)
        recorder.record_move(player, move, mcts_policy, pred_rating, agreements)

    # After game: compute actual ratings for all moves
    actual_ratings = compute_all_actual_ratings(recorder.game_record)
    recorder.set_actual_ratings(actual_ratings)

    # Post-game analysis
    loser  = 1 - board.winner
    winner = board.winner

    backward_analyzer.analyze(recorder.game_record, loser, model, strategy_guide)
    counter_planner.plan(recorder.game_record, loser)
    deviation_logger.log_profitable_deviations(recorder.game_record, winner)
    strategy_updater.update(recorder.game_record)
    agreement_weight_updater.update(recorder.game_record)
    emergent_detector.analyze_game(recorder.game_record)

    mcts_tt.clear()
    endgame_tt.clear()
    return recorder.finalize()
```

### 7.2 Training Step

```
function training_step(batch, model, optimizer, agreement_engine, config):
    # Forward pass
    policy_pred, value_pred, rating_pred = model(batch.state_tensors, batch.strategy_vectors)

    # Compute agreement bonuses (not violations) for each sample
    strategy_data = [
        AgreementData(
            bonus_score=agreement_engine.total_strategy_bonus(
                batch.agreements[i], agreement_weights
            )
        )
        for i in range(len(batch))
    ]

    # Loss (strategy component is a reward, not a penalty)
    loss, breakdown = total_loss(batch, {policy_pred, value_pred, rating_pred},
                                  strategy_data, config)

    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

    return loss.item(), breakdown
```

---

## 8. Training Configuration

```python
# Network
NUM_RES_BLOCKS         = 15
CHANNELS               = 128
STRATEGY_DIM           = 16       # Expanded from 12 for counter fields
STRATEGY_EMBED         = 16

# MCTS (time-based, not simulation-count-based)
MCTS_TIME_BUDGET_TRAIN = 300      # ms per move during self-play
MCTS_TIME_BUDGET_EVAL  = 1000     # ms per move during evaluation
MCTS_TIME_BUDGET_PLAY  = 2000     # ms per move vs human
C_PUCT                 = 1.5
LAMBDA_GUIDE           = 0.3      # Path guide bonus weight in UCB
LAMBDA_STRAT           = 0.2      # Agreement bonus weight in UCB
                                  # Note: floor is 0, not negative — no punishing deviations
TOP_K_LOOKAHEAD        = 5
OPP_PRED_SIMS          = 0        # 0 = argmax of policy (fast)

# Transposition Tables
MCTS_TT_MAX_SIZE       = 500_000
SOLVER_TT_MAX_SIZE     = 2_000_000

# Endgame Solver Thresholds
ENDGAME_WALLS_HARD     = 4        # Total walls ≤ 4: always MTD(f)
ENDGAME_WALLS_SOFT     = 6        # + max_path ≤ 5: MTD(f)
ENDGAME_PATH_SOFT      = 5
WIN_VALUE              = 1000     # Infinity sentinel for alpha-beta

# Training
BATCH_SIZE             = 512
LEARNING_RATE          = 0.001
LR_DECAY               = 0.95     # Every 5000 steps
WEIGHT_DECAY           = 1e-4
GRAD_CLIP              = 1.0
BUFFER_SIZE            = 500_000
MIN_BUFFER             = 10_000
UPDATE_FREQ            = 100
CHAMPION_UPDATE        = 1000

# Loss weights
W_POLICY               = 1.0
W_VALUE                = 1.0
W_RATING               = 0.5
W_RATING_MAX           = 0.3
W_STRATEGY_INIT        = 0.5      # Agreement REWARD weight (not violation penalty)
                                  # Annealed to 0.1 by late training
W_DEFEAT               = 0.5

# Self-play
NUM_WORKERS            = 8
GAMES_PER_EVAL         = 50
DIRICHLET_ALPHA        = 0.3
DIRICHLET_WEIGHT       = 0.25

# Emergent Strategy Detection
EMERGENT_FINGERPRINT_DEPTH    = 8
EMERGENT_MIN_CLUSTER_SIZE     = 15
EMERGENT_WIN_RATE_THRESHOLD   = 0.62
EMERGENT_MAX_KNOWN_SIMILARITY = 0.65
EMERGENT_CLUSTER_INTERVAL     = 500   # Games between cluster runs

# Agreement Weight Learning
AGREEMENT_WEIGHT_LR        = 0.01
AGREEMENT_WEIGHT_DECAY_RATE = 0.999   # For rules model consistently ignores
```

**Training Throughput Estimate (24 hours, RTX 3070 + 8 CPU workers)**:

| Configuration | Games/Day | Quality |
|---|---|---|
| Original (fixed sims, tree MCTS) | ~15,000 | Baseline |
| + Transposition Tables (DAG MCTS) | ~21,000–22,500 (+40%) | Higher (pooled Q/N) |
| + Time-based MCTS | ~22,000–24,000 | Consistent across phases |
| + MTD(f) endgame (less time wasted) | Marginal gain | Better endgame experiences |

The TT benefit is largest early in training when the model is weakest — many paths converge to the same bad positions, so the transposition rate is very high.

---

## 9. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                           TRAINING LOOP                                  │
│                                                                          │
│  ┌──────────────┐  game_records    ┌──────────────────────────────────┐  │
│  │  Self-Play   │─────────────────▶│     Experience Buffer            │  │
│  │  Workers     │                  │     (Prioritized Replay)         │  │
│  │  (N parallel)│◀──weights update └──────────────┬───────────────────┘  │
│  └──────┬───────┘                                  │ batches              │
│         │                                          ▼                     │
│         │              ┌───────────────────────────────────────────────┐ │
│         │              │               TRAINER                         │ │
│  reads  │              │  - Forward pass                               │ │
│  ───────┘              │  - Agreement bonus computation (not penalties)│ │
│                        │  - 5-component loss (strategy = reward term)  │ │
│  ┌─────────────────┐   │  - Backward + update                         │ │
│  │ Strategy Guide  │──▶│                                               │ │
│  │ (agreement-     │   └───────────────────────────────────────────────┘ │
│  │  based rules)   │                                                    │
│  └────────┬────────┘                                                    │
│           │                                                             │
│    updates│                                                             │
│           ▼                                                             │
│  ┌──────────────────────────────────────────────────┐                  │
│  │         STRATEGY FEEDBACK LOOP                   │                  │
│  │                                                  │                  │
│  │  AgreementWeightUpdater  ◄── per-game outcomes   │                  │
│  │  EmergentStrategyDetector ◄── deviation log      │                  │
│  │  StrategyTracker         ◄── backward analysis   │                  │
│  │  CounterPlanner          ◄── detected strategies  │                  │
│  └──────────────────────────────────────────────────┘                  │
└──────────────────────────────────────────────────────────────────────────┘

INFERENCE (per move):
BoardState
    │
    ├─ ZobristHasher ──────────────▶ current_hash (O(1) incremental)
    │
    ├─ BFS Pathfinder ─────────────▶ path_map_own, path_map_opp
    │
    ├─ Opening Recognizer ─────────▶ strategy_label (known | emergent | unknown)
    │
    ├─ Counter Planner ────────────▶ counter_recommendation + confidence
    │
    ├─ Agreement Engine ───────────▶ agreement_scores per candidate move
    │
    ▼
Board Tensor [10×9×9] + Strategy Vector [16]
    │
    ▼
QuoridorSINN
    ├── Policy Head ──▶ action priors [136]
    ├── Value Head  ──▶ position value [-1, +1]
    └── Rating Head ──▶ predicted move quality [0, 5]
    │
    ▼
Phase Check:
    ├─ Endgame (≤4 walls OR ≤6 walls + path≤5)?
    │   └─ MTD(f) Solver + SolverTT ──▶ EXACT optimal move (<20ms)
    │
    └─ Normal?
        └─ DAG-MCTS (TT-backed, time-budgeted, agreement UCB)
              │
              ▼
           Top-5 Moves ──▶ 1-Step Opponent Lookahead ──▶ Final Move
```

---

## 10. Implementation Priorities & Phases

### Phase 1: Foundation (Week 1)
1. `game_engine/board.py` — Board state + `current_hash` field
2. `game_engine/zobrist_hash.py` — Full + incremental hash
3. `game_engine/moves.py` — Generation, application, `apply_move` updates hash
4. `game_engine/pathfinder.py` — BFS (used everywhere; must be fast)
5. `game_engine/rules.py` — Wall legality + anti-blockade
6. `game_engine/game.py` — Game loop

**Validate**: Random vs random game; correct termination; hash is consistent; BFS is correct.

### Phase 2: Strategy Guide (Week 1–2)
1. `strategy/strategy_guide.py` — Rules as `agreement()` functions (never returning negative)
2. `strategy/agreement_engine.py` — `compute_strategy_agreements()` and `total_strategy_bonus()`
3. `strategy/opening_recognizer.py` — Classification for known strategies
4. `strategy/strategy_tracker.py` — Stats database with counter tracking
5. `strategy/counter_planner.py` — Counter lookup for known strategies

**Validate**: Feed board states; agreements in [0,1]; no negatives; counter lookup returns sensible recommendations.

### Phase 3: Neural Network (Week 2)
1. `models/board_encoder.py`
2. `models/residual_block.py`
3. `models/path_attention.py`
4. `models/strategy_layer.py` — Updated for 16-dim strategy vector
5. `models/policy_head.py`, `value_head.py`, `rating_head.py`
6. `models/quoridor_net.py`

**Validate**: Random input [10×9×9] + [16] → correct output shapes, no NaN/Inf.

### Phase 4: Search (Week 2–3)
1. `search/transposition_table.py` — Both variants
2. `search/mcts.py` — DAG structure + time-budget + agreement-based UCB
3. `search/endgame_solver.py` — MTD(f) + alpha-beta + move ordering
4. `search/opponent_predictor.py`
5. `search/move_selector.py` — Phase switching

**Validate**: MCTS produces sensible distributions; TT collapses transpositions correctly; endgame solver finds forced wins in test positions.

### Phase 5: Rating + Analysis (Week 3)
1. `analysis/move_rater.py` — Path differential delta → 0–5 rating + agreement bonus
2. `analysis/game_recorder.py`
3. `analysis/backward_analyzer.py`
4. `analysis/deviation_logger.py` — Log profitable deviations
5. `strategy/emergent_strategy_detector.py` — Fingerprinting + clustering

**Validate**: Rating is reasonable on hand-crafted test positions; backward analysis identifies obvious mistakes; deviation logger fires on moves with zero agreement + high rating.

### Phase 6: Training Pipeline (Week 3–4)
1. `training/experience_buffer.py` — Priority tiers: corrective > deviation > standard
2. `training/loss_functions.py` — Agreement reward (negative loss) in strategy component
3. `training/trainer.py`
4. `training/self_play_worker.py` — Full game loop including TT lifecycle
5. `training/self_play_manager.py` — Parallel workers
6. `strategy/agreement_weight_updater.py` — Per-rule weight learning

**Validate**: Loss decreases over 1000 steps; strategy component of loss is negative (reward working); agreement weights start updating.

### Phase 7: Full Training Run (Week 4)
1. `train.py`, `evaluate.py`, `play.py`
2. 24-hour training run

**Monitor**:
- Average move rating per 1000 games: should trend upward
- Strategy agreement loss: should remain negative (model following good advice)
- Agreement weights: should update — some rules increase, some decay (model transcending them)
- Emergent strategies detected: check after 5000+ games
- Value loss convergence
- Win rate vs frozen champion

---

## 11. Design Decisions Summary

| Decision | Choice | Rationale |
|---|---|---|
| **Strategy loss type** | Agreement reward (never violation penalty) | Human strategy = advice. Penalizing deviations creates a ceiling; rewarding agreements creates a floor with no ceiling. Model can transcend any encoded rule. |
| **Agreement weight learning** | Weights updated per game based on outcome | Rules that reliably predict good moves strengthen; rules the model consistently ignores decay to near-zero. The guide adapts to the model. |
| **Emergent strategy detection** | Fingerprint clustering every N games | Self-play will invent strategies; we need to detect, name, and build counters to them without human involvement. |
| **Counter-strategy loop** | Detection → lookup/learn → inject into vector → record | Full closed loop: model knows what the opponent is doing AND what has historically worked against it. |
| **Transposition Tables** | Zobrist hash, per-game TTs (both variants) | 40–50% more effective simulations in same time budget. Benefit is largest early in training when many paths converge to bad positions. |
| **Time-based MCTS** | Budget in ms, not fixed sim count | Consistent quality: late-game positions (fewer moves) get more sims for same time budget. |
| **MTD(f) Endgame** | Activated at ≤4 walls (hard) or ≤6+path≤5 (soft) | Exact play where MCTS is weakest. Eliminates the most common class of late-game blunders entirely. |
| **Base architecture** | ResNet 15 blocks, 128 channels | Spatial locality for walls; proven in AlphaZero; trainable in one day on a single GPU. |
| **Path as guide** | BFS maps as attention gates | Direct translation of path differential as the strategic core of Quoridor. |
| **Rating system** | 0–5 continuous on path differential delta | Richer signal than win/loss; agreement bonus adds at most +0.5 (path signal always dominates). |
| **Post-game analysis** | Backward scan + deviation logging | Identifies root cause of defeat AND profitable innovations simultaneously. |
| **Self-play** | Latest vs Champion (delayed update) | Prevents policy collapse; stable training target. |
| **From scratch** | No pretrained models; agreement bonus guides early training | As required. Agreement bonuses replace the warm-start that pretrained weights would provide. |
```

---

*End of Final Technical Plan.*
*Strategy is advice. The model earns the right to disagree.*
