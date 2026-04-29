"""
Configuration for random-window episodic alpha generation.

Each "episode" consumes a randomly-sampled segment of `WINDOW_BARS` consecutive
3-second bars from the full multi-year dataset.  The factor is then required
to predict the next `FUTURE_BARS` (~5 minutes) of risk-adjusted returns.

Knobs here are the *defaults*; the training script exposes them as flags.
"""

# ---- Random-window sampling ------------------------------------------------
WINDOW_BARS: int = 1200          # observation/state length (1 hour @ 3s)
FUTURE_BARS: int = 100           # prediction horizon (~5 min @ 3s)
EXECUTION_DELAY: int = 1         # bars between signal and entry
LOOKBACK_BARS: int = 1200        # rolling z-score window for position sizing
EPISODE_HISTORY_BUFFER: int = 1200  # extra bars before the window for warm-up
                                    # (allows DELTA_TIMES up to 1200 inside the
                                    # observation window itself)

# ---- Factor pool -----------------------------------------------------------
POOL_CAPACITY: int = 20          # initial / steady-state pool size
IC_MUT_THRESHOLD: float = 0.95   # de-dup correlation ceiling
DECORRELATION_BONUS: float = 0.5 # reward bonus for low avg-corr w/ pool

# ---- Reward shaping (risk-adjusted) ---------------------------------------
# Main objective is Sortino on the future-100-bar bar-by-bar PnL series.
SORTINO_WEIGHT: float = 1.0
SHARPE_WEIGHT:  float = 0.0      # set >0 to mix in Sharpe
IC_WEIGHT:      float = 0.5      # IR / IC bonus (rank IC of z vs forward ret)
TURNOVER_COST:  float = 0.0006

# Drawdown / fat-tail penalties.  Penalty kicks in when |metric| exceeds
# threshold; magnitude scales linearly with breach in `tanh` units.
MAX_DRAWDOWN_THRESHOLD: float = 0.02   # 2% over the 100-bar horizon
MAX_DRAWDOWN_PENALTY:   float = 2.0
FAT_TAIL_THRESHOLD:     float = 0.01   # max abs single-bar return
FAT_TAIL_PENALTY:       float = 2.0
KURTOSIS_THRESHOLD:     float = 8.0    # excess kurtosis trigger
KURTOSIS_PENALTY:       float = 1.0

# Complexity penalty (expression length)
COMPLEXITY_PENALTY: float = 0.005

# ---- Genetic Programming (factor-pool evolution) ---------------------------
GP_ENABLED: bool = True
GP_EVERY_N_EPOCHS: int = 20      # invoke GP every N RL epochs (in [10, 50])
GP_OFFSPRING: int = 10           # children produced per GP round
GP_CROSSOVER_RATE: float = 0.6
GP_MUTATION_RATE: float = 0.4
GP_MAX_TRIES: int = 50           # tries per offspring before giving up
GP_TOURNAMENT_K: int = 3
GP_REPLACE_WORST_N: int = 5      # how many worst pool members may be replaced

# ---- Multi-window aggregation (when scoring a factor for inclusion) --------
SCORE_WINDOWS: int = 4           # average reward over K random windows when
                                 # deciding pool inclusion (stability vs noise)
