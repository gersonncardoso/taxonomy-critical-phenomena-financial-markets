# config.py - Central configuration for pipeline parameters

# Date range for analysis
START_DATE = '1995-01-01'
END_DATE = '2025-12-31'

# Data directories
RAW_DATA_DIR = 'data/raw/'
PROCESSED_DATA_DIR = 'data/processed/'
FIGURES_DIR = 'figures/'

# Network construction options
COMPUTE_MST = False  # Only planar (PMFG)
COMPUTE_PLANAR = True
COMPUTE_COMPLETE = False  # Desativado: só gera planar

# Asset selection rules
LIQUIDITY_THRESHOLD = 0.8  # 80% of trading days
PREFER_PN = True  # Prefer PN over ON when both exist

# Rolling window parameters
WINDOW_LENGTH_DAYS = 252  # ~12 months
WINDOW_STEP_DAYS = 21     # ~1 month

# Statistical filtering
P_VALUE_EDGE_STYLE = 'dashed'  # Non-significant edges are dashed, not removed
P_VALUE_THRESHOLD = 0.05

# Output options
SAVE_NETWORKS = True
SAVE_METRICS = True
SAVE_FIGURES = True

# Paper 1 is restricted to the B3 rolling financial-network pipeline.
INCLUDE_GDELT = False
INCLUDE_MRQAP = False

# Other options can be added as needed
