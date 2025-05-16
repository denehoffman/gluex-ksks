from pathlib import Path

ANALYSIS_PATH = Path.cwd() / 'analysis'
DATASETS_PATH = ANALYSIS_PATH / 'datasets'
DATA_PATH = ANALYSIS_PATH / 'datasets' / 'data'
SIGMC_PATH = ANALYSIS_PATH / 'datasets' / 'sigmc'
BKGMC_PATH = ANALYSIS_PATH / 'datasets' / 'bkgmc'
BGGEN_PATH = ANALYSIS_PATH / 'datasets' / 'bggen'

PLOTS_PATH = ANALYSIS_PATH / 'plots'
REPORTS_PATH = ANALYSIS_PATH / 'reports'
MISC_PATH = ANALYSIS_PATH / 'misc'

RUN_RANGES = {
    's17': (30000, 39999),
    's18': (40000, 49999),
    'f18': (50000, 59999),
    's20': (71275, 79999),
}

TRUE_POL_ANGLES = {
    's17': {'0.0': 1.8, '45.0': 47.9, '90.0': 94.5, '135.0': -41.6},
    's18': {'0.0': 4.1, '45.0': 48.5, '90.0': 94.2, '135.0': -42.4},
    'f18': {'0.0': 3.3, '45.0': 48.3, '90.0': 92.9, '135.0': -42.1},
    's20': {'0.0': 1.4, '45.0': 47.1, '90.0': 93.4, '135.0': -42.2},
}
