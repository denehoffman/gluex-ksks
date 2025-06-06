from dataclasses import dataclass
from pathlib import Path
from numpy import pi
from laddu import available_parallelism

DATA_TYPES = ['data', 'sigmc', 'bkgmc', 'bggen']

ANALYSIS_PATH = Path.cwd() / 'analysis'
STATE_PATH = Path.cwd() / '.modak'
LOG_PATH = ANALYSIS_PATH / 'logs'
RAW_DATASETS_PATH = ANALYSIS_PATH / 'raw_datasets'
DATASETS_PATH = ANALYSIS_PATH / 'datasets'
RAW_DATA_PATH = RAW_DATASETS_PATH / 'data'
DATA_PATH = DATASETS_PATH / 'data'
RAW_SIGMC_PATH = RAW_DATASETS_PATH / 'sigmc'
SIGMC_PATH = DATASETS_PATH / 'sigmc'
RAW_BKGMC_PATH = RAW_DATASETS_PATH / 'bkgmc'
BKGMC_PATH = DATASETS_PATH / 'bkgmc'
RAW_BGGEN_PATH = RAW_DATASETS_PATH / 'bggen'
BGGEN_PATH = DATASETS_PATH / 'bggen'


RAW_DATASET_PATH = {
    'data': RAW_DATA_PATH,
    'sigmc': RAW_SIGMC_PATH,
    'bkgmc': RAW_BKGMC_PATH,
    'bggen': RAW_BGGEN_PATH,
}
DATASET_PATH = {
    'data': DATA_PATH,
    'sigmc': SIGMC_PATH,
    'bkgmc': BKGMC_PATH,
    'bggen': BGGEN_PATH,
}

FITS_PATH = ANALYSIS_PATH / 'fits'
PLOTS_PATH = ANALYSIS_PATH / 'plots'
REPORTS_PATH = ANALYSIS_PATH / 'reports'
MISC_PATH = ANALYSIS_PATH / 'misc'

RUN_RANGES = {
    's17': (30000, 39999),
    's18': (40000, 49999),
    'f18': (50000, 59999),
    's20': (71275, 79999),
}

RUN_PERIODS = list(RUN_RANGES.keys())
RUN_PERIODS_BGGEN = ['s18']
RUN_PERIODS_BY_DATA_TYPE = {
    'data': RUN_PERIODS,
    'sigmc': RUN_PERIODS,
    'bkgmc': RUN_PERIODS,
    'bggen': RUN_PERIODS_BGGEN,
}

TRUE_POL_ANGLES = {
    's17': {'0.0': 1.8, '45.0': 47.9, '90.0': 94.5, '135.0': -41.6},
    's18': {'0.0': 4.1, '45.0': 48.5, '90.0': 94.2, '135.0': -42.4},
    'f18': {'0.0': 3.3, '45.0': 48.3, '90.0': 92.9, '135.0': -42.1},
    's20': {'0.0': 1.4, '45.0': 47.1, '90.0': 93.4, '135.0': -42.2},
}

PARTICLE_TO_LATEX = {
    'Proton': 'p',
    'PiPlus1': r'\pi^+_1',
    'PiMinus1': r'\pi^-_1',
    'PiPlus2': r'\pi^+_2',
    'PiMinus2': r'\pi^-_2',
}


def mkdirs_raw() -> None:
    for raw_path in {RAW_DATA_PATH, RAW_SIGMC_PATH, RAW_BKGMC_PATH}:
        (raw_path / 'logs').mkdir(parents=True, exist_ok=True)
        (raw_path / 's17').mkdir(parents=True, exist_ok=True)
        (raw_path / 's18').mkdir(parents=True, exist_ok=True)
        (raw_path / 'f18').mkdir(parents=True, exist_ok=True)
        (raw_path / 's20').mkdir(parents=True, exist_ok=True)
    (RAW_BGGEN_PATH / 'logs').mkdir(parents=True, exist_ok=True)
    (RAW_BGGEN_PATH / 's18').mkdir(parents=True, exist_ok=True)


def mkdirs():
    ANALYSIS_PATH.mkdir(parents=True, exist_ok=True)
    LOG_PATH.mkdir(parents=True, exist_ok=True)
    RAW_DATASETS_PATH.mkdir(parents=True, exist_ok=True)
    DATASETS_PATH.mkdir(parents=True, exist_ok=True)
    RAW_DATA_PATH.mkdir(parents=True, exist_ok=True)
    DATA_PATH.mkdir(parents=True, exist_ok=True)
    RAW_SIGMC_PATH.mkdir(parents=True, exist_ok=True)
    SIGMC_PATH.mkdir(parents=True, exist_ok=True)
    RAW_BKGMC_PATH.mkdir(parents=True, exist_ok=True)
    BKGMC_PATH.mkdir(parents=True, exist_ok=True)
    RAW_BGGEN_PATH.mkdir(parents=True, exist_ok=True)
    BGGEN_PATH.mkdir(parents=True, exist_ok=True)
    FITS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)
    REPORTS_PATH.mkdir(parents=True, exist_ok=True)
    MISC_PATH.mkdir(parents=True, exist_ok=True)
    mkdirs_raw()


RED = '#e41a1c'
LIGHT_RED = '#f28c8d'
PINK = '#f781bf'
ORANGE = '#ff7f00'
YELLOW = '#ffff33'
PURPLE = '#984ea3'
GREEN = '#4daf4a'
BLUE = '#377eb8'
LIGHT_BLUE = '#97bfe0'
BROWN = '#a65628'
GRAY = '#999999'
BLACK = '#000000'
WHITE = '#ffffff'

RFL_PDF_RANGE = (0.0, 0.5)

MESON_MASS_RANGE = (1.0, 2.0)
MESON_MASS_BINS = 50
MESON_MASS_2D_BINS = 80

BARYON_MASS_RANGE = (1.4, 3.7)
BARYON_MASS_BINS = 115
BARYON_MASS_2D_BINS = 184

COSTHETA_RANGE = (-1.0, 1.0)
COSTHETA_BINS = 80

PHI_RANGE = (-pi, pi)
PHI_BINS = 80

RF_RANGE = (-20.0, 20.0)
RF_BINS = 200

CHISQDOF_RANGE = (0.0, 10.0)
CHISQDOF_BINS = 50

PROTONZ_RANGE = (40.0, 90.0)
PROTONZ_BINS = 50

RFL_RANGE = (0.0, 0.2)
RFL_BINS = 100
RFL_FIT_RANGE = (0.0, 2.0)
RFL_FIT_BINS = 2000

MM2_RANGE = (-0.01, 0.01)
MM2_BINS = 100

ME_RANGE = (-1.0, 1.0)
ME_BINS = 100

YOUDENJ_RANGE = (0.0, 10.0)
YOUDENJ_BINS = 200

BEAM_ENERGY_RANGE = (7.8, 9.0)
BEAM_ENERGY_BINS = 120

P_RANGE = (0.0, 10.0)
P_BINS = 500

BETA_RANGE = (-0.2, 1.2)
BETA_BINS = 400

DELTA_BETA_RANGE = (-1.0, 1.0)
DELTA_BETA_BINS = 400

DELTA_T_RANGE = (-10.0, 10.0)
DELTA_T_BINS = 500

DEDX_RANGE = (0.0, 25.0)
DEDX_BINS = 250

DETECTOR_THETA_DEG_RANGE = {'BCAL': (10.0, 140.0), 'FCAL': (0.0, 12.0)}
DETECTOR_THETA_DEG_BINS = {'BCAL': 260, 'FCAL': 120}

E_OVER_P_RANGE = (0.0, 4.0)
E_OVER_P_BINS = 200


@dataclass
class Particle:
    label: str
    color: str
    center: float
    width: float
    row: int
    established: bool = True


f0_980 = Particle(r'$f_0(980)$', RED, 0.990, 0.100, 0)
f0_1370 = Particle(r'$f_0(1370)$', RED, 1.350, 0.500, 0)
f0_1500 = Particle(r'$f_0(1500)$', RED, 1.522, 0.108, 1)
f0_1710 = Particle(r'$f_0(1710)$', RED, 1.733, 0.150, 0)
f0_1770 = Particle(r'$f_0(1770)$', RED, 1.784, 0.161, 1, established=False)
f0_2020 = Particle(r'$f_0(2020)$', RED, 1.982, 0.440, 2)
f0_2100 = Particle(r'$f_0(2100)$', RED, 2.095, 0.287, 1)
f0_2200 = Particle(r'$f_0(2200)$', RED, 2.187, 0.210, 0, established=False)

a0_980 = Particle(r'$a_0(980)$', GREEN, 0.980, 0.100, 2)
a0_1450 = Particle(r'$a_0(1450)$', GREEN, 1.439, 0.258, 2)
a0_1710 = Particle(r'$a_0(1710)$', GREEN, 1.713, 0.107, 2, established=False)

f2_1270 = Particle(r'$f_2(1270)$', ORANGE, 1.2754, 0.1866, 3)
f2_1430 = Particle(r'$f_2(1430)$', ORANGE, 1.430, 0.013, 3, established=False)
f2_1525 = Particle(r"$f_2'(1525)$", ORANGE, 1.5173, 0.072, 3)
f2_1565 = Particle(r'$f_2(1565)$', ORANGE, 1.571, 0.132, 4)
f2_1640 = Particle(r'$f_2(1640)$', ORANGE, 1.639, 0.100, 3, established=False)
f2_1810 = Particle(r'$f_2(1810)$', ORANGE, 1.815, 0.197, 3, established=False)
f2_1950 = Particle(r'$f_2(1950)$', ORANGE, 1.936, 0.464, 4)
f2_2010 = Particle(r'$f_2(2010)$', ORANGE, 2.010, 0.200, 5)
f2_2150 = Particle(r'$f_2(2150)$', ORANGE, 2.157, 0.152, 3, established=False)

a2_1320 = Particle(r'$a_2(1320)$', PURPLE, 1.3181, 0.1098, 5)
a2_1700 = Particle(r'$a_2(1700)$', PURPLE, 1.706, 0.380, 5)

MESON_PARTICLES = [
    f0_980,
    f0_1370,
    f0_1500,
    f0_1710,
    f0_1770,
    f0_2020,
    f0_2100,
    f0_2200,
    a0_980,
    a0_1450,
    a0_1710,
    f2_1270,
    f2_1430,
    f2_1525,
    f2_1565,
    f2_1640,
    f2_1810,
    f2_1950,
    f2_2010,
    f2_2150,
    a2_1320,
    a2_1700,
]
MAX_FITS = 2
_n_threads = available_parallelism()
NUM_THREADS = (_n_threads - 1) // 2 if available_parallelism() > 2 else 1
GUIDED_MAX_STEPS = 400
NBOOT = 30
