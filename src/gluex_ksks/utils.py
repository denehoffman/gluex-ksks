from __future__ import annotations

import pickle
import sqlite3
import laddu as ld
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable
from datetime import datetime

from matplotlib.colors import CenteredNorm, ListedColormap
import numpy as np
import polars as pl
import uproot
from scipy.stats import chi2
from scipy.optimize import OptimizeResult, minimize
from uproot.behaviors.TBranch import HasBranches
import matplotlib.pyplot as plt

from gluex_ksks.constants import (
    MISC_PATH,
    RCDB_SELECTION_PREFIX,
    RCDB_SELECTION_SUFFIX,
    REST_VERSION_TIMESTAMPS,
    RFL_FIT_BINS,
    RFL_FIT_RANGE,
    RUN_RANGES,
    TRUE_POL_ANGLES,
)
from gluex_ksks.types import FloatArray, IntArray

if TYPE_CHECKING:
    from pathlib import Path

    from numpy.typing import NDArray


def root_to_dataframe(input_path: Path, tree: str = 'kin') -> pl.DataFrame:
    tt = uproot.open(
        f'{input_path}:{tree}',
    )
    assert isinstance(tt, HasBranches)  # noqa: S101
    root_data = tt.arrays(library='np')
    return pl.from_dict(root_data)


@dataclass
class ScalingFactors:
    hodoscope_hi_factor: float
    hodoscope_lo_factor: float
    microscope_factor: float
    energy_bound_hi: float
    energy_bound_lo: float


class CCDBData:
    def __init__(
        self,
        accidental_scaling_factors: dict[
            int,
            ScalingFactors,
        ],
    ):
        self.accidental_scaling_factors: dict[int, ScalingFactors] = (
            accidental_scaling_factors
        )

    def get_scaling(
        self,
        run_number: int,
        beam_energy: float,
    ) -> float:
        factors = self.accidental_scaling_factors.get(run_number)
        if factors is None:
            return 1.0
        if beam_energy > factors.energy_bound_hi:
            return factors.hodoscope_hi_factor
        if beam_energy > factors.energy_bound_lo:
            return factors.microscope_factor
        return factors.hodoscope_lo_factor

    def get_accidental_weight(
        self,
        run_number: int,
        beam_energy: float,
        rf: float,
        weight: float,
        *,
        is_mc: bool,
    ) -> float:
        relative_beam_bucket = int(np.floor(rf / 4.008016032 + 0.5))
        if abs(relative_beam_bucket) == 1:
            return 0.0
        if abs(relative_beam_bucket) == 0:
            return weight
        scale = (
            1.0
            if is_mc
            else self.get_scaling(
                run_number,
                beam_energy,
            )
        )
        return weight * (-scale / 6.0)  # (4 - 1) * 2 out-of-time peaks


class Histogram:
    def __init__(
        self, counts: FloatArray, bins: FloatArray, errors: FloatArray | None = None
    ):
        self.counts = counts
        self.bins = bins
        self.errors = errors if errors is not None else np.zeros_like(counts)

    @property
    def nbins(self) -> int:
        return len(self.bins) - 1

    @property
    def centers(self) -> FloatArray:
        return (self.bins[1:] + self.bins[:-1]) / 2.0

    @staticmethod
    def empty(nbins: int, range: tuple[float, float]) -> Histogram:
        bins = np.histogram_bin_edges([], bins=nbins, range=range)
        counts = np.zeros(nbins)
        errors = np.zeros(nbins)
        return Histogram(counts, bins, errors)

    @staticmethod
    def sum(histograms: list[Histogram]) -> Histogram | None:
        if not histograms:
            return None
        bins = histograms[0].bins
        for histogram in histograms:
            assert histogram.bins == bins  # noqa: S101
        counts = np.sum(
            np.array([histogram.counts for histogram in histograms]), axis=0
        )
        errors = np.sqrt(
            np.sum(
                np.array(
                    [
                        np.power(
                            histogram.errors
                            if histogram.errors is not None
                            else np.zeros_like(histogram.counts),
                            2,
                        )
                        for histogram in histograms
                    ]
                ),
                axis=0,
            )
        )
        return Histogram(counts, bins, errors)

    def get_bin_index(self, value: float) -> int | None:
        ibin = np.digitize(value, self.bins)
        if ibin == 0 or ibin == len(self.bins):
            return None
        return int(ibin) - 1


class RCDBData:
    def __init__(
        self,
        pol_angles: dict[int, tuple[str, str, float | None]],
        pol_magnitudes: dict[str, dict[str, Histogram]],
    ):
        self.pol_angles: dict[int, tuple[str, str, float | None]] = pol_angles
        self.pol_magnitudes: dict[str, dict[str, Histogram]] = pol_magnitudes

    def get_eps_xy(
        self,
        run_number: int,
        beam_energy: float,
    ) -> tuple[float, float, bool]:
        pol_angle = self.pol_angles.get(run_number)
        if pol_angle is None:
            return (np.nan, np.nan, False)
        run_period, pol_name, angle = pol_angle
        if angle is None:
            return (np.nan, np.nan, False)
        pol_hist = self.pol_magnitudes[run_period][pol_name]
        energy_index = np.digitize(beam_energy, pol_hist.bins)
        if energy_index >= len(pol_hist.counts):
            return (np.nan, np.nan, False)
        magnitude: float = float(pol_hist.counts[energy_index])
        return magnitude * np.cos(angle), magnitude * np.sin(angle), True


@dataclass
class FitResult:
    n2ll: float
    npars: int
    nobs: int
    opt: OptimizeResult

    @staticmethod
    def from_opt(opt: OptimizeResult, nobs: int) -> FitResult:
        return FitResult(
            opt.fun,
            len(opt.x),
            nobs,
            opt,
        )

    @property
    def aic(self) -> float:
        return 2.0 * self.npars + self.n2ll  # 2k + -2ln(L)

    @property
    def bic(self) -> float:
        return self.npars * np.log(self.nobs) + self.n2ll  # kln(n) + -2ln(L)

    @property
    def values(self) -> FloatArray:
        return self.opt.x

    @property
    def errors(self) -> FloatArray:
        return np.sqrt(np.diag(self.opt.hess_inv.todense()))


def get_run_period(run_number: int) -> str | None:
    for rp, (lo, hi) in RUN_RANGES.items():
        if lo <= run_number <= hi:
            return rp
    return None


def get_run_period_bound(run_number: int) -> str | None:
    for rp, (_, hi) in RUN_RANGES.items():
        if run_number <= hi:
            return rp
    return None


def get_pol_angle(run_period: str | None, angle_deg: str) -> float | None:
    if run_period is None:
        return None
    pol_angle_deg = TRUE_POL_ANGLES[run_period].get(angle_deg)
    if pol_angle_deg is None:
        return None
    return pol_angle_deg * np.pi / 180.0


def get_ccdb() -> CCDBData:
    return pickle.load((MISC_PATH / 'ccdb.pkl').open('rb'))  # noqa: S301


def get_rcdb() -> RCDBData:
    return pickle.load((MISC_PATH / 'rcdb.pkl').open('rb'))  # noqa: S301


def select_mesons_tag(select_mesons: bool | None) -> str:
    tag = 'None'
    if select_mesons is not None:
        tag = 'mesons' if select_mesons else 'baryons'
    return tag


def density_hist_to_pdf(
    rfl: Histogram,
) -> Callable[[FloatArray], FloatArray]:
    def pdf(x: float) -> float:
        idx = np.searchsorted(rfl.bins, x, side='right') - 1
        if idx < 0 or idx >= len(rfl.counts):
            return 0.0
        return rfl.counts[idx]

    vpdf = np.vectorize(pdf)

    return vpdf


def density_hists_to_pdf(
    rfl1: Histogram,
    rfl2: Histogram,
) -> Callable[[FloatArray, FloatArray], FloatArray]:
    pdf1 = density_hist_to_pdf(rfl1)
    pdf2 = density_hist_to_pdf(rfl2)

    def pdf(rfl1: FloatArray, rfl2: FloatArray) -> FloatArray:
        return pdf1(rfl1) * pdf2(rfl2)

    return pdf


def get_quantile_edges(
    variable: FloatArray, *, bins: int, weights: FloatArray
) -> FloatArray:
    # This is a custom wrapper method around numpy.quantile that
    # first rescales the weights so that they are between 0 and 1
    # and then runs the quantile method with those weights
    scaled_weights = (weights - np.min(weights)) / (np.max(weights) - np.min(weights))
    return np.quantile(
        variable,
        np.linspace(0, 1, bins + 1),
        weights=scaled_weights,
        method='inverted_cdf',
    )


def get_quantile_indices(
    variable: FloatArray, *, bins: int, weights: FloatArray
) -> list[IntArray]:
    quantiles = get_quantile_edges(variable, bins=bins, weights=weights)
    quantiles[-1] = np.inf  # ensure the maximum value gets placed in the last bin
    quantile_assignment = np.digitize(variable, quantiles)
    return [np.where(quantile_assignment == i)[0] for i in range(1, bins + 1)]


@dataclass
class SPlotArrays:
    rfl1: FloatArray
    rfl2: FloatArray
    weight: FloatArray
    control: FloatArray

    @staticmethod
    def from_polars(data: pl.LazyFrame, *, control: str) -> SPlotArrays:
        data_df = data.select('RFL1', 'RFL2', 'weight', control).collect()
        return SPlotArrays(
            rfl1=data_df['RFL1'].to_numpy(),
            rfl2=data_df['RFL2'].to_numpy(),
            weight=data_df['weight'].to_numpy(),
            control=data_df[control].to_numpy(),
        )

    def __len__(self) -> int:
        return len(self.rfl1)

    @property
    def nevents(self) -> float:
        return float(np.sum(self.weight))


def exp_pdf_single(*, rfl: FloatArray, lda: float) -> FloatArray:
    return np.exp(-rfl * lda) * lda


def exp_pdf(*, rfl1: FloatArray, rfl2: FloatArray, lda: float) -> FloatArray:
    return exp_pdf_single(rfl=rfl1, lda=lda) * exp_pdf_single(rfl=rfl2, lda=lda)


def fit_lda(
    *,
    rfl1: FloatArray,
    rfl2: FloatArray,
    weight: FloatArray,
    lda0: float,
) -> OptimizeResult:
    def nll(*args: float) -> float:
        return float(
            -2.0
            * np.sum(
                np.sort(
                    weight
                    * np.log(
                        exp_pdf(rfl1=rfl1, rfl2=rfl2, lda=args[0])
                        + np.finfo(float).tiny
                    )
                )
            )
        )

    opt = minimize(nll, (lda0,), bounds=[(0.0, 200.0)])
    return opt


@dataclass
class SigMCFitComponents:
    rfl1: Histogram
    rfl2: Histogram

    @property
    def pdf1(self) -> Callable[[FloatArray], FloatArray]:
        return density_hist_to_pdf(self.rfl1)

    @property
    def pdf2(self) -> Callable[[FloatArray], FloatArray]:
        return density_hist_to_pdf(self.rfl2)

    @property
    def pdf(self) -> Callable[[FloatArray, FloatArray], FloatArray]:
        return density_hists_to_pdf(self.rfl1, self.rfl2)


def get_sigmc_fit_components(
    *,
    arrays: SPlotArrays,
) -> SigMCFitComponents:
    hist1 = Histogram(
        *np.histogram(
            arrays.rfl1,
            bins=RFL_FIT_BINS,
            range=RFL_FIT_RANGE,
            weights=arrays.weight,
            density=True,
        )
    )
    hist2 = Histogram(
        *np.histogram(
            arrays.rfl2,
            bins=RFL_FIT_BINS,
            range=RFL_FIT_RANGE,
            weights=arrays.weight,
            density=True,
        )
    )
    return SigMCFitComponents(hist1, hist2)


def get_bkgmc_lda0s_list(
    *,
    arrays: SPlotArrays,
    mass_bins: FloatArray,
) -> list[float]:
    lda0s: list[float] = []
    for c_lo, c_hi in zip(mass_bins[:-1], mass_bins[1:]):
        mask: NDArray[np.bool] = (c_lo <= arrays.control) & (arrays.control < c_hi)
        lda = fit_lda(
            rfl1=arrays.rfl1[mask],
            rfl2=arrays.rfl2[mask],
            weight=arrays.weight[mask],
            lda0=90.0,
        ).x[0]
        lda0s.append(lda)
    return lda0s


@dataclass
class FactorizationResult:
    h0: FitResult
    h1s: list[FitResult]
    ndof: int

    @property
    def likelihood_ratio(self):
        return self.h0.n2ll - sum([h1.n2ll for h1 in self.h1s])

    @property
    def p(self) -> float:
        return float(chi2(self.ndof).sf(self.likelihood_ratio))


@dataclass
class SPlotFitResult:
    sigmc_fit_components: SigMCFitComponents
    yields: list[float]
    bkg_ldas: list[float]
    total_fit: FitResult
    v: FloatArray
    weighted_total: float

    @property
    def aic(self) -> float:
        return self.total_fit.aic

    @property
    def bic(self) -> float:
        return self.total_fit.bic

    @property
    def nbkg(self) -> int:
        return len(self.bkg_ldas)

    @property
    def sig_yield(self) -> float:
        return self.yields[0]

    @property
    def bkg_yields(self) -> list[float]:
        return self.yields[1:]

    def pdfs1(self, rfl1: FloatArray) -> list[FloatArray]:
        return [self.sigmc_fit_components.pdf1(rfl1)] + [
            exp_pdf_single(rfl=rfl1, lda=bkg_lda) for bkg_lda in self.bkg_ldas
        ]

    def pdfs2(self, rfl2: FloatArray) -> list[FloatArray]:
        return [self.sigmc_fit_components.pdf2(rfl2)] + [
            exp_pdf_single(rfl=rfl2, lda=bkg_lda) for bkg_lda in self.bkg_ldas
        ]

    def pdfs(self, rfl1: FloatArray, rfl2: FloatArray) -> list[FloatArray]:
        return [self.sigmc_fit_components.pdf(rfl1, rfl2)] + [
            exp_pdf(rfl1=rfl1, rfl2=rfl2, lda=bkg_lda) for bkg_lda in self.bkg_ldas
        ]


def add_m_meson(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        return (ks1_lab + ks2_lab).m

    return data.with_columns(
        pl.struct(
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_meson')
    )


def add_ksb_costheta(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        return min(ks1_com.vec3.costheta, ks2_com.vec3.costheta)

    return data.with_columns(
        pl.struct(
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('ksb_costheta')
    )


def add_m_baryon(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> float:
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        ksb_lab = ks1_lab if ks1_com.vec3.costheta < ks2_com.vec3.costheta else ks2_lab
        return (ksb_lab + p_lab).m

    return data.with_columns(
        pl.struct(
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(process, return_dtype=pl.Float64)
        .alias('m_baryon')
    )


def add_hx_angles(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> dict[str, float]:
        beam_px = struct['p4_0_Px']
        beam_py = struct['p4_0_Py']
        beam_pz = struct['p4_0_Pz']
        beam_e = struct['p4_0_E']

        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        ks1_px = struct['p4_2_Px']
        ks1_py = struct['p4_2_Py']
        ks1_pz = struct['p4_2_Pz']
        ks1_e = struct['p4_2_E']

        ks2_px = struct['p4_3_Px']
        ks2_py = struct['p4_3_Py']
        ks2_pz = struct['p4_3_Pz']
        ks2_e = struct['p4_3_E']

        beam_lab = ld.Vec4(beam_px, beam_py, beam_pz, beam_e)
        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        ks1_lab = ld.Vec4(ks1_px, ks1_py, ks1_pz, ks1_e)
        ks2_lab = ld.Vec4(ks2_px, ks2_py, ks2_pz, ks2_e)
        com_boost = p_lab + ks1_lab + ks2_lab
        beam_com = beam_lab.boost(-com_boost.beta)
        p_com = p_lab.boost(-com_boost.beta)
        ks1_com = ks1_lab.boost(-com_boost.beta)
        ks2_com = ks2_lab.boost(-com_boost.beta)
        event = ld.Event(p4s=[beam_com, p_com, ks1_com, ks2_com], aux=[], weight=1.0)
        angles_hx = ld.Angles(0, [1], [2], [2, 3], 'HX')
        return {
            'hx_costheta': angles_hx.costheta.value(event),
            'hx_phi': angles_hx.phi.value(event),
        }

    return data.with_columns(
        pl.struct(
            'p4_0_Px',
            'p4_0_Py',
            'p4_0_Pz',
            'p4_0_E',
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_2_Px',
            'p4_2_Py',
            'p4_2_Pz',
            'p4_2_E',
            'p4_3_Px',
            'p4_3_Py',
            'p4_3_Pz',
            'p4_3_E',
        )
        .map_elements(
            process,
            return_dtype=pl.Struct(
                {
                    'hx_costheta': pl.Float64,
                    'hx_phi': pl.Float64,
                }
            ),
        )
        .alias('hx_angles')
    ).unnest('hx_angles')


def add_alt_hypos(data: pl.LazyFrame) -> pl.LazyFrame:
    def process(struct) -> dict[str, float]:
        p_px = struct['p4_1_Px']
        p_py = struct['p4_1_Py']
        p_pz = struct['p4_1_Pz']
        p_e = struct['p4_1_E']

        piplus1_px = struct['p4_4_Px']
        piplus1_py = struct['p4_4_Py']
        piplus1_pz = struct['p4_4_Pz']
        piplus1_e = struct['p4_4_E']

        piminus1_px = struct['p4_5_Px']
        piminus1_py = struct['p4_5_Py']
        piminus1_pz = struct['p4_5_Pz']
        piminus1_e = struct['p4_5_E']

        piplus2_px = struct['p4_6_Px']
        piplus2_py = struct['p4_6_Py']
        piplus2_pz = struct['p4_6_Pz']
        piplus2_e = struct['p4_6_E']

        piminus2_px = struct['p4_7_Px']
        piminus2_py = struct['p4_7_Py']
        piminus2_pz = struct['p4_7_Pz']
        piminus2_e = struct['p4_7_E']

        p_lab = ld.Vec4(p_px, p_py, p_pz, p_e)
        piplus1_lab = ld.Vec4(piplus1_px, piplus1_py, piplus1_pz, piplus1_e)
        piminus1_lab = ld.Vec4(piminus1_px, piminus1_py, piminus1_pz, piminus1_e)
        piplus2_lab = ld.Vec4(piplus2_px, piplus2_py, piplus2_pz, piplus2_e)
        piminus2_lab = ld.Vec4(piminus2_px, piminus2_py, piminus2_pz, piminus2_e)
        return {
            'piplus1_piminus1_m': (piplus1_lab + piminus1_lab).m,
            'piplus2_piminus2_m': (piplus2_lab + piminus2_lab).m,
            'piplus1_piminus2_m': (piplus1_lab + piminus2_lab).m,
            'piplus2_piminus1_m': (piplus2_lab + piminus1_lab).m,
            'p_piplus1_m': (p_lab + piplus1_lab).m,
            'p_piplus2_m': (p_lab + piplus2_lab).m,
            'p_piminus1_m': (p_lab + piminus1_lab).m,
            'p_piminus2_m': (p_lab + piminus2_lab).m,
            'p_piplus1_piminus1_m': (p_lab + piplus1_lab + piminus1_lab).m,
            'p_piplus1_piminus2_m': (p_lab + piplus1_lab + piminus2_lab).m,
            'p_piplus2_piminus1_m': (p_lab + piplus2_lab + piminus1_lab).m,
            'p_piplus2_piminus2_m': (p_lab + piplus2_lab + piminus2_lab).m,
            'p_piminus1_piminus2_m': (p_lab + piminus1_lab + piminus2_lab).m,
        }

    return data.with_columns(
        pl.struct(
            'p4_1_Px',
            'p4_1_Py',
            'p4_1_Pz',
            'p4_1_E',
            'p4_4_Px',
            'p4_4_Py',
            'p4_4_Pz',
            'p4_4_E',
            'p4_5_Px',
            'p4_5_Py',
            'p4_5_Pz',
            'p4_5_E',
            'p4_6_Px',
            'p4_6_Py',
            'p4_6_Pz',
            'p4_6_E',
            'p4_7_Px',
            'p4_7_Py',
            'p4_7_Pz',
            'p4_7_E',
        )
        .map_elements(
            process,
            return_dtype=pl.Struct(
                {
                    'piplus1_piminus1_m': pl.Float64,
                    'piplus2_piminus2_m': pl.Float64,
                    'piplus1_piminus2_m': pl.Float64,
                    'piplus2_piminus1_m': pl.Float64,
                    'p_piplus1_m': pl.Float64,
                    'p_piplus2_m': pl.Float64,
                    'p_piminus1_m': pl.Float64,
                    'p_piminus2_m': pl.Float64,
                    'p_piplus1_piminus1_m': pl.Float64,
                    'p_piplus1_piminus2_m': pl.Float64,
                    'p_piplus2_piminus1_m': pl.Float64,
                    'p_piplus2_piminus2_m': pl.Float64,
                    'p_piminus1_piminus2_m': pl.Float64,
                }
            ),
        )
        .alias('m_alt_hypos')
    ).unnest('m_alt_hypos')


def to_latex(
    value: float, unc: float | None = None, unc_sys: float | None = None
) -> str:
    if unc is None:
        if np.isnan(value):
            return r' \textemdash '
        mantissa, exponent = f'{value:.2E}'.split('E')
        return f'${mantissa} \\times 10^{{{exponent}}}$'
    if value == 0.0 and unc == 0.0:
        return r'$0.0$ (fixed)'
    if np.isnan(value) or np.isnan(unc):
        return r' \textemdash '
    truncation = -int(np.floor(np.log10(abs(unc)))) + 1
    if unc_sys is not None:
        truncation_sys = -int(np.floor(np.log10(abs(unc_sys)))) + 1
        truncation = max(truncation, truncation_sys)
    unc_trunc = round(unc, truncation)
    val_trunc = round(value, truncation)
    ndigits = -truncation
    expo = int(np.floor(np.log10(abs(val_trunc if val_trunc != 0.0 else unc_trunc))))
    val_mantissa = val_trunc / 10**expo
    unc_mantissa = unc_trunc / 10**expo
    if unc_sys is not None:
        unc_sys_mantissa = unc_sys / 10**expo if unc_sys is not None else None
        return rf'$({val_mantissa:.{expo - ndigits}f} \pm {unc_mantissa:.{expo - ndigits}f} \pm {unc_sys_mantissa:.{expo - ndigits}f}) \times 10^{{{expo}}}$'
    return rf'$({val_mantissa:.{expo - ndigits}f} \pm {unc_mantissa:.{expo - ndigits}f}) \times 10^{{{expo}}}$'


def custom_colormap() -> tuple[ListedColormap, CenteredNorm]:
    n = 256
    pos = plt.get_cmap('afmhot_r', n)
    neg = plt.get_cmap('GnBu', n)

    n_neg = 128
    n_pos = 127
    n_white = 1

    pos_colors = pos(np.linspace(0.0, 1.0, n_pos))
    neg_colors = neg(np.linspace(0.0, 1.0, n_neg))[::-1]
    white = np.ones((n_white, 4))  # RGBA white

    colors = np.vstack([neg_colors, white, pos_colors])
    cmap = ListedColormap(colors)

    norm = CenteredNorm(vcenter=0.0)

    return cmap, norm


def get_ccdb_table(
    table_path: str, *, use_timestamp: bool = True
) -> dict[int, list[list[str]]]:
    with sqlite3.connect(str(MISC_PATH / 'ccdb.sqlite')) as ccdb:
        path_parts = table_path.split('/')
        cursor = ccdb.cursor()
        query = """
            SELECT tt.nColumns, rr.runMin, rr.runMax, cs.vault, a.created
            FROM directories d0
            """
        for i, path_part in enumerate(path_parts[1:-1]):
            query += f"JOIN directories d{i + 1} ON d{i + 1}.parentId = d{i}.id AND d{i + 1}.name = '{path_part}'"
        query += f"""
            JOIN typeTables tt ON d{len(path_parts) - 2}.id = tt.directoryId AND tt.name = '{path_parts[-1]}'
            JOIN constantSets cs ON tt.id = cs.constantTypeId
            JOIN assignments a ON cs.id = a.constantSetId
            JOIN runRanges rr ON a.runRangeId = rr.id
            LEFT JOIN variations v ON a.variationId = v.id
            WHERE d0.name = '{path_parts[0]}'
            AND v.name IS 'default'
            ORDER BY rr.runMin, a.created
            """
        cursor.execute(query)
        res = cursor.fetchall()
        res_table = {}
        for n_columns, run_min, run_max, vault, timestamp in res:
            # We do this because some of the tables have ridiculous run ranges
            run_min = max(run_min, RUN_RANGES['s17'][0])
            run_max = min(run_max, RUN_RANGES['s20'][1])
            ts = datetime.strptime(timestamp, '%Y-%m-%d %H:%M:%S')
            run_period = get_run_period_bound(run_min)
            if run_period is None:
                continue
            max_timestamp = REST_VERSION_TIMESTAMPS[run_period]
            if ts <= max_timestamp or use_timestamp is False:
                data = vault.split('|')
                table = [
                    data[i : i + n_columns] for i in range(0, len(data), n_columns)
                ]
                for run in range(run_min, run_max + 1):
                    res_table[run] = table
        return res_table


def get_rcdb_text_condition(condition: str) -> dict[int, str]:
    with sqlite3.connect(str(MISC_PATH / 'rcdb.sqlite')) as rcdb:
        cursor = rcdb.cursor()
        query = f"""
        {RCDB_SELECTION_PREFIX}
        SELECT r.number, c.text_value
        FROM conditions c
        JOIN condition_types ct ON c.condition_type_id = ct.id
        JOIN runs r ON c.run_number = r.number
        WHERE ct.name = '{condition}'
        {RCDB_SELECTION_SUFFIX}
        ORDER BY r.number
        """
        cursor.execute(query)
        res = cursor.fetchall()
        res_table = {}
        for run_number, value in res:
            res_table[run_number] = value
        return res_table


def get_all_polarized_run_numbers() -> list[int]:
    with sqlite3.connect(str(MISC_PATH / 'rcdb.sqlite')) as rcdb:
        cursor = rcdb.cursor()
        query = f"""
        {RCDB_SELECTION_PREFIX}
        SELECT r.number
        FROM runs r
        WHERE r.number > 0
        {RCDB_SELECTION_SUFFIX}
        ORDER BY r.number
        """
        print(query)
        cursor.execute(query)
        return [r[0] for r in cursor.fetchall()]


@dataclass
class PSFluxTable:
    df_scale: dict[int, float]
    df_ps_accept: dict[int, tuple[float, float, float]]
    df_photon_endpoint: dict[int, float]
    df_tagm_tagged_flux_table: dict[int, list[list[str]]]
    df_tagm_scaled_energy_table: dict[int, list[list[str]]]
    df_tagh_tagged_flux_table: dict[int, list[list[str]]]
    df_tagh_scaled_energy_table: dict[int, list[list[str]]]
    df_photon_endpoint_calib: dict[int, float]
    df_target_scattering_centers: dict[int, tuple[float, float]]


def get_coherent_peak(run_number: int) -> tuple[float, float]:
    if get_run_period(run_number) != 's20':
        return (8.2, 8.8)
    return (8.0, 8.6)
