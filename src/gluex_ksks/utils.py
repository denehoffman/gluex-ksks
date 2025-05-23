from __future__ import annotations

import pickle
import laddu as ld
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, overload

import numpy as np
import polars as pl
import uproot
from scipy.stats import chi2
from scipy.optimize import OptimizeResult, minimize
from uproot.behaviors.TBranch import HasBranches

from gluex_ksks.constants import MISC_PATH, RFL_RANGE, RUN_RANGES, TRUE_POL_ANGLES
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
    keys = list(root_data.keys())
    for key in keys:
        if key.startswith('P4_'):
            root_data[key.replace('P4_', 'p4_')] = root_data.pop(key)
        if key.startswith('Weight'):
            root_data[key.replace('Weight', 'weight')] = root_data.pop(key)
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
        relative_beam_bucket = int(np.floor(rf / 4.008016032) + 0.5)
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
        return weight * (-scale / 8.0)


@dataclass
class Histogram:
    counts: NDArray[np.floating]
    bins: NDArray[np.floating]

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
        return Histogram(counts, bins)


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
        magnitude: float = pol_hist.counts[energy_index]
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


def get_run_period(run_number: int) -> str | None:
    for rp, (lo, hi) in RUN_RANGES.items():
        if lo <= run_number <= hi:
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
    def from_polars(data: pl.DataFrame, *, control: str) -> SPlotArrays:
        return SPlotArrays(
            rfl1=data['RFL1'].to_numpy(),
            rfl2=data['RFL2'].to_numpy(),
            weight=data['weight'].to_numpy(),
            control=data[control].to_numpy(),
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
    nbins: int,
) -> SigMCFitComponents:
    hist1 = Histogram(
        *np.histogram(
            arrays.rfl1,
            bins=nbins,
            range=RFL_RANGE,
            weights=arrays.weight,
            density=True,
        )
    )
    hist2 = Histogram(
        *np.histogram(
            arrays.rfl2,
            bins=nbins,
            range=RFL_RANGE,
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
        return self.h0.aic - sum([h1.aic for h1 in self.h1s])

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


@overload
def add_m_meson(data: pl.DataFrame) -> pl.DataFrame: ...


@overload
def add_m_meson(data: pl.LazyFrame) -> pl.LazyFrame: ...


def add_m_meson(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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


@overload
def add_ksb_costheta(data: pl.DataFrame) -> pl.DataFrame: ...


@overload
def add_ksb_costheta(data: pl.LazyFrame) -> pl.LazyFrame: ...


def add_ksb_costheta(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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


@overload
def add_m_baryon(data: pl.DataFrame) -> pl.DataFrame: ...


@overload
def add_m_baryon(data: pl.LazyFrame) -> pl.LazyFrame: ...


def add_m_baryon(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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


@overload
def add_hx_angles(data: pl.DataFrame) -> pl.DataFrame: ...


@overload
def add_hx_angles(data: pl.LazyFrame) -> pl.LazyFrame: ...


def add_hx_angles(data: pl.DataFrame | pl.LazyFrame) -> pl.DataFrame | pl.LazyFrame:
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
