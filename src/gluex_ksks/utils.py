from __future__ import annotations

import pickle
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import polars as pl
import uproot
from uproot.behaviors.TBranch import HasBranches

from gluex_ksks.constants import MISC_PATH, RUN_RANGES, TRUE_POL_ANGLES

if TYPE_CHECKING:
    from pathlib import Path

    from iminuit import Minuit
    from numpy.typing import NDArray


def root_to_parquet(input_path: Path, output_path: Path, tree: str = 'kin'):
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
    dataframe = pl.from_dict(root_data)
    dataframe.write_parquet(output_path)


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
    n_parameters: int
    n_events: int
    values: dict[str, float]
    errors: dict[str, float]

    @staticmethod
    def from_minuit(minuit: Minuit, n_events: int) -> FitResult:
        assert minuit.fval is not None  # noqa: S101
        return FitResult(
            minuit.fval,
            minuit.nfit,
            n_events,
            minuit.values.to_dict(),
            minuit.errors.to_dict(),
        )

    @property
    def aic(self) -> float:
        return 2.0 * self.n_parameters + self.n2ll  # 2k + -2ln(L)

    @property
    def bic(self) -> float:
        return self.n_parameters * np.log(self.n_events) + self.n2ll  # kln(n) + -2ln(L)


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
