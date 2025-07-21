from abc import ABC, abstractmethod
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import Literal, override

import laddu as ld
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from scipy.stats import norm

from gluex_ksks.constants import BLACK, GUIDED_MAX_STEPS, RED
from gluex_ksks.wave import Wave
from gluex_ksks.types import FloatArray
from gluex_ksks.utils import Histogram


@dataclass
class Binning:
    bins: int
    range: tuple[float, float]

    @property
    def edges(self) -> FloatArray:
        return np.linspace(*self.range, self.bins + 1, endpoint=True)

    @property
    def centers(self) -> FloatArray:
        return (self.edges[:-1] + self.edges[1:]) / 2


class PathSet(ABC):
    @property
    @abstractmethod
    def data_paths(self) -> list[Path]: ...

    @property
    @abstractmethod
    def accmc_paths(self) -> list[Path]: ...

    def get_data_datasets(
        self,
    ) -> list[ld.Dataset]:
        return [ld.open(path, rest_frame_indices=[1, 2, 3]) for path in self.data_paths]

    def get_data_datasets_binned(
        self,
        binning: Binning,
    ) -> list[ld.BinnedDataset]:
        datasets = self.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        return [
            dataset.bin_by(res_mass, binning.bins, binning.range)
            for dataset in datasets
        ]

    def get_accmc_datasets(
        self,
    ) -> list[ld.Dataset]:
        return [
            ld.open(path, rest_frame_indices=[1, 2, 3]) for path in self.accmc_paths
        ]

    def get_accmc_datasets_binned(
        self,
        binning: Binning,
    ) -> list[ld.BinnedDataset]:
        datasets = self.get_accmc_datasets()
        res_mass = ld.Mass([2, 3])
        return [
            dataset.bin_by(res_mass, binning.bins, binning.range)
            for dataset in datasets
        ]


@dataclass
class SinglePathSet(PathSet):
    data: Path
    accmc: Path

    @property
    @override
    def data_paths(self) -> list[Path]:
        return [self.data]

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return [self.accmc]


@dataclass
class FullPathSet(PathSet):
    s17: SinglePathSet
    s18: SinglePathSet
    f18: SinglePathSet
    s20: SinglePathSet

    @property
    @override
    def data_paths(self) -> list[Path]:
        return (
            self.s17.data_paths
            + self.s18.data_paths
            + self.f18.data_paths
            + self.s20.data_paths
        )

    @property
    @override
    def accmc_paths(self) -> list[Path]:
        return (
            self.s17.accmc_paths
            + self.s18.accmc_paths
            + self.f18.accmc_paths
            + self.s20.accmc_paths
        )


class LoggingObserver(ld.Observer):
    def __init__(self, logger):
        self.logger = logger

    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        self.logger.info(f'Step {step}: {status.fx} {status.x}')
        return status, False


def add_parameter_text(
    fig: Figure,
    param_names: list[str],
    param_values: FloatArray,
    prev_values: FloatArray | None,
):
    num_params = len(param_names)
    min_font_size, max_font_size = 8, 14
    min_spacing, max_spacing = 0.015, 0.04
    font_size = np.max([min_font_size, np.min([max_font_size, 14 - 0.2 * num_params])])
    y_step = max(min_spacing, min(max_spacing, 0.85 / num_params))
    differences = (
        param_values - prev_values
        if prev_values is not None
        else np.zeros_like(param_values)
    )
    y_start = 0.95
    fig.subplots_adjust(right=0.7)
    ax_text = fig.add_axes((0.72, 0.1, 0.25, 0.8))
    ax_text.axis('off')
    for i, (name, value, diff) in enumerate(
        zip(param_names, param_values, differences)
    ):
        y_pos = y_start - i * y_step
        color = 'blue' if diff > 0 else 'red'
        arrow = '↑' if diff > 0 else '↓'
        ax_text.text(
            0,
            y_pos,
            f'{name:10}',
            fontsize=font_size,
            verticalalignment='center',
        )
        ax_text.text(
            0.3,
            y_pos,
            f'{value:.4f}',
            fontsize=font_size,
            verticalalignment='center',
        )
        ax_text.text(
            0.6,
            y_pos,
            f'{diff:+.4f} {arrow}',
            fontsize=font_size,
            verticalalignment='center',
            color=color,
        )


class GuidedLoggingObserver(ld.Observer):
    def __init__(
        self,
        masses: list[FloatArray],
        n_accmc_weighted: list[float],
        nlls: list[ld.NLL],
        wavesets: list[list[Wave]],
        histograms: dict[int, Histogram],
        error_bars: dict[
            int,
            tuple[
                FloatArray,
                FloatArray,
                FloatArray,
            ],
        ],
        *,
        binning: Binning,
        phase_factor: bool,
        ndof: int,
        logger,
    ):
        self.masses: list[FloatArray] = masses
        self.n_accmc_weighted: list[float] = n_accmc_weighted
        self.n_accmc_tot: float = sum(self.n_accmc_weighted)
        self.nlls: list[ld.NLL] = nlls
        self.wavesets: list[list[Wave]] = wavesets
        self.histograms: dict[int, Histogram] = histograms
        self.error_bars: dict[
            int,
            tuple[
                FloatArray,
                FloatArray,
                FloatArray,
            ],
        ] = error_bars
        self.binning: Binning = binning
        self.phase_factor: bool = phase_factor
        self.ndof: int = ndof
        self.previous_x: FloatArray | None = None
        self.logger = logger

    @override
    def callback(self, step: int, status: ld.Status) -> tuple[ld.Status, bool]:
        if step % 20 == 0:
            self.logger.info(f'Step {step}: {status.fx} ({status.fx / self.ndof})')
        fig, ax = plt.subplots(
            nrows=len(self.nlls) + 1,
            ncols=len(self.wavesets),
            sharex=True,
            figsize=(6 * len(self.wavesets) + 6, 4 * len(self.nlls) + 4),
        )
        fit_histograms: dict[int, list[Histogram]] = {}
        for waveset in self.wavesets:
            fit_histograms[Wave.encode_waves(waveset)] = [
                Histogram(
                    *np.histogram(
                        self.masses[i],
                        weights=self.nlls[i].project_with(
                            status.x,
                            Wave.get_waveset_names(
                                waveset,
                                mass_dependent=True,
                                phase_factor=self.phase_factor,
                            ),
                        ),
                        bins=self.binning.edges,
                    )
                )
                for i in range(len(self.nlls))
            ]
        for idataset in range(len(self.nlls)):
            for iwaveset in range(len(self.wavesets)):
                waveset = Wave.encode_waves(self.wavesets[iwaveset])
                fit_hist = fit_histograms[waveset][idataset]
                wave_hist = self.histograms[waveset]
                wave_error_bars = self.error_bars[waveset]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                ax[idataset][iwaveset].plot(
                    centers,
                    wave_hist.counts
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    marker='.',
                    linestyle='none',
                    color=BLACK,
                )
                ax[idataset][iwaveset].errorbar(
                    centers,
                    wave_error_bars[1]
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    yerr=(
                        wave_error_bars[0]
                        * self.n_accmc_weighted[idataset]
                        / self.n_accmc_tot,
                        wave_error_bars[2]
                        * self.n_accmc_weighted[idataset]
                        / self.n_accmc_tot,
                    ),
                    fmt='none',
                    color=BLACK,
                )
                ax[idataset][iwaveset].stairs(
                    fit_hist.counts,
                    fit_hist.bins,
                    color=BLACK,
                    fill=True,
                    alpha=0.2,
                )
                ax[idataset][iwaveset].stairs(
                    fit_hist.counts,
                    fit_hist.bins,
                    baseline=wave_hist.counts
                    * self.n_accmc_weighted[idataset]
                    / self.n_accmc_tot,
                    fill=True,
                    color=RED,
                    alpha=0.2,
                )
        itot = len(self.nlls)
        for iwaveset in range(len(self.wavesets)):
            waveset = Wave.encode_waves(self.wavesets[iwaveset])
            fit_counts = np.sum(
                [
                    fit_histograms[waveset][idataset].counts
                    for idataset in range(len(self.nlls))
                ],
                axis=0,
            )
            fit_bins = fit_histograms[waveset][0].bins
            wave_hist = self.histograms[waveset]
            wave_error_bars = self.error_bars[waveset]
            centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
            ax[itot][iwaveset].plot(
                centers,
                wave_hist.counts,
                marker='.',
                linestyle='none',
                color=BLACK,
            )
            ax[itot][iwaveset].errorbar(
                centers,
                wave_error_bars[1],
                yerr=(wave_error_bars[0], wave_error_bars[2]),
                fmt='none',
                color=BLACK,
            )
            ax[itot][iwaveset].stairs(
                fit_counts,
                fit_bins,
                color=BLACK,
                fill=True,
                alpha=0.2,
            )
            ax[itot][iwaveset].stairs(
                fit_counts,
                fit_bins,
                baseline=wave_hist.counts,
                fill=True,
                color=RED,
                alpha=0.2,
            )
        add_parameter_text(fig, self.nlls[0].parameters, status.x, self.previous_x)
        plt.savefig('guided_fit.svg')
        plt.close()
        self.previous_x = status.x
        if status.fx / self.ndof <= 1.0:
            return status, True
        return status, False


@dataclass
class BinnedFitResult:
    statuses: list[ld.Status]
    waves: list[Wave]
    model: ld.Model
    paths: PathSet
    binning: Binning
    phase_factor: bool
    data_hist_cache: Histogram | None = None
    fit_histograms_cache: dict[int, Histogram] | None = None

    def get_data_histogram(self) -> Histogram:
        if data_hist := self.data_hist_cache:
            return data_hist
        data_datasets = self.paths.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        values = np.concatenate(
            [res_mass.value_on(dataset) for dataset in data_datasets]
        )
        weights = np.concatenate([dataset.weights for dataset in data_datasets])
        data_hist = Histogram(
            *np.histogram(
                values,
                bins=self.binning.bins,
                range=self.binning.range,
                weights=weights,
            ),
            np.sqrt(
                np.histogram(
                    values,
                    bins=self.binning.bins,
                    range=self.binning.range,
                    weights=np.power(weights, 2),
                )[0]
            ),
        )
        self.data_hist_cache = data_hist
        return data_hist

    def get_histograms(
        self,
    ) -> dict[int, Histogram]:
        if fit_histograms := self.fit_histograms_cache:
            return fit_histograms
        data_datasets = self.paths.get_data_datasets_binned(self.binning)
        accmc_datasets = self.paths.get_accmc_datasets_binned(self.binning)
        wavesets = Wave.power_set(self.waves)
        counts: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        edges = self.binning.edges
        for ibin, status in enumerate(self.statuses):
            nlls = [
                ld.NLL(
                    self.model,
                    ds_data[ibin],
                    ds_accmc[ibin],
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
            for waveset in wavesets:
                counts[Wave.encode_waves(waveset)].append(
                    np.sum(
                        np.concatenate(
                            [
                                nll.project_with(
                                    status.x,
                                    Wave.get_waveset_names(
                                        waveset,
                                        mass_dependent=False,
                                        phase_factor=self.phase_factor,
                                    ),
                                )
                                for nll in nlls
                            ]
                        )
                    )
                )
        fit_hists = {
            Wave.encode_waves(waveset): Histogram(
                np.array(counts[Wave.encode_waves(waveset)]), edges
            )
            for waveset in wavesets
        }
        self.fit_histograms_cache = fit_hists
        return fit_hists


def fit_binned(
    waves: list[Wave],
    paths: PathSet,
    binning: Binning,
    *,
    iters: int,
    phase_factor: bool = True,
    threads: int,
    logger,
) -> BinnedFitResult:
    data_datasets = paths.get_data_datasets_binned(binning)
    accmc_datasets = paths.get_accmc_datasets_binned(binning)
    model = Wave.get_model(waves, mass_dependent=False, phase_factor=phase_factor)
    statuses: list[ld.Status] = []
    for ibin in range(binning.bins):
        manager = ld.LikelihoodManager()
        bin_model = ld.likelihood_sum(
            [
                manager.register(ld.NLL(model, ds_data[ibin], ds_accmc[ibin]).as_term())
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
        )
        nll = manager.load(bin_model)
        best_nll = np.inf
        best_status = None
        rng = np.random.default_rng(0)
        init_mag = 1000.0
        for _ in range(iters):
            p_init = rng.uniform(-init_mag, init_mag, len(nll.parameters))
            status = nll.minimize(
                p_init,
                threads=threads,
                skip_hessian=True,
            )
            if status.converged:
                if status.fx < best_nll:
                    best_nll = status.fx
                    best_status = status
        if best_status is None:
            raise Exception('No fit converged')
        best_status_with_hessian = nll.minimize(
            best_status.x,
            threads=threads,
        )
        statuses.append(best_status_with_hessian)
    return BinnedFitResult(statuses, waves, model, paths, binning, phase_factor)


@dataclass
class UnbinnedFitResult:
    status: ld.Status
    waves: list[Wave]
    model: ld.Model
    paths: PathSet
    phase_factor: bool
    data_hist_cache: Histogram | None = None
    fit_histograms_cache: dict[int, list[Histogram]] | None = None

    def get_data_histogram(self, binning: Binning) -> Histogram:
        if data_hist := self.data_hist_cache:
            return data_hist
        data_datasets = self.paths.get_data_datasets()
        res_mass = ld.Mass([2, 3])
        values = np.concatenate(
            [res_mass.value_on(dataset) for dataset in data_datasets]
        )
        weights = np.concatenate([dataset.weights for dataset in data_datasets])
        data_hist = Histogram(
            *np.histogram(
                values,
                weights=weights,
                bins=binning.edges,
            ),
            np.sqrt(
                np.histogram(
                    values,
                    weights=np.power(weights, 2),
                    bins=binning.edges,
                )[0]
            ),
        )
        self.data_hist_cache = data_hist
        return data_hist

    def get_histograms_by_run_period(
        self, binning: Binning
    ) -> dict[int, list[Histogram]]:
        if fit_histograms := self.fit_histograms_cache:
            return fit_histograms
        data_datasets = self.paths.get_data_datasets()
        accmc_datasets = self.paths.get_accmc_datasets()
        wavesets = Wave.power_set(self.waves)
        histograms: dict[int, list[Histogram]] = {}
        nlls = [
            ld.NLL(
                self.model,
                ds_data,
                ds_accmc,
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
        res_mass = ld.Mass([2, 3])
        for waveset in wavesets:
            histograms[Wave.encode_waves(waveset)] = [
                Histogram(
                    *np.histogram(
                        res_mass.value_on(accmc_datasets[i]),
                        weights=nlls[i].project_with(
                            self.status.x,
                            Wave.get_waveset_names(
                                waveset,
                                mass_dependent=True,
                                phase_factor=self.phase_factor,
                            ),
                        ),
                        bins=binning.edges,
                    )
                )
                for i in range(len(accmc_datasets))
            ]
        self.fit_histograms_cache = histograms
        return histograms

    def get_histograms(
        self,
        binning: Binning,
    ) -> dict[int, Histogram]:
        if fit_histograms := self.fit_histograms_cache:
            hists = {
                wave: Histogram.sum(hists) for wave, hists in fit_histograms.items()
            }
            return {wave: hist for wave, hist in hists.items() if hist is not None}
        data_datasets = self.paths.get_data_datasets()
        accmc_datasets = self.paths.get_accmc_datasets()
        wavesets = Wave.power_set(self.waves)
        histograms: dict[int, Histogram] = {}
        nlls = [
            ld.NLL(
                self.model,
                ds_data,
                ds_accmc,
            )
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
        res_mass = ld.Mass([2, 3])
        for waveset in wavesets:
            histograms[Wave.encode_waves(waveset)] = Histogram(
                *np.histogram(
                    np.concatenate(
                        [
                            res_mass.value_on(accmc_dataset)
                            for accmc_dataset in accmc_datasets
                        ]
                    ),
                    weights=np.concatenate(
                        [
                            nll.project_with(
                                self.status.x,
                                Wave.get_waveset_names(
                                    waveset,
                                    mass_dependent=True,
                                    phase_factor=self.phase_factor,
                                ),
                            )
                            for nll in nlls
                        ]
                    ),
                    bins=binning.edges,
                )
            )
        return histograms


def fit_unbinned(
    waves: list[Wave],
    paths: PathSet,
    *,
    p0: FloatArray | None = None,
    iters: int,
    phase_factor: bool = True,
    threads: int,
    logger,
) -> UnbinnedFitResult:
    data_datasets = paths.get_data_datasets()
    accmc_datasets = paths.get_accmc_datasets()
    model = Wave.get_model(waves, mass_dependent=True, phase_factor=phase_factor)
    manager = ld.LikelihoodManager()
    likelihood_model = ld.likelihood_sum(
        [
            manager.register(ld.NLL(model, ds_data, ds_accmc).as_term())
            for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
        ]
    )
    nll = manager.load(likelihood_model)
    best_nll = np.inf
    best_status = None
    rng = np.random.default_rng(0)
    init_mag = 1000.0
    for _ in range(iters):
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-init_mag, init_mag, len(nll.parameters))
        )
        status = nll.minimize(
            [float(p) for p in p_init],
            threads=threads,
            skip_hessian=True,
        )
        if status.converged:
            if status.fx < best_nll:
                best_nll = status.fx
                best_status = status
    if best_status is None:
        raise Exception('No fit converged')
    best_status_with_hessian = nll.minimize(
        best_status.x,
        threads=threads,
    )
    return UnbinnedFitResult(
        best_status_with_hessian, waves, model, paths, phase_factor
    )


@dataclass
class BinnedFitResultUncertainty:
    samples: list[list[FloatArray]]
    fit_result: BinnedFitResult
    _: KW_ONLY
    lcu_cache: (
        dict[
            int,
            dict[
                str,
                dict[
                    int,
                    tuple[
                        FloatArray,
                        FloatArray,
                        FloatArray,
                    ],
                ],
            ],
        ]
        | None
    ) = None

    def get_error_bars(
        self,
        *,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
        confidence_percent: int = 90,
    ) -> dict[
        int,
        tuple[FloatArray, FloatArray, FloatArray],
    ]:
        lcu = self.get_lower_center_upper(
            bootstrap_mode=bootstrap_mode,
            confidence_percent=confidence_percent,
        )
        error_bars: dict[
            int,
            tuple[FloatArray, FloatArray, FloatArray],
        ] = {}
        for wave, wave_lcu in lcu.items():
            yerr = (
                (wave_lcu[2] - wave_lcu[0]) / 2,
                (wave_lcu[2] + wave_lcu[0]) / 2,
                (wave_lcu[2] - wave_lcu[0]) / 2,
            )  # symmetric, prevents issues with wave_lcu[0] < 0
            error_bars[wave] = yerr
        return error_bars

    def get_lower_center_upper(
        self,
        *,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
        confidence_percent: int = 90,
    ) -> dict[
        int,
        tuple[FloatArray, FloatArray, FloatArray],
    ]:
        if (
            self.lcu_cache is not None
            and (confidence_cache := self.lcu_cache.get(confidence_percent)) is not None
            and (cache := confidence_cache.get(bootstrap_mode)) is not None
        ):
            return cache
        else:
            self.fill_cache(confidence_percent=confidence_percent)
            if (
                self.lcu_cache is not None
                and (confidence_cache := self.lcu_cache.get(bootstrap_mode)) is not None
                and (cache := confidence_cache.get(bootstrap_mode)) is not None
            ):
                return cache
            else:
                raise RuntimeError(
                    f'No cache found for mode {bootstrap_mode} at {confidence_percent}% confidence'
                )

    def fill_cache(
        self,
        *,
        confidence_percent: int = 90,
    ):
        data_datasets = self.fit_result.paths.get_data_datasets_binned(
            self.fit_result.binning
        )
        accmc_datasets = self.fit_result.paths.get_accmc_datasets_binned(
            self.fit_result.binning
        )
        wavesets = Wave.power_set(self.fit_result.waves)
        lower_quantile_se: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        center_quantile_se: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        upper_quantile_se: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        lower_quantile_ci: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        center_quantile_ci: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        upper_quantile_ci: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        lower_quantile_ci_bc: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        center_quantile_ci_bc: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        upper_quantile_ci_bc: dict[int, list[float]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        fit_histograms = self.fit_result.get_histograms()
        for ibin in range(self.fit_result.binning.bins):
            intensities_in_bin: dict[int, list[float]] = {
                Wave.encode_waves(waveset): [] for waveset in wavesets
            }
            for isample, sample in enumerate(self.samples[ibin]):
                nlls = [
                    ld.NLL(
                        self.fit_result.model,
                        ds_data[ibin].bootstrap(isample),
                        ds_accmc[ibin],
                    )
                    for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
                ]
                for waveset in wavesets:
                    intensities_in_bin[Wave.encode_waves(waveset)].append(
                        np.sum(
                            np.concatenate(
                                [
                                    nll.project_with(
                                        [float(p) for p in sample],
                                        Wave.get_waveset_names(
                                            waveset,
                                            mass_dependent=False,
                                            phase_factor=self.fit_result.phase_factor,
                                        ),
                                    )
                                    for nll in nlls
                                ]
                            )
                        )
                    )
            for waveset in wavesets:
                a_lo = (1 - confidence_percent / 100) / 2
                a_hi = 1 - a_lo
                quantiles = np.quantile(
                    intensities_in_bin[Wave.encode_waves(waveset)],
                    [a_lo, 0.5, a_hi],
                )
                lower_quantile_ci[Wave.encode_waves(waveset)].append(quantiles[0])
                center_quantile_ci[Wave.encode_waves(waveset)].append(quantiles[1])
                upper_quantile_ci[Wave.encode_waves(waveset)].append(quantiles[2])
                fit_value = fit_histograms[Wave.encode_waves(waveset)].counts[ibin]
                n_b = len(self.samples[ibin])
                phi = norm().cdf
                phi_inv = norm().ppf

                intensities = np.array(intensities_in_bin[Wave.encode_waves(waveset)])
                bootstrap_mean = intensities.mean(axis=0)

                cdf_b = (
                    np.sum(intensities < fit_value) / n_b
                    if fit_value < bootstrap_mean
                    else np.sum(intensities <= fit_value) / n_b
                )

                a = (1 - confidence_percent / 100) / 2
                z_a_lo = phi_inv(a)
                z_a_hi = phi_inv(1 - a)
                z_0 = phi_inv(cdf_b)
                a_lo = phi(2 * z_0 + z_a_lo)
                a_hi = phi(2 * z_0 + z_a_hi)

                quantiles = np.quantile(
                    intensities_in_bin[Wave.encode_waves(waveset)],
                    [a_lo, 0.5, a_hi],
                )
                lower_quantile_ci_bc[Wave.encode_waves(waveset)].append(quantiles[0])
                center_quantile_ci_bc[Wave.encode_waves(waveset)].append(quantiles[1])
                upper_quantile_ci_bc[Wave.encode_waves(waveset)].append(quantiles[2])
                wave_intensities_in_bin = intensities_in_bin[Wave.encode_waves(waveset)]
                std_err = np.std(wave_intensities_in_bin, ddof=1)
                quantiles = np.array(
                    [fit_value - std_err, fit_value, fit_value + std_err],
                    dtype=np.float64,
                )
                lower_quantile_se[Wave.encode_waves(waveset)].append(quantiles[0])
                center_quantile_se[Wave.encode_waves(waveset)].append(quantiles[1])
                upper_quantile_se[Wave.encode_waves(waveset)].append(quantiles[2])
        lcu_se = {
            Wave.encode_waves(waveset): (
                np.array(lower_quantile_se[Wave.encode_waves(waveset)]),
                np.array(center_quantile_se[Wave.encode_waves(waveset)]),
                np.array(upper_quantile_se[Wave.encode_waves(waveset)]),
            )
            for waveset in wavesets
        }
        lcu_ci = {
            Wave.encode_waves(waveset): (
                np.array(lower_quantile_ci[Wave.encode_waves(waveset)]),
                np.array(center_quantile_ci[Wave.encode_waves(waveset)]),
                np.array(upper_quantile_ci[Wave.encode_waves(waveset)]),
            )
            for waveset in wavesets
        }
        lcu_ci_bc = {
            Wave.encode_waves(waveset): (
                np.array(lower_quantile_ci_bc[Wave.encode_waves(waveset)]),
                np.array(center_quantile_ci_bc[Wave.encode_waves(waveset)]),
                np.array(upper_quantile_ci_bc[Wave.encode_waves(waveset)]),
            )
            for waveset in wavesets
        }
        if self.lcu_cache is None:
            self.lcu_cache = {}
        self.lcu_cache[confidence_percent] = {
            'SE': lcu_se,
            'CI': lcu_ci,
            'CI-BC': lcu_ci_bc,
        }


def calculate_bootstrap_uncertainty_binned(
    fit_result: BinnedFitResult,
    *,
    nboot: int,
    threads: int,
    logger,
) -> BinnedFitResultUncertainty:
    data_datasets = fit_result.paths.get_data_datasets_binned(fit_result.binning)
    accmc_datasets = fit_result.paths.get_accmc_datasets_binned(fit_result.binning)
    samples: list[list[FloatArray]] = []
    for ibin in range(fit_result.binning.bins):
        logger.info(f'Bootstrapping {ibin=}')
        bin_samples: list[FloatArray] = []
        for iboot in range(nboot):
            if iboot % 10 == 0:
                logger.info(f'Bootstrapping iteration {iboot}')
            manager = ld.LikelihoodManager()
            bin_model = ld.likelihood_sum(
                [
                    manager.register(
                        ld.NLL(
                            fit_result.model,
                            ds_data[ibin].bootstrap(iboot),
                            ds_accmc[ibin],
                        ).as_term()
                    )
                    for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
                ]
            )
            nll = manager.load(bin_model)
            status = nll.minimize(
                fit_result.statuses[ibin].x,
                threads=threads,
                skip_hessian=True,
            )
            if status.converged:
                bin_samples.append(status.x)
        samples.append(bin_samples)
    return BinnedFitResultUncertainty(
        samples,
        fit_result,
        uncertainty='bootstrap',
    )


@dataclass
class UnbinnedFitResultUncertainty:
    samples: list[FloatArray]
    fit_result: UnbinnedFitResult
    histogram_cache: dict[int, list[Histogram]] | None = None
    lcu_cache: (
        dict[
            int,
            dict[
                str,
                dict[
                    int,
                    tuple[
                        FloatArray,
                        FloatArray,
                        FloatArray,
                    ],
                ],
            ],
        ]
        | None
    ) = None

    def get_lower_center_upper(
        self,
        binning: Binning,
        *,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'CI-BC',
        confidence_percent: int = 90,
    ) -> dict[
        int,
        tuple[FloatArray, FloatArray, FloatArray],
    ]:
        if (
            self.lcu_cache is not None
            and (confidence_cache := self.lcu_cache.get(confidence_percent)) is not None
            and (cache := confidence_cache.get(bootstrap_mode)) is not None
        ):
            return cache
        else:
            self.fill_cache(binning, confidence_percent=confidence_percent)
            if (
                self.lcu_cache is not None
                and (confidence_cache := self.lcu_cache.get(confidence_percent))
                is not None
                and (cache := confidence_cache.get(bootstrap_mode)) is not None
            ):
                return cache
            else:
                raise RuntimeError(
                    f'No cache found for mode {bootstrap_mode} at {confidence_percent}% confidence'
                )

    def get_bootstrap_histograms(self, binning: Binning) -> dict[int, list[Histogram]]:
        if self.histogram_cache is not None:
            return self.histogram_cache
        data_datasets = self.fit_result.paths.get_data_datasets()
        accmc_datasets = self.fit_result.paths.get_accmc_datasets()
        wavesets = Wave.power_set(self.fit_result.waves)
        res_mass = ld.Mass([2, 3])
        bootstrapped_histograms: dict[int, list[Histogram]] = {
            Wave.encode_waves(waveset): [] for waveset in wavesets
        }
        for isample, sample in enumerate(self.samples):
            nlls = [
                ld.NLL(
                    self.fit_result.model,
                    ds_data.bootstrap(isample),
                    ds_accmc,
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
            for waveset in wavesets:
                bootstrapped_histograms[Wave.encode_waves(waveset)].append(
                    Histogram(
                        *np.histogram(
                            np.concatenate(
                                [
                                    res_mass.value_on(accmc_dataset)
                                    for accmc_dataset in accmc_datasets
                                ]
                            ),
                            weights=np.concatenate(
                                [
                                    nll.project_with(
                                        sample,
                                        Wave.get_waveset_names(
                                            waveset,
                                            mass_dependent=True,
                                            phase_factor=self.fit_result.phase_factor,
                                        ),
                                    )
                                    for nll in nlls
                                ]
                            ),
                            bins=binning.edges,
                        )
                    )
                )
        self.histogram_cache = bootstrapped_histograms
        return self.histogram_cache

    def fill_cache(
        self,
        binning: Binning,
        *,
        confidence_percent: int = 90,
    ):
        wavesets = Wave.power_set(self.fit_result.waves)
        lcu_se: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}
        lcu_ci: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}
        lcu_ci_bc: dict[int, tuple[FloatArray, FloatArray, FloatArray]] = {}
        fit_histograms = self.fit_result.get_histograms(binning)
        bootstrap_histograms = self.get_bootstrap_histograms(binning)
        for waveset in wavesets:
            intensities = np.array(
                [
                    histogram.counts
                    for histogram in bootstrap_histograms[Wave.encode_waves(waveset)]
                ]
            )
            a_lo = (1 - confidence_percent / 100) / 2
            a_hi = 1 - a_lo
            quantiles = np.quantile(intensities, [a_lo, 0.5, a_hi], axis=0)
            lcu_ci[Wave.encode_waves(waveset)] = (
                quantiles[0],
                quantiles[1],
                quantiles[2],
            )
            fit_values = fit_histograms[Wave.encode_waves(waveset)].counts
            n_b = len(self.samples)
            phi = norm().cdf
            phi_inv = norm().ppf

            bootstrap_means = intensities.mean(axis=0)

            lt_mask = fit_values > bootstrap_means
            le_mask = ~lt_mask
            cdfs = np.zeros_like(fit_values, dtype=float)
            cdfs[lt_mask] = (
                np.sum(intensities[:, lt_mask] < fit_values[lt_mask], axis=0) / n_b
            )
            cdfs[le_mask] = (
                np.sum(intensities[:, le_mask] <= fit_values[le_mask], axis=0) / n_b
            )

            a = (1 - confidence_percent / 100) / 2
            z_a_lo = phi_inv(a)
            z_a_hi = phi_inv(1 - a)
            z_0s = phi_inv(cdfs)
            a_los = phi(2 * z_0s + z_a_lo)
            a_his = phi(2 * z_0s + z_a_hi)

            quantiles = np.array(
                [
                    np.quantile(
                        intensities[:, ibin],
                        [a_los[ibin], 0.5, a_his[ibin]],
                    )
                    for ibin in range(len(intensities[0]))
                ]
            ).T
            lcu_ci_bc[Wave.encode_waves(waveset)] = (
                quantiles[0],
                quantiles[1],
                quantiles[2],
            )
            std_errs = np.std(intensities, ddof=1, axis=0)
            quantiles = np.array(
                [fit_values - std_errs, fit_values, fit_values + std_errs],
                dtype=np.float64,
            )
            lcu_se[Wave.encode_waves(waveset)] = (
                quantiles[0],
                quantiles[1],
                quantiles[2],
            )
        if self.lcu_cache is None:
            self.lcu_cache = {}
        self.lcu_cache[confidence_percent] = {
            'SE': lcu_se,
            'CI': lcu_ci,
            'CI-BC': lcu_ci_bc,
        }


def calculate_bootstrap_uncertainty_unbinned(
    fit_result: UnbinnedFitResult,
    *,
    nboot: int,
    threads: int,
    logger,
) -> UnbinnedFitResultUncertainty:
    data_datasets = fit_result.paths.get_data_datasets()
    accmc_datasets = fit_result.paths.get_accmc_datasets()
    samples: list[FloatArray] = []
    logger.info('Bootstrapping Unbinned fit')
    for iboot in range(nboot):
        if iboot % 10 == 0:
            logger.info(f'Bootstrapping iteration {iboot}')
        manager = ld.LikelihoodManager()
        bin_model = ld.likelihood_sum(
            [
                manager.register(
                    ld.NLL(
                        fit_result.model,
                        ds_data.bootstrap(iboot),
                        ds_accmc,
                    ).as_term()
                )
                for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
            ]
        )
        nll = manager.load(bin_model)
        status = nll.minimize(
            fit_result.status.x,
            threads=threads,
            skip_hessian=True,
        )
        if status.converged:
            samples.append(status.x)
    return UnbinnedFitResultUncertainty(
        samples,
        fit_result,
    )


@dataclass
class GuidedFitResult:
    binned_fit_result: BinnedFitResultUncertainty
    fit_result: UnbinnedFitResult


def fit_guided(
    binned_fit_result_uncertainty: BinnedFitResultUncertainty,
    *,
    p0: FloatArray | None = None,
    bootstrap_mode: Literal['SE', 'CI', 'CI-BC'] | str = 'SE',
    iters: int,
    threads: int,
    logger,
) -> GuidedFitResult:
    logger.info('Starting Guided Fit')
    waves = binned_fit_result_uncertainty.fit_result.waves
    binning = binned_fit_result_uncertainty.fit_result.binning
    phase_factor = binned_fit_result_uncertainty.fit_result.phase_factor
    paths = binned_fit_result_uncertainty.fit_result.paths
    model = Wave.get_model(waves, mass_dependent=True, phase_factor=phase_factor)
    data_datasets = binned_fit_result_uncertainty.fit_result.paths.get_data_datasets()
    accmc_datasets = binned_fit_result_uncertainty.fit_result.paths.get_accmc_datasets()
    n_accmc_tot = sum(
        [accmc_dataset.n_events_weighted for accmc_dataset in accmc_datasets]
    )
    histograms = binned_fit_result_uncertainty.fit_result.get_histograms()
    res_mass = ld.Mass([2, 3])
    manager = ld.LikelihoodManager()
    wavesets = Wave.power_set(waves)
    error_sets = None
    quantiles = binned_fit_result_uncertainty.get_lower_center_upper(
        bootstrap_mode=bootstrap_mode
    )
    error_bars = binned_fit_result_uncertainty.get_error_bars(
        bootstrap_mode=bootstrap_mode
    )
    error_sets = [
        [
            (
                quantiles[Wave.encode_waves(waveset)][2]
                - quantiles[Wave.encode_waves(waveset)][0]
            )
            / 2
            * accmc_dataset.n_events_weighted
            / n_accmc_tot
            for waveset in wavesets
        ]
        for accmc_dataset in accmc_datasets
    ]
    nlls = [
        ld.NLL(model, ds_data, ds_accmc)
        for ds_data, ds_accmc in zip(data_datasets, accmc_datasets)
    ]
    nlls_clone = nlls[::]
    masses: list[FloatArray] = [
        res_mass.value_on(ds_accmc) for ds_accmc in accmc_datasets
    ]
    likelihood_model = ld.likelihood_sum(
        [
            manager.register(
                ld.experimental.BinnedGuideTerm(
                    nlls[i],
                    res_mass,
                    amplitude_sets=[
                        Wave.get_waveset_names(
                            waveset,
                            mass_dependent=True,
                            phase_factor=phase_factor,
                        )
                        for waveset in wavesets
                    ],
                    bins=binning.bins,
                    range=binning.range,
                    count_sets=[
                        histograms[Wave.encode_waves(waveset)].counts
                        * accmc_dataset.n_events_weighted
                        / n_accmc_tot
                        for waveset in wavesets
                    ],
                    error_sets=error_sets[i],
                )
            )
            for i, accmc_dataset in enumerate(accmc_datasets)
        ]
    )
    nll = manager.load(likelihood_model)
    ndof = binning.bins * len(wavesets) - len(nll.parameters)
    best_nll = np.inf
    best_status = None
    rng = np.random.default_rng(0)
    init_mag = 1000.0
    for _ in range(iters):
        p_init = (
            p0
            if p0 is not None
            else rng.uniform(-init_mag, init_mag, len(nll.parameters))
        )
        status = nll.minimize(
            [float(p) for p in p_init],
            observers=GuidedLoggingObserver(
                masses,
                [accmc_dataset.n_events_weighted for accmc_dataset in accmc_datasets],
                nlls_clone,
                wavesets,
                histograms,
                error_bars,
                phase_factor=phase_factor,
                binning=binning,
                ndof=ndof,
                logger=logger,
            ),
            threads=threads,
            max_steps=GUIDED_MAX_STEPS,
            skip_hessian=True,
        )
        if status.fx < best_nll:
            best_nll = status.fx
            best_status = status
    if best_status is None:
        raise Exception("'best_status' is None, this should be unreachable!")
    return GuidedFitResult(
        binned_fit_result_uncertainty,
        UnbinnedFitResult(
            best_status,
            waves,
            model,
            paths,
            phase_factor,
        ),
    )
