import pickle
from typing import Literal, override
from modak import Task
import polars as pl
import matplotlib.pyplot as plt
import numpy as np

from gluex_ksks.constants import (
    BLACK,
    FITS_PATH,
    LOG_PATH,
    MESON_MASS_BINS,
    MESON_MASS_RANGE,
    NBOOT,
    NUM_THREADS,
    PLOTS_PATH,
    REPORTS_PATH,
    RUN_PERIODS,
)
from gluex_ksks.pwa import (
    BinnedFitResult,
    BinnedFitResultUncertainty,
    Binning,
    FullPathSet,
    GuidedFitResult,
    SinglePathSet,
    UnbinnedFitResult,
    UnbinnedFitResultUncertainty,
    calculate_bootstrap_uncertainty_binned,
    calculate_bootstrap_uncertainty_unbinned,
    fit_binned,
    fit_guided,
    fit_unbinned,
)
from gluex_ksks.tasks.cuts import FiducialCuts
from gluex_ksks.tasks.splot import SPlotWeights
from gluex_ksks.utils import select_mesons_tag, to_latex
from gluex_ksks.wave import Wave


class NarrowData(Task):
    def __init__(
        self,
        *,
        data_type: str,
        run_period: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        tag = select_mesons_tag(select_mesons)
        if method is None or nspec is None:
            inputs: list[Task] = [
                FiducialCuts(
                    data_type=data_type,
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                )
            ]
        elif data_type == 'data':
            assert method is not None
            assert nspec is not None
            inputs = [
                SPlotWeights(
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=method,
                    nspec=nspec,
                )
            ]
        else:
            raise ValueError('Unsupported data type + cut combination')
        outputs = [
            inputs[0].outputs[0].parent / f'{inputs[0].outputs[0].stem}_narrow.parquet'
        ]
        super().__init__(
            f'narrow_{data_type}_{run_period}_{protonz_cut}_{mass_cut}_{chisqdof}_{tag}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        pl.scan_parquet(self.inputs[0].outputs[0]).select(
            'p4_0_Px',
            'p4_0_Py',
            'p4_0_Pz',
            'p4_0_E',
            'aux_0_x',
            'aux_0_y',
            'aux_0_z',
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
            'weight',
        ).sink_parquet(self.outputs[0])


class BinnedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
    ):
        self.waves = waves
        self.nbins = nbins
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            *[
                NarrowData(
                    data_type='data',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=method,
                    nspec=nspec,
                )
                for run_period in RUN_PERIODS
            ],
            *[
                NarrowData(
                    data_type='sigmc',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=None,
                    nspec=None,
                )
                for run_period in RUN_PERIODS
            ],
        ]
        outputs = [
            FITS_PATH
            / f'binned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}.pkl',
        ]
        super().__init__(
            f'binned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}',
            inputs=inputs,
            outputs=outputs,
            resources={'fit': 1},
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info('Beginning Binned Fit')
        analysis_path_set = FullPathSet(
            *[
                SinglePathSet(data_path, accmc_path)
                for data_path, accmc_path in zip(
                    [self.inputs[i].outputs[0] for i in range(len(RUN_PERIODS))],
                    [
                        self.inputs[i + len(RUN_PERIODS)].outputs[0]
                        for i in range(len(RUN_PERIODS))
                    ],
                )
            ]
        )
        binning = Binning(self.nbins, MESON_MASS_RANGE)
        fit_result = fit_binned(
            self.waves,
            analysis_path_set,
            binning,
            iters=3,
            threads=NUM_THREADS,
            logger=self.logger,
        )
        pickle.dump(fit_result, self.outputs[0].open('wb'))


class BinnedFitUncertainty(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot: int,
    ):
        self.waves = waves
        self.nbins = nbins
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
            )
        ]
        outputs = [
            inputs[0].outputs[0].parent
            / f'{inputs[0].outputs[0].stem}_boot_{nboot}.pkl'
        ]
        super().__init__(
            f'binned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}',
            inputs=inputs,
            outputs=outputs,
            resources={'fit': 1},
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        binned_fit_result: BinnedFitResult = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        result = calculate_bootstrap_uncertainty_binned(
            binned_fit_result, nboot=self.nboot, threads=NUM_THREADS, logger=self.logger
        )
        result.get_lower_center_upper(bootstrap_mode='SE')
        result.get_lower_center_upper(bootstrap_mode='CI')
        result.get_lower_center_upper(bootstrap_mode='CI-BC')
        pickle.dump(result, self.outputs[0].open('wb'))


class PlotBinnedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot: int,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'],
    ):
        self.waves = waves
        self.nbins = nbins
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot,
            )
        ]
        outputs = [PLOTS_PATH / (inputs[0].outputs[0].stem + f'_{bootstrap_mode}.png')]
        super().__init__(
            f'binned_fit_plot{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}_{bootstrap_mode}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        data_hist = binned_fit_result.fit_result.get_data_histogram()
        fit_hists = binned_fit_result.fit_result.get_histograms()
        fit_error_bars = binned_fit_result.get_error_bars()
        plt.style.use('gluex_ksks.thesis')
        if Wave.needs_full_plot(self.waves):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        continue
                    ax[i][j].stairs(
                        data_hist.counts,
                        data_hist.bins,
                        color=BLACK,
                        label='Data',
                    )
                    fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                    err = fit_error_bars[Wave.encode_waves(self.waves)]
                    centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                    ax[i][j].errorbar(
                        centers,
                        fit_hist.counts,
                        yerr=0,
                        fmt='.',
                        markersize=3,
                        color=BLACK,
                        label='Fit Total',
                    )
                    ax[i][j].errorbar(
                        centers,
                        err[1],
                        yerr=(err[0], err[2]),
                        fmt='none',
                        color=BLACK,
                    )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                err = fit_error_bars[Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        latex_group = Wave.get_latex_group_at_index((i, j))
                        ax[i][j].text(
                            0.5,
                            0.5,
                            f'No {latex_group}',
                            ha='center',
                            va='center',
                            transform=ax[i][j].transAxes,
                        )
                    else:
                        ax[i][j].legend()
                        ax[i][j].set_ylim(0)
        else:
            fig, ax = plt.subplots(ncols=2, sharey=True)
            for i in {0, 1}:
                ax[i].stairs(
                    data_hist.counts,
                    data_hist.bins,
                    color=BLACK,
                    label='Data',
                )
                fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                err = fit_error_bars[Wave.encode_waves(self.waves)]
                centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                ax[i].errorbar(
                    centers,
                    fit_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=BLACK,
                    label='Fit Total',
                )
                ax[i].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=BLACK,
                )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                err = fit_error_bars[Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
            ax[0].legend()
            ax[1].legend()
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[0])
        plt.close()


class BinnedFitReport(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot: int,
    ):
        self.chisqdof = chisqdof
        self.waves = waves
        self.nbins = nbins
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot,
            )
        ]
        outputs = [REPORTS_PATH / (inputs[0].outputs[0].stem + '.tex')]
        super().__init__(
            f'binned_fit_report{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        output_str = r"""\begin{center}
    \begin{longtable}{clrrr}\toprule
        Bin (GeV/$c^2$) & Wave & Real & Imaginary & Total ($\abs{F}^2$) \\\midrule
        \endhead"""
        statuses = binned_fit_result.fit_result.statuses
        edges = binned_fit_result.fit_result.binning.edges
        for ibin in range(len(statuses)):
            bin_status = statuses[ibin]
            last_bin = ibin == len(statuses) - 1
            bin_edges = rf'{edges[ibin]:.3f}\textendash {edges[ibin + 1]:.3f}'
            for iwave, wave in enumerate(self.waves):
                last_wave = iwave == len(self.waves) - 1
                coefficient_name = wave.coefficient_name
                i_re = binned_fit_result.fit_result.model.parameters.index(
                    f'{coefficient_name} real'
                )
                c_re = bin_status.x[i_re]
                e_re = float(
                    np.std(
                        [
                            binned_fit_result.samples[ibin][j][i_re]
                            for j in range(len(binned_fit_result.samples[ibin]))
                        ],
                        ddof=1,
                    )
                )
                if wave.l == 0:
                    c_im = 0.0
                    e_im = 0.0
                    c_tot = c_re**2
                    e_tot = float(
                        np.std(
                            [
                                binned_fit_result.samples[ibin][j][i_re] ** 2
                                for j in range(len(binned_fit_result.samples[ibin]))
                            ],
                            ddof=1,
                        )
                    )
                else:
                    i_im = binned_fit_result.fit_result.model.parameters.index(
                        f'{coefficient_name} imag'
                    )
                    c_im = bin_status.x[i_im]
                    e_im = float(
                        np.std(
                            [
                                binned_fit_result.samples[ibin][j][i_im]
                                for j in range(len(binned_fit_result.samples[ibin]))
                            ],
                            ddof=1,
                        )
                    )
                    c_tot = c_re**2 + c_im**2
                    e_tot = float(
                        np.std(
                            [
                                binned_fit_result.samples[ibin][j][i_re] ** 2
                                + binned_fit_result.samples[ibin][j][i_im] ** 2
                                for j in range(len(binned_fit_result.samples[ibin]))
                            ],
                            ddof=1,
                        )
                    )
                output_str += rf"""
        {bin_edges if iwave == 0 else ''} & {wave.latex} & {to_latex(c_re, e_re)} & {to_latex(c_im, e_im)} & {to_latex(c_tot, e_tot)} \\*"""
                if last_wave:
                    if last_bin:
                        output_str += r'\bottomrule'
                    else:
                        output_str += r'\midrule'
        output_str += rf"""
    \caption{{The parameter values and uncertainties for the binned fit of {Wave.to_latex_string(self.waves)} waves to data with $\chi^2_\nu < {self.chisqdof:.2f}$. Uncertainties are calculated from the standard error over ${self.nboot}$ bootstrap iterations.}}\label{{tab:binned-fit-chisqdof-{self.chisqdof:.2f}-{Wave.to_kebab_string(self.waves)}}}
    \end{{longtable}}
\end{{center}}
% NLL = {sum([status.fx for status in binned_fit_result.fit_result.statuses])}
"""
        self.outputs[0].write_text(output_str)


class GuidedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot: int,
    ):
        self.waves = waves
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot,
            )
        ]
        outputs = [
            FITS_PATH
            / f'guided_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}.pkl',
        ]
        super().__init__(
            f'guided_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}',
            inputs=inputs,
            outputs=outputs,
            resources={'fit': 1},
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info('Beginning Guided Fit')
        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        fit_result = fit_guided(
            binned_fit_result,
            bootstrap_mode='SE',
            iters=3,
            threads=NUM_THREADS,
            logger=self.logger,
        )
        pickle.dump(fit_result, self.outputs[0].open('wb'))


class PlotGuidedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot: int,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'],
    ):
        self.waves = waves
        self.nbins = nbins
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            GuidedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot,
            ),
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot,
            ),
        ]
        outputs = [PLOTS_PATH / (inputs[0].outputs[0].stem + f'_{bootstrap_mode}.png')]
        super().__init__(
            f'guided_fit_plot{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot}_{bootstrap_mode}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        guided_fit_result: GuidedFitResult = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            self.inputs[1].outputs[0].open('rb')
        )
        data_hist = binned_fit_result.fit_result.get_data_histogram()
        fit_hists = binned_fit_result.fit_result.get_histograms()
        fit_error_bars = binned_fit_result.get_error_bars()
        unbinned_fit_hists = guided_fit_result.fit_result.get_histograms(
            binned_fit_result.fit_result.binning
        )
        plt.style.use('gluex_ksks.thesis')
        if Wave.needs_full_plot(self.waves):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        continue
                    ax[i][j].stairs(
                        data_hist.counts,
                        data_hist.bins,
                        color=BLACK,
                        label='Data',
                    )
                    fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                    unbinned_fit_hist = unbinned_fit_hists[
                        Wave.encode_waves(self.waves)
                    ]
                    err = fit_error_bars[Wave.encode_waves(self.waves)]
                    centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                    ax[i][j].errorbar(
                        centers,
                        fit_hist.counts,
                        yerr=0,
                        fmt='.',
                        markersize=3,
                        color=BLACK,
                        label='Fit Total',
                    )
                    ax[i][j].errorbar(
                        centers,
                        err[1],
                        yerr=(err[0], err[2]),
                        fmt='none',
                        color=BLACK,
                    )
                    ax[i][j].stairs(
                        unbinned_fit_hist.counts,
                        unbinned_fit_hist.bins,
                        color=BLACK,
                        label='Fit (Guided)',
                        fill=True,
                        alpha=0.2,
                    )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                unbinned_wave_hist = unbinned_fit_hists[Wave.encode(wave)]
                err = fit_error_bars[Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
                ax[plot_index[0]][plot_index[1]].stairs(
                    unbinned_wave_hist.counts,
                    unbinned_wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Guided)',
                    fill=True,
                    alpha=0.2,
                )
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        latex_group = Wave.get_latex_group_at_index((i, j))
                        ax[i][j].text(
                            0.5,
                            0.5,
                            f'No {latex_group}',
                            ha='center',
                            va='center',
                            transform=ax[i][j].transAxes,
                        )
                    else:
                        ax[i][j].legend()
                        ax[i][j].set_ylim(0)
        else:
            fig, ax = plt.subplots(ncols=2, sharey=True)
            for i in {0, 1}:
                ax[i].stairs(
                    data_hist.counts,
                    data_hist.bins,
                    color=BLACK,
                    label='Data',
                )
                fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                unbinned_fit_hist = unbinned_fit_hists[Wave.encode_waves(self.waves)]
                err = fit_error_bars[Wave.encode_waves(self.waves)]
                centers = (fit_hist.bins[1:] + fit_hist.bins[:-1]) / 2
                ax[i].errorbar(
                    centers,
                    fit_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=BLACK,
                    label='Fit Total',
                )
                ax[i].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=BLACK,
                )
                ax[i].stairs(
                    unbinned_fit_hist.counts,
                    unbinned_fit_hist.bins,
                    color=BLACK,
                    label='Fit (Guided)',
                    fill=True,
                    alpha=0.2,
                )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                unbinned_wave_hist = unbinned_fit_hists[Wave.encode(wave)]
                err = fit_error_bars[Wave.encode(wave)]
                centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
                ax[wave.plot_index(double=True)[0]].stairs(
                    unbinned_wave_hist.counts,
                    unbinned_wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Guided)',
                    fill=True,
                    alpha=0.2,
                )
            ax[0].legend()
            ax[1].legend()
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[0])
        plt.close()


class UnbinnedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins_guided: int,
        nboot_guided: int,
        guided: bool,
    ):
        self.waves = waves
        self.guided = guided
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            *[
                NarrowData(
                    data_type='data',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=method,
                    nspec=nspec,
                )
                for run_period in RUN_PERIODS
            ],
            *[
                NarrowData(
                    data_type='sigmc',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=None,
                    nspec=None,
                )
                for run_period in RUN_PERIODS
            ],
        ]
        if self.guided:
            inputs.append(
                GuidedFit(
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                    method=method,
                    nspec=nspec,
                    waves=waves,
                    nbins=nbins_guided,
                    nboot=nboot_guided,
                )
            )
        outputs = [
            FITS_PATH
            / f'unbinned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins_guided}_boot_{nboot_guided}{"_guided" if guided else ""}.pkl',
        ]
        super().__init__(
            f'unbinned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins_guided}_boot_{nboot_guided}{"_guided" if guided else ""}',
            inputs=inputs,
            outputs=outputs,
            resources={'fit': 1},
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info(f'Beginning Unbinned Fit{" (guided)" if self.guided else ""}')
        analysis_path_set = FullPathSet(
            *[
                SinglePathSet(data_path, accmc_path)
                for data_path, accmc_path in zip(
                    [self.inputs[i].outputs[0] for i in range(len(RUN_PERIODS))],
                    [
                        self.inputs[i + len(RUN_PERIODS)].outputs[0]
                        for i in range(len(RUN_PERIODS))
                    ],
                )
            ]
        )
        p0 = None
        if self.guided:
            guided_result: GuidedFitResult = pickle.load(
                self.inputs[-1].outputs[0].open('rb')
            )
            p0 = guided_result.fit_result.status.x
        fit_result = fit_unbinned(
            self.waves,
            analysis_path_set,
            p0=p0,
            iters=3,
            threads=NUM_THREADS,
            logger=self.logger,
        )
        pickle.dump(fit_result, self.outputs[0].open('wb'))


class UnbinnedFitUncertainty(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins_guided: int,
        nboot_guided: int,
        guided: bool,
        nboot: int,
    ):
        self.waves = waves
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            UnbinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=nbins_guided,
                nboot_guided=nboot_guided,
                guided=guided,
            )
        ]
        outputs = [
            inputs[0].outputs[0].parent
            / f'{inputs[0].outputs[0].stem}_boot_{nboot}.pkl'
        ]
        super().__init__(
            f'unbinned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins_guided}_{nboot_guided}_{guided}_boot_{nboot}',
            inputs=inputs,
            outputs=outputs,
            resources={'fit': 1},
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        result = calculate_bootstrap_uncertainty_unbinned(
            unbinned_fit_result,
            nboot=self.nboot,
            threads=NUM_THREADS,
            logger=self.logger,
        )
        pickle.dump(result, self.outputs[0].open('wb'))


class PlotUnbinnedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot_guided: int,
        guided: bool,
    ):
        self.waves = waves
        self.nbins = nbins
        self.guided = guided
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            UnbinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=nbins,
                nboot_guided=nboot_guided,
                guided=guided,
            )
        ]
        outputs = [PLOTS_PATH / (inputs[0].outputs[0].stem + '.png')]
        super().__init__(
            f'unbinned_fit_plot{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot_guided}_{guided}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        binning = Binning(self.nbins, MESON_MASS_RANGE)
        data_hist = unbinned_fit_result.get_data_histogram(binning)
        fit_hists = unbinned_fit_result.get_histograms(binning)
        plt.style.use('gluex_ksks.thesis')
        if Wave.needs_full_plot(self.waves):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        continue
                    ax[i][j].stairs(
                        data_hist.counts,
                        data_hist.bins,
                        color=BLACK,
                        label='Data',
                    )
                    fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                    ax[i][j].stairs(
                        fit_hist.counts,
                        fit_hist.bins,
                        color=BLACK,
                        label=f'Fit (Unbinned{", Guided" if self.guided else ""})',
                        fill=True,
                        alpha=0.2,
                    )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].stairs(
                    wave_hist.counts,
                    wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        latex_group = Wave.get_latex_group_at_index((i, j))
                        ax[i][j].text(
                            0.5,
                            0.5,
                            f'No {latex_group}',
                            ha='center',
                            va='center',
                            transform=ax[i][j].transAxes,
                        )
                    else:
                        ax[i][j].legend()
                        ax[i][j].set_ylim(0)
        else:
            fig, ax = plt.subplots(ncols=2, sharey=True)
            for i in {0, 1}:
                ax[i].stairs(
                    data_hist.counts,
                    data_hist.bins,
                    color=BLACK,
                    label='Data',
                )
                fit_hist = fit_hists[Wave.encode_waves(self.waves)]
                ax[i].stairs(
                    fit_hist.counts,
                    fit_hist.bins,
                    color=BLACK,
                    label='Fit (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            for wave in self.waves:
                wave_hist = fit_hists[Wave.encode(wave)]
                ax[wave.plot_index(double=True)[0]].stairs(
                    wave_hist.counts,
                    wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            ax[0].legend()
            ax[1].legend()
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[0])
        plt.close()


class UnbinnedFitReport(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins_guided: int,
        nboot_guided: int,
        guided: bool,
        nboot: int,
    ):
        self.chisqdof = chisqdof
        self.waves = waves
        self.guided = guided
        self.nboot = nboot
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            UnbinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=nbins_guided,
                nboot_guided=nboot_guided,
                guided=guided,
                nboot=nboot,
            )
        ]
        outputs = [REPORTS_PATH / (inputs[0].outputs[0].stem + '.tex')]
        super().__init__(
            f'unbinned_fit_report{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins_guided}_{nboot_guided}_{guided}_{nboot}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        unbinned_fit_result: UnbinnedFitResultUncertainty = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        output_str = r"""\begin{table}[ht]
    \begin{center}
        \begin{tabular}{llrrr}\toprule
        Wave & Resonance & Real & Imaginary & Total ($\abs{F}^2$) \\\midrule"""
        model = unbinned_fit_result.fit_result.model
        status = unbinned_fit_result.fit_result.status
        latest_wave = None
        for i, parameter in enumerate(model.parameters):
            if parameter.endswith('real'):
                parameter_real = parameter
                parameter_imag = parameter_real.replace('real', 'imag')
                latex_res_name, latex_wave_name = (
                    Wave.kmatrix_parameter_name_to_latex_parts(parameter_real)
                )
                value_real = status.x[i]
                unc_real = float(
                    np.std(
                        [
                            unbinned_fit_result.samples[j][i]
                            for j in range(len(unbinned_fit_result.samples))
                        ],
                        ddof=1,
                    )
                )
                if parameter_imag in model.parameters:
                    value_imag = status.x[i + 1]
                    unc_imag = float(
                        np.std(
                            [
                                unbinned_fit_result.samples[j][i + 1]
                                for j in range(len(unbinned_fit_result.samples))
                            ],
                            ddof=1,
                        )
                    )
                    total_mag = value_real**2 + value_imag**2
                    total_mag_unc = float(
                        np.std(
                            [
                                unbinned_fit_result.samples[j][i] ** 2
                                + unbinned_fit_result.samples[j][i + 1] ** 2
                                for j in range(len(unbinned_fit_result.samples))
                            ],
                            ddof=1,
                        )
                    )
                else:
                    value_imag = 0.0
                    unc_imag = 0.0
                    total_mag = value_real**2
                    total_mag_unc = float(
                        np.std(
                            [
                                unbinned_fit_result.samples[j][i] ** 2
                                for j in range(len(unbinned_fit_result.samples))
                            ],
                            ddof=1,
                        )
                    )
                if latex_wave_name == latest_wave:
                    wave = ''
                else:
                    wave = latex_wave_name
                    latest_wave = latex_wave_name
                output_str += f'\n{wave} & {latex_res_name} & {to_latex(value_real, unc_real)} & {to_latex(value_imag, unc_imag)} & {to_latex(total_mag, total_mag_unc)} \\\\'

        output_str += rf"""\bottomrule
        \end{{tabular}}
    \caption{{The parameter values and uncertainties for the unbinned {'(guided) ' if self.guided else ''}fit of {Wave.to_latex_string(self.waves)} waves to data with $\chi^2_\nu < {self.chisqdof:.2f}$. Uncertainties are calculated from the standard error over ${self.nboot}$ bootstrap iterations.}}\label{{tab:unbinned-fit-chisqdof-{self.chisqdof:.1f}{'-guided' if self.guided else ''}-{Wave.to_kebab_string(self.waves)}}}
    \end{{center}}
\end{{table}}
% NLL = {status.fx}
"""
        self.outputs[0].write_text(output_str)


class PlotUnbinnedAndBinnedFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
        nbins: int,
        nboot_guided: int,
        guided: bool,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'],
    ):
        self.waves = waves
        self.nbins = nbins
        self.guided = guided
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=nbins,
                nboot=nboot_guided,
            ),
            UnbinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=nbins,
                nboot_guided=nboot_guided,
                guided=guided,
            ),
        ]
        outputs = [
            PLOTS_PATH
            / f'binned_and_unbinned_fit{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot_guided}{"_guided" if guided else ""}_{bootstrap_mode}.png',
        ]
        super().__init__(
            f'binned_and_unbinned_fit_plot{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{nbins}_boot_{nboot_guided}_{guided}_{bootstrap_mode}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        binned_fit_result: BinnedFitResultUncertainty = pickle.load(
            self.inputs[0].outputs[0].open('rb')
        )
        unbinned_fit_result: UnbinnedFitResult = pickle.load(
            self.inputs[1].outputs[0].open('rb')
        )
        data_hist = binned_fit_result.fit_result.get_data_histogram()
        binned_fit_hists = binned_fit_result.fit_result.get_histograms()
        binned_fit_error_bars = binned_fit_result.get_error_bars()
        unbinned_fit_hists = unbinned_fit_result.get_histograms(
            binned_fit_result.fit_result.binning
        )
        plt.style.use('gluex_ksks.thesis')
        if Wave.needs_full_plot(self.waves):
            fig, ax = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=True)
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        continue
                    ax[i][j].stairs(
                        data_hist.counts,
                        data_hist.bins,
                        color=BLACK,
                        label='Data',
                    )
                    binned_fit_hist = binned_fit_hists[Wave.encode_waves(self.waves)]
                    err = binned_fit_error_bars[Wave.encode_waves(self.waves)]
                    centers = (binned_fit_hist.bins[1:] + binned_fit_hist.bins[:-1]) / 2
                    ax[i][j].errorbar(
                        centers,
                        binned_fit_hist.counts,
                        yerr=0,
                        fmt='.',
                        markersize=3,
                        color=BLACK,
                        label='Fit Total',
                    )
                    ax[i][j].errorbar(
                        centers,
                        err[1],
                        yerr=(err[0], err[2]),
                        fmt='none',
                        color=BLACK,
                    )
                    unbinned_fit_hist = unbinned_fit_hists[
                        Wave.encode_waves(self.waves)
                    ]
                    ax[i][j].stairs(
                        unbinned_fit_hist.counts,
                        unbinned_fit_hist.bins,
                        color=BLACK,
                        label='Fit (Unbinned{", Guided" if self.guided else ""})',
                        fill=True,
                        alpha=0.2,
                    )
            for wave in self.waves:
                binned_wave_hist = binned_fit_hists[Wave.encode(wave)]
                err = binned_fit_error_bars[Wave.encode(wave)]
                centers = (binned_wave_hist.bins[1:] + binned_wave_hist.bins[:-1]) / 2
                plot_index = wave.plot_index(double=False)
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    binned_wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[plot_index[0]][plot_index[1]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
                unbinned_wave_hist = unbinned_fit_hists[Wave.encode(wave)]
                ax[plot_index[0]][plot_index[1]].stairs(
                    unbinned_wave_hist.counts,
                    unbinned_wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            for i in {0, 1}:
                for j in {0, 1, 2}:
                    if not Wave.has_wave_at_index(self.waves, (i, j)):
                        latex_group = Wave.get_latex_group_at_index((i, j))
                        ax[i][j].text(
                            0.5,
                            0.5,
                            f'No {latex_group}',
                            ha='center',
                            va='center',
                            transform=ax[i][j].transAxes,
                        )
                    else:
                        ax[i][j].legend()
                        ax[i][j].set_ylim(0)
        else:
            fig, ax = plt.subplots(ncols=2, sharey=True)
            for i in {0, 1}:
                ax[i].stairs(
                    data_hist.counts,
                    data_hist.bins,
                    color=BLACK,
                    label='Data',
                )
                binned_fit_hist = binned_fit_hists[Wave.encode_waves(self.waves)]
                err = binned_fit_error_bars[Wave.encode_waves(self.waves)]
                centers = (binned_fit_hist.bins[1:] + binned_fit_hist.bins[:-1]) / 2
                ax[i].errorbar(
                    centers,
                    binned_fit_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=BLACK,
                    label='Fit Total',
                )
                ax[i].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=BLACK,
                )
                unbinned_fit_hist = unbinned_fit_hists[Wave.encode_waves(self.waves)]
                ax[i].stairs(
                    unbinned_fit_hist.counts,
                    unbinned_fit_hist.bins,
                    color=BLACK,
                    label='Fit (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            for wave in self.waves:
                binned_wave_hist = binned_fit_hists[Wave.encode(wave)]
                err = binned_fit_error_bars[Wave.encode(wave)]
                centers = (binned_wave_hist.bins[1:] + binned_wave_hist.bins[:-1]) / 2
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    binned_wave_hist.counts,
                    yerr=0,
                    fmt='.',
                    markersize=3,
                    color=wave.plot_color,
                    label=wave.latex,
                )
                ax[wave.plot_index(double=True)[0]].errorbar(
                    centers,
                    err[1],
                    yerr=(err[0], err[2]),
                    fmt='none',
                    color=wave.plot_color,
                )
                unbinned_wave_hist = unbinned_fit_hists[Wave.encode(wave)]
                ax[wave.plot_index(double=True)[0]].stairs(
                    unbinned_wave_hist.counts,
                    unbinned_wave_hist.bins,
                    color=wave.plot_color,
                    label=f'{wave.latex} (Unbinned{", Guided" if self.guided else ""})',
                    fill=True,
                    alpha=0.2,
                )
            ax[0].legend()
            ax[1].legend()
            ax[0].set_ylim(0)
            ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[0])
        plt.close()


class ProcessBinned(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
    ):
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs = [
            PlotBinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot=NBOOT,
                bootstrap_mode='CI-BC',
            ),
            BinnedFitReport(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot=NBOOT,
            ),
        ]
        outputs = []
        super().__init__(
            f'process_binned_fits{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    def run(self):
        pass


class ProcessUnbinned(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        waves: list[Wave],
    ):
        wave_string = Wave.encode_waves(waves)
        tag = select_mesons_tag(select_mesons)
        inputs = [
            PlotBinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot=NBOOT,
                bootstrap_mode='CI-BC',
            ),
            BinnedFitReport(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot=NBOOT,
            ),
            PlotUnbinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=False,
            ),
            PlotGuidedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot=NBOOT,
                bootstrap_mode='CI-BC',
            ),
            PlotUnbinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=True,
            ),
            UnbinnedFitReport(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=False,
                nboot=NBOOT,
            ),
            UnbinnedFitReport(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins_guided=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=True,
                nboot=NBOOT,
            ),
            PlotUnbinnedAndBinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=False,
                bootstrap_mode='CI-BC',
            ),
            PlotUnbinnedAndBinnedFit(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=waves,
                nbins=MESON_MASS_BINS,
                nboot_guided=NBOOT,
                guided=True,
                bootstrap_mode='CI-BC',
            ),
        ]
        outputs = []
        super().__init__(
            f'process_unbinned_fits{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    def run(self):
        pass
