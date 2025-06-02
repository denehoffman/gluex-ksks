import pickle
from typing import Literal, override
from modak import Task

import matplotlib.pyplot as plt
import numpy as np

from gluex_ksks.constants import (
    BLACK,
    LOG_PATH,
    MESON_MASS_RANGE,
    PINK,
    PLOTS_PATH,
    PURPLE,
)
from gluex_ksks.pwa import BinnedFitResultUncertainty
from gluex_ksks.tasks.fits import BinnedFitUncertainty
from gluex_ksks.utils import select_mesons_tag
from gluex_ksks.wave import Wave


CLAS_EDGES = np.arange(1.0, 1.90, 0.05)
CLAS_S_WAVE_FRAC = np.array(
    [
        1.0,
        1.0,
        0.973,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.928,
        0.903,
        0.803,
        0.791,
        0.762,
        0.660,
        0.690,
        0.845,
    ]
)
CLAS_S_WAVE_FRAC_ERR = np.array(
    [
        0.045,
        0.031,
        0.025,
        0.023,
        0.022,
        0.013,
        0.020,
        0.028,
        0.025,
        0.037,
        0.039,
        0.044,
        0.056,
        0.052,
        0.053,
        0.071,
        0.086,
    ]
)
CLAS_S_WAVE_FRAC_SIDEBAND = np.array(
    [
        1.0,
        1.0,
        0.982,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        0.922,
        0.890,
        0.879,
        0.897,
        0.883,
        0.910,
        0.902,
        0.941,
        0.994,
    ]
)
CLAS_S_WAVE_FRAC_ERR_SIDEBAND = np.array(
    [
        0.031,
        0.029,
        0.018,
        0.015,
        0.011,
        0.063,
        0.011,
        0.026,
        0.019,
        0.023,
        0.021,
        0.024,
        0.032,
        0.031,
        0.033,
        0.041,
        0.096,
    ]
)


class PlotCLASComparison(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        nboot: int,
        bootstrap_mode: Literal['SE', 'CI', 'CI-BC'],
    ):
        self.waves = [Wave(0, 0, '+'), Wave(2, 2, '+')]
        self.nbins = 20
        self.nboot = nboot
        wave_string = Wave.encode_waves(self.waves)
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            BinnedFitUncertainty(
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
                waves=self.waves,
                nbins=self.nbins,
                nboot=nboot,
            )
        ]
        outputs = [
            PLOTS_PATH
            / ('CLAS_both_' + inputs[0].outputs[0].stem + f'_{bootstrap_mode}.svg'),
            PLOTS_PATH
            / ('CLAS_peak_' + inputs[0].outputs[0].stem + f'_{bootstrap_mode}.svg'),
            PLOTS_PATH
            / ('CLAS_sideband_' + inputs[0].outputs[0].stem + f'_{bootstrap_mode}.svg'),
        ]
        super().__init__(
            f'clas_comparison_plot{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}_{wave_string}_{self.nbins}_boot_{nboot}_{bootstrap_mode}',
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
        clas_centers = (CLAS_EDGES[1:] + CLAS_EDGES[:-1]) / 2
        bin_width = clas_centers[1] - clas_centers[0]
        data_counts_clas = data_hist.counts[: len(CLAS_S_WAVE_FRAC)]
        data_errors_clas = (
            data_hist.errors[: len(CLAS_S_WAVE_FRAC)]
            if data_hist.errors is not None
            else np.zeros_like(data_counts_clas)
        )
        clas_s_wave_counts = data_counts_clas * CLAS_S_WAVE_FRAC
        clas_d_wave_counts = data_counts_clas * (1 - CLAS_S_WAVE_FRAC)
        clas_errors = np.sqrt(
            np.power(CLAS_S_WAVE_FRAC * data_errors_clas, 2)
            + np.power(data_counts_clas * CLAS_S_WAVE_FRAC_ERR, 2)
        )

        data_counts_clas_sideband = data_hist.counts[: len(CLAS_S_WAVE_FRAC_SIDEBAND)]
        data_errors_clas_sideband = (
            data_hist.errors[: len(CLAS_S_WAVE_FRAC_SIDEBAND)]
            if data_hist.errors is not None
            else np.zeros_like(data_counts_clas_sideband)
        )
        clas_s_wave_counts_sideband = (
            data_counts_clas_sideband * CLAS_S_WAVE_FRAC_SIDEBAND
        )
        clas_d_wave_counts_sideband = data_counts_clas_sideband * (
            1 - CLAS_S_WAVE_FRAC_SIDEBAND
        )
        clas_errors_sideband = np.sqrt(
            np.power(CLAS_S_WAVE_FRAC_SIDEBAND * data_errors_clas_sideband, 2)
            + np.power(data_counts_clas_sideband * CLAS_S_WAVE_FRAC_ERR_SIDEBAND, 2)
        )
        fit_hists = binned_fit_result.fit_result.get_histograms()
        fit_error_bars = binned_fit_result.get_error_bars()
        plt.style.use('gluex_ksks.thesis')

        # both
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
        # S-wave
        wave_hist = fit_hists[Wave.encode(self.waves[0])]
        err = fit_error_bars[Wave.encode(self.waves[0])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[0].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[0].plot_color,
            label=self.waves[0].latex,
        )
        ax[0].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[0].plot_color,
        )
        ax[0].errorbar(
            clas_centers,
            clas_s_wave_counts,
            yerr=clas_errors,
            fmt='v',
            markersize=1,
            color=PURPLE,
            label='CLAS S-Wave',
        )
        ax[0].errorbar(
            clas_centers,
            clas_s_wave_counts_sideband,
            yerr=clas_errors_sideband,
            fmt='^',
            markersize=1,
            color=PINK,
            label='CLAS S-Wave (sideband)',
        )

        # D-wave
        wave_hist = fit_hists[Wave.encode(self.waves[1])]
        err = fit_error_bars[Wave.encode(self.waves[1])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[1].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[1].plot_color,
            label=self.waves[1].latex,
        )
        ax[1].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[1].plot_color,
        )
        ax[1].errorbar(
            clas_centers,
            clas_d_wave_counts,
            yerr=clas_errors,
            fmt='v',
            markersize=1,
            color=PURPLE,
            label='CLAS D-Wave',
        )
        ax[1].errorbar(
            clas_centers,
            clas_d_wave_counts_sideband,
            yerr=clas_errors_sideband,
            fmt='^',
            markersize=1,
            color=PINK,
            label='CLAS D-Wave (sideband)',
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

        # peak
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
        # S-wave
        wave_hist = fit_hists[Wave.encode(self.waves[0])]
        err = fit_error_bars[Wave.encode(self.waves[0])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[0].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[0].plot_color,
            label=self.waves[0].latex,
        )
        ax[0].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[0].plot_color,
        )
        ax[0].errorbar(
            clas_centers,
            clas_s_wave_counts,
            yerr=clas_errors,
            fmt='v',
            markersize=1,
            color=PURPLE,
            label='CLAS S-Wave',
        )

        # D-wave
        wave_hist = fit_hists[Wave.encode(self.waves[1])]
        err = fit_error_bars[Wave.encode(self.waves[1])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[1].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[1].plot_color,
            label=self.waves[1].latex,
        )
        ax[1].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[1].plot_color,
        )
        ax[1].errorbar(
            clas_centers,
            clas_d_wave_counts,
            yerr=clas_errors,
            fmt='v',
            markersize=1,
            color=PURPLE,
            label='CLAS D-Wave',
        )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[1])
        plt.close()

        # sideband
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
        # S-wave
        wave_hist = fit_hists[Wave.encode(self.waves[0])]
        err = fit_error_bars[Wave.encode(self.waves[0])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[0].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[0].plot_color,
            label=self.waves[0].latex,
        )
        ax[0].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[0].plot_color,
        )
        ax[0].errorbar(
            clas_centers,
            clas_s_wave_counts_sideband,
            yerr=clas_errors_sideband,
            fmt='^',
            markersize=1,
            color=PINK,
            label='CLAS S-Wave (sideband)',
        )

        # D-wave
        wave_hist = fit_hists[Wave.encode(self.waves[1])]
        err = fit_error_bars[Wave.encode(self.waves[1])]
        centers = (wave_hist.bins[1:] + wave_hist.bins[:-1]) / 2
        ax[1].errorbar(
            centers,
            wave_hist.counts,
            yerr=0,
            fmt='.',
            markersize=3,
            color=self.waves[1].plot_color,
            label=self.waves[1].latex,
        )
        ax[1].errorbar(
            centers,
            err[1],
            yerr=(err[0], err[2]),
            fmt='none',
            color=self.waves[1].plot_color,
        )
        ax[1].errorbar(
            clas_centers,
            clas_d_wave_counts_sideband,
            yerr=clas_errors_sideband,
            fmt='^',
            markersize=1,
            color=PINK,
            label='CLAS D-Wave (sideband)',
        )
        ax[0].legend()
        ax[1].legend()
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)

        fig.supxlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int((MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / self.nbins * 1000)
        fig.supylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        fig.savefig(self.outputs[2])
        plt.close()
