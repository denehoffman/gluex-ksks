from pathlib import Path
from typing import Any, Literal, override
from matplotlib import patches
from modak import Task

import numpy as np
from gluex_ksks.constants import (
    BARYON_MASS_BINS,
    BARYON_MASS_RANGE,
    BLACK,
    BLUE,
    CHISQDOF_BINS,
    CHISQDOF_RANGE,
    COSTHETA_BINS,
    COSTHETA_RANGE,
    GREEN,
    LOG_PATH,
    ME_BINS,
    ME_RANGE,
    MESON_MASS_BINS,
    MESON_MASS_RANGE,
    MESON_PARTICLES,
    MM2_BINS,
    MM2_RANGE,
    ORANGE,
    PHI_BINS,
    PHI_RANGE,
    PLOTS_PATH,
    PROTONZ_BINS,
    PROTONZ_RANGE,
    PURPLE,
    RED,
    RF_BINS,
    RF_RANGE,
    RFL_BINS,
    RFL_RANGE,
    RUN_PERIODS,
    YOUDENJ_BINS,
    YOUDENJ_RANGE,
)
from gluex_ksks.tasks.cuts import FiducialCuts
import polars as pl
from gluex_ksks.tasks.splot import SPlotWeights
from gluex_ksks.utils import (
    add_hx_angles,
    add_ksb_costheta,
    add_m_baryon,
    add_m_meson,
    select_mesons_tag,
)
import matplotlib.pyplot as plt


def _get_inputs(
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> list[Task]:
    if method is None or nspec is None:
        return [
            FiducialCuts(
                data_type=data_type,
                run_period=run_period,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            )
            for run_period in RUN_PERIODS
        ]
    elif data_type == 'data':
        assert method is not None
        assert nspec is not None
        return [
            SPlotWeights(
                run_period=run_period,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            )
            for run_period in RUN_PERIODS
        ]
    else:
        raise ValueError('Unsupported data type + cut combination')


def _get_inputs_combined(
    *,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
) -> list[Task]:
    return [
        *[
            FiducialCuts(
                data_type='data',
                run_period=run_period,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            )
            for run_period in RUN_PERIODS
        ],
        *[
            FiducialCuts(
                data_type='sigmc',
                run_period=run_period,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            )
            for run_period in RUN_PERIODS
        ],
        *[
            FiducialCuts(
                data_type='bkgmc',
                run_period=run_period,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            )
            for run_period in RUN_PERIODS
        ],
    ]


def _get_outputs(
    names: str | list[str],
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> list[Path]:
    tag = select_mesons_tag(select_mesons)
    if isinstance(names, str):
        names = [names]
    return [
        PLOTS_PATH
        / f'{name}_{data_type}{"_pz" if protonz_cut else ""}{f"_chisqdof_{chisqdof:.1f}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}.svg'
        for name in names
    ]


def _get_unique_name(
    name: str,
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> str:
    tag = select_mesons_tag(select_mesons)
    return f'plot_{name}_{data_type}{"_pz" if protonz_cut else ""}{f"_chisqdof_{chisqdof:.1f}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}'


def _get_super(
    name: str,
    output_names: str | list[str],
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> dict[str, Any]:
    return {
        'name': _get_unique_name(
            name=name,
            data_type=data_type,
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        ),
        'inputs': _get_inputs(
            data_type=data_type,
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        ),
        'outputs': _get_outputs(
            names=output_names,
            data_type=data_type,
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        ),
        'log_directory': LOG_PATH,
    }


def _get_super_combined(
    name: str,
    output_names: str | list[str],
    *,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
) -> dict[str, Any]:
    return {
        'name': _get_unique_name(
            name=name,
            data_type='combined',
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=None,
            nspec=None,
        ),
        'inputs': _get_inputs_combined(
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        ),
        'outputs': _get_outputs(
            names=output_names,
            data_type='combined',
            protonz_cut=protonz_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=None,
            nspec=None,
        ),
        'log_directory': LOG_PATH,
    }


def _log_plot_start(
    logger,
    name: str,
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> None:
    tag = select_mesons_tag(select_mesons)
    logger.info(
        f'Plotting {name} for {data_type} (method={method}, nspec={nspec}) with pz={protonz_cut}, chisqdof={chisqdof}, select={tag}'
    )


def _log_plot_end(
    logger,
    name: str,
    *,
    data_type: str,
    protonz_cut: bool,
    chisqdof: float | None,
    select_mesons: bool | None,
    method: Literal['fixed', 'free'] | None,
    nspec: int | None,
) -> None:
    tag = select_mesons_tag(select_mesons)
    logger.info(
        f'Finished plotting {name} for {data_type} (method={method}, nspec={nspec}) with pz={protonz_cut}, chisqdof={chisqdof}, select={tag}'
    )


def _read_inputs(inputs: list[Task]) -> pl.DataFrame:
    return pl.concat(
        [pl.read_parquet(inp.outputs[0]) for inp in inputs[: len(RUN_PERIODS)]],
        how='diagonal',
        rechunk=True,
    )


def _read_inputs_combined(
    inputs: list[Task],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return (
        pl.concat(
            [pl.read_parquet(inp.outputs[0]) for inp in inputs[: len(RUN_PERIODS)]],
            how='diagonal',
            rechunk=True,
        ),
        pl.concat(
            [
                pl.read_parquet(inp.outputs[0])
                for inp in inputs[len(RUN_PERIODS) : 2 * len(RUN_PERIODS)]
            ],
            how='diagonal',
            rechunk=True,
        ),
        pl.concat(
            [
                pl.read_parquet(inp.outputs[0])
                for inp in inputs[2 * len(RUN_PERIODS) : 3 * len(RUN_PERIODS)]
            ],
            how='diagonal',
            rechunk=True,
        ),
    )


class PlotMesonMass(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'meson_mass',
                ['meson_mass', 'ksb_costheta_v_meson_mass', 'meson_pdg'],
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'meson mass', **self.__dict__)
        df_data = add_ksb_costheta(
            add_m_meson(
                _read_inputs(self.inputs),
            )
        )
        counts, edges = np.histogram(
            df_data['m_meson'],
            bins=MESON_MASS_BINS,
            range=MESON_MASS_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['m_meson'],
            bins=MESON_MASS_BINS,
            range=MESON_MASS_RANGE,
            weights=df_data['weight'].pow(2),
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        bin_width = int(
            (MESON_MASS_RANGE[1] - MESON_MASS_RANGE[0]) / MESON_MASS_BINS * 1000
        )
        ax.set_ylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.hist2d(
            df_data['m_meson'],
            df_data['ksb_costheta'],
            bins=(MESON_MASS_BINS, COSTHETA_BINS),
            range=(MESON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
        )
        ax.set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\cos\left(\theta\right)$ of $K_{S,B}^0$')
        plt.savefig(self.outputs[1])
        plt.close()

        _, (hist_ax, bar_ax) = plt.subplots(
            2,
            1,
            sharex=True,
            gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.0},
        )
        hist_ax.hist(
            df_data['m_meson'],
            weights=df_data['weight'],
            bins=MESON_MASS_BINS,
            range=MESON_MASS_RANGE,
            color=BLUE,
        )

        bar_height: float = 0.8
        bar_spacing: float = 0.3
        bar_ax.axhline(
            3 * (bar_height + bar_spacing) + bar_spacing / 2,
            ls=':',
            color=BLACK,
        )
        xmin, xmax = 0.95, 2.05
        for particle in MESON_PARTICLES:
            center = particle.center
            width = particle.width
            if center - width < xmin:
                xmin = center - width - 0.05
            if center + width > xmax:
                xmax = center + width + 0.05
            color = particle.color
            row = particle.row

            rect_bottom = row * (bar_height + bar_spacing) + (
                2 * bar_spacing if row > 2 else 0
            )

            rect = patches.Rectangle(
                (center - width / 2, rect_bottom),
                width,
                bar_height,
                edgecolor=color,
                facecolor=color,
                fill=particle.established,
            )
            bar_ax.add_patch(rect)
        bar_ax.set_xlabel('Invariant Mass of $K_S^0K_S^0$ (GeV/$c^2$)')
        hist_ax.set_ylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        patch_colors = [RED, GREEN, ORANGE, PURPLE]
        patch_labels = [r'$f_0$', r'$a_0$', r'$f_2$', r'$a_2$']
        rect_patches = [
            patches.Patch(facecolor=color, edgecolor=color, label=label)
            for color, label in zip(patch_colors, patch_labels)
        ]
        hist_ax.legend(handles=rect_patches, labels=patch_labels)
        hist_ax.set_xlim(xmin, xmax)
        bar_ax.set_ylim(
            -bar_spacing * 5,
            max(p.row for p in MESON_PARTICLES) * (bar_height + bar_spacing)
            + bar_height
            + bar_spacing * 7,
        )
        bar_ax.set_yticks([])
        bar_ax.spines['top'].set_visible(False)
        bar_ax.spines['right'].set_visible(False)
        bar_ax.spines['left'].set_visible(False)
        plt.savefig(self.outputs[2])
        plt.close()

        _log_plot_end(self.logger, 'meson mass', **self.__dict__)


class PlotBaryonMass(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'baryon_mass',
                ['baryon_mass', 'ksb_costheta_v_baryon_mass'],
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'baryon mass', **self.__dict__)
        df_data = add_ksb_costheta(
            add_m_baryon(
                _read_inputs(self.inputs),
            )
        )
        counts, edges = np.histogram(
            df_data['m_baryon'],
            bins=BARYON_MASS_BINS,
            range=BARYON_MASS_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['m_baryon'],
            bins=BARYON_MASS_BINS,
            range=BARYON_MASS_RANGE,
            weights=df_data['weight'].pow(2),
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$ (GeV/$c^2$)')
        bin_width = int(
            (BARYON_MASS_RANGE[1] - BARYON_MASS_RANGE[0]) / BARYON_MASS_BINS * 1000
        )
        ax.set_ylabel(f'Counts / {bin_width} (MeV/$c^2$)')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.hist2d(
            df_data['m_baryon'],
            df_data['ksb_costheta'],
            bins=(BARYON_MASS_BINS, COSTHETA_BINS),
            range=(BARYON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
        )
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\cos\left(\theta\right)$ of $K_{S,B}^0$')
        plt.savefig(self.outputs[1])
        plt.close()
        _log_plot_end(self.logger, 'baryon mass', **self.__dict__)


class PlotAngles(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'angle_plots',
                ['costheta_hx_v_meson_mass', 'phi_hx_v_meson_mass'],
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'angle plots', **self.__dict__)
        df_data = add_hx_angles(
            add_m_meson(
                _read_inputs(self.inputs),
            )
        )

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist2d(
            df_data['m_meson'],
            df_data['hx_costheta'],
            bins=(MESON_MASS_BINS, COSTHETA_BINS),
            range=(MESON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
        )
        ax.set_xlabel('Invariant Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\cos\left(\theta_{\text{HX}}\right)$')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.hist2d(
            df_data['m_meson'],
            df_data['hx_phi'],
            bins=(MESON_MASS_BINS, PHI_BINS),
            range=(MESON_MASS_RANGE, PHI_RANGE),
            weights=df_data['weight'],
        )
        ax.set_xlabel('Invariant Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\phi_{\text{HX}}$ (rad)')
        plt.savefig(self.outputs[1])
        plt.close()
        _log_plot_end(self.logger, 'angle plots', **self.__dict__)


class PlotRF(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'rf',
                ['rf_weighted', 'rf_unweighted'],
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'rf', **self.__dict__)
        df_data = _read_inputs(self.inputs)
        counts, edges = np.histogram(
            df_data['RF'],
            bins=RF_BINS,
            range=RF_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['RF'],
            bins=RF_BINS,
            range=RF_RANGE,
            weights=df_data['weight'].pow(2),
        )
        counts_unweighted, _ = np.histogram(
            df_data['RF'],
            bins=RF_BINS,
            range=RF_RANGE,
            weights=None,
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel(r'$\Delta t_{\text{RF}}$ (ns)')
        bin_width = int((RF_RANGE[1] - RF_RANGE[0]) / RF_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width} (ps)')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(counts_unweighted, edges, color=BLUE)
        ax.errorbar(
            bin_centers, counts, yerr=np.sqrt(counts_unweighted), fmt='none', color=BLUE
        )
        ax.set_xlabel(r'$\Delta t_{\text{RF}}$ (ns)')
        bin_width = int((RF_RANGE[1] - RF_RANGE[0]) / RF_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width} (ps)')
        plt.savefig(self.outputs[1])
        plt.close()
        _log_plot_end(self.logger, 'rf', **self.__dict__)


class PlotChiSqDOF(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'chisqdof',
                'chisqdof',
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'chisqdof', **self.__dict__)
        df_data = _read_inputs(self.inputs)
        counts, edges = np.histogram(
            df_data['ChiSqDOF'],
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['ChiSqDOF'],
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            weights=df_data['weight'].pow(2),
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        bin_width = round((CHISQDOF_RANGE[1] - CHISQDOF_RANGE[0]) / CHISQDOF_BINS, 1)
        ax.set_ylabel(f'Counts / {bin_width}')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'chisqdof', **self.__dict__)


class PlotChiSqDOFCombined(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'chisqdof',
                'chisqdof',
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'chisqdof', **self.__dict__)
        df_data, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist(
            df_data['ChiSqDOF'],
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            weights=df_data['weight'],
            density=True,
            histtype='step',
            color=BLUE,
            label='Data',
        )
        ax.hist(
            df_sigmc['ChiSqDOF'],
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            weights=df_sigmc['weight'],
            density=True,
            histtype='step',
            color=GREEN,
            label=r'$K_S^0 K_S^0$ MC',
        )
        ax.hist(
            df_bkgmc['ChiSqDOF'],
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
            histtype='step',
            color=RED,
            label=r'$4\pi$ MC',
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        bin_width = round((CHISQDOF_RANGE[1] - CHISQDOF_RANGE[0]) / CHISQDOF_BINS, 1)
        ax.set_ylabel(f'Normalized Counts / {bin_width}')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'chisqdof', **self.__dict__)


class PlotYoudenJAndROC(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'chisqdof_youdenj_and_roc',
                ['chisqdof_youdenj', 'chisqdof_roc'],
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'chisqdof_youdenj_and_roc', **self.__dict__)
        _, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        h_sigmc, edges = np.histogram(
            df_sigmc['ChiSqDOF'], bins=YOUDENJ_BINS, range=YOUDENJ_RANGE, density=True
        )
        h_bkgmc, _ = np.histogram(
            df_bkgmc['ChiSqDOF'], bins=YOUDENJ_BINS, range=YOUDENJ_RANGE, density=True
        )

        sigmc_cumsum = np.cumsum(h_sigmc)
        bkgmc_cumsum = np.cumsum(h_bkgmc)
        sigmc_eff = sigmc_cumsum / sigmc_cumsum[-1]
        bkgmc_eff = bkgmc_cumsum / bkgmc_cumsum[-1]
        youden_j = sigmc_eff - bkgmc_eff
        max_j_index = np.argmax(youden_j)
        cut_values = edges[:-1]
        max_j = cut_values[max_j_index]

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.plot(
            cut_values,
            youden_j,
            color=BLUE,
        )
        ax.axvline(max_j, color=RED, ls=':', label='Cut Value')
        ax.text(
            max_j,
            0.3,
            f'{max_j:0.2f}',
            color=RED,
            ha='right',
            va='top',
            rotation=90,
            transform=ax.get_xaxis_transform(),
        )
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        ax.set_ylabel(r'Youden J-Statistic ($J = \epsilon_S - \epsilon_B$)')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.plot(
            bkgmc_eff,
            sigmc_eff,
            lw=1.5,
            color=PURPLE,
            label='ROC Curve',
        )
        ax.plot([0, 1], [0, 1], color=BLACK, lw=1.5, ls=':')
        ax.set_xlabel(r'False Positive Rate ($\epsilon_B$)')
        ax.set_ylabel(r'True Positive Rate ($\epsilon_S$)')
        ax.grid(True)
        ax.legend()
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('square')
        plt.savefig(self.outputs[1])
        plt.close()
        _log_plot_end(self.logger, 'chisqdof_youdenj_and_roc', **self.__dict__)


class PlotProtonZ(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super(
                'protonz',
                'protonz',
                data_type=data_type,
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
                method=method,
                nspec=nspec,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'protonz', **self.__dict__)
        df_data = _read_inputs(self.inputs)
        counts, edges = np.histogram(
            df_data['Proton_Z'],
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['Proton_Z'],
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            weights=df_data['weight'].pow(2),
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel(r'Proton $z$-vertex (cm)')
        bin_width = int((PROTONZ_RANGE[1] - PROTONZ_RANGE[0]) / PROTONZ_BINS)
        ax.set_ylabel(f'Counts / {bin_width} (cm)')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'protonz', **self.__dict__)


class PlotProtonZCombined(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'protonz',
                'protonz',
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'protonz', **self.__dict__)
        df_data, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist(
            df_data['Proton_Z'],
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            weights=df_data['weight'],
            density=True,
            histtype='step',
            color=BLUE,
            label='Data',
        )
        ax.hist(
            df_sigmc['Proton_Z'],
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            weights=df_sigmc['weight'],
            density=True,
            histtype='step',
            color=GREEN,
            label=r'$K_S^0 K_S^0$ MC',
        )
        ax.hist(
            df_bkgmc['Proton_Z'],
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
            histtype='step',
            color=RED,
            label=r'$4\pi$ MC',
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'Proton $z$-vertex (cm)')
        bin_width = int((PROTONZ_RANGE[1] - PROTONZ_RANGE[0]) / PROTONZ_BINS)
        ax.set_ylabel(f'Normalized Counts / {bin_width} (cm)')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'protonz', **self.__dict__)


class PlotRFLCombined(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'rfl',
                'rfl',
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'rfl', **self.__dict__)
        df_data, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist(
            df_data['RFL1'],
            bins=RFL_BINS,
            range=RFL_RANGE,
            weights=df_data['weight'],
            density=True,
            histtype='step',
            color=BLUE,
            label='Data',
        )
        ax.hist(
            df_sigmc['RFL1'],
            bins=RFL_BINS,
            range=RFL_RANGE,
            weights=df_sigmc['weight'],
            density=True,
            histtype='step',
            color=GREEN,
            label=r'$K_S^0 K_S^0$ MC',
        )
        ax.hist(
            df_bkgmc['RFL1'],
            bins=RFL_BINS,
            range=RFL_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
            histtype='step',
            color=RED,
            label=r'$4\pi$ MC',
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'$K_S^0$ Rest Frame Lifetime (ns)')
        bin_width = int((RFL_RANGE[1] - RFL_RANGE[0]) / RFL_BINS * 1000)
        ax.set_ylabel(f'Normalized Counts / {bin_width} (ps)')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'rfl', **self.__dict__)


class PlotMM2Combined(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'mm2',
                'mm2',
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'mm2', **self.__dict__)
        df_data, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist(
            df_data['MM2'],
            bins=MM2_BINS,
            range=MM2_RANGE,
            weights=df_data['weight'],
            density=True,
            histtype='step',
            color=BLUE,
            label='Data',
        )
        ax.hist(
            df_sigmc['MM2'],
            bins=MM2_BINS,
            range=MM2_RANGE,
            weights=df_sigmc['weight'],
            density=True,
            histtype='step',
            color=GREEN,
            label=r'$K_S^0 K_S^0$ MC',
        )
        ax.hist(
            df_bkgmc['MM2'],
            bins=MM2_BINS,
            range=MM2_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
            histtype='step',
            color=RED,
            label=r'$4\pi$ MC',
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'Missing Mass Squared ($\text{{GeV}}^2/c^4$)')
        bin_width = round((MM2_RANGE[1] - MM2_RANGE[0]) / MM2_BINS, 3)
        ax.set_ylabel(rf'Normalized Counts / {bin_width} ($\text{{GeV}}^2/c^4$)')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'mm2', **self.__dict__)


class PlotMECombined(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(select_mesons)
        super().__init__(
            **_get_super_combined(
                'me',
                'me',
                protonz_cut=protonz_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            ),
        )

    @override
    def run(self) -> None:
        _log_plot_start(self.logger, 'me', **self.__dict__)
        df_data, df_sigmc, df_bkgmc = _read_inputs_combined(self.inputs)

        plt.style.use('thesis')
        _, ax = plt.subplots()
        ax.hist(
            df_data['ME'],
            bins=ME_BINS,
            range=ME_RANGE,
            weights=df_data['weight'],
            density=True,
            histtype='step',
            color=BLUE,
            label='Data',
        )
        ax.hist(
            df_sigmc['ME'],
            bins=ME_BINS,
            range=ME_RANGE,
            weights=df_sigmc['weight'],
            density=True,
            histtype='step',
            color=GREEN,
            label=r'$K_S^0 K_S^0$ MC',
        )
        ax.hist(
            df_bkgmc['ME'],
            bins=ME_BINS,
            range=ME_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
            histtype='step',
            color=RED,
            label=r'$4\pi$ MC',
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'Missing Energy ($\text{{GeV}}^2$)')
        bin_width = round((ME_RANGE[1] - ME_RANGE[0]) / ME_BINS, 3)
        ax.set_ylabel(rf'Normalized Counts / {bin_width} ($\text{{GeV}}^2$)')
        plt.savefig(self.outputs[0])
        plt.close()
        _log_plot_end(self.logger, 'me', **self.__dict__)


class PlotAll(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.nspec = nspec
        super().__init__(
            name=_get_unique_name('all', **self.__dict__),
            inputs=[
                PlotMesonMass(**self.__dict__),
                PlotBaryonMass(**self.__dict__),
                PlotAngles(**self.__dict__),
                PlotChiSqDOF(**self.__dict__),
                PlotChiSqDOFCombined(**self.__dict__),
                PlotYoudenJAndROC(**self.__dict__),
                PlotProtonZ(**self.__dict__),
                PlotProtonZCombined(**self.__dict__),
                PlotRFLCombined(**self.__dict__),
                PlotMM2Combined(**self.__dict__),
                PlotMECombined(**self.__dict__),
                PlotRF(**self.__dict__),
            ],
            outputs=[],
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        pass
