from typing import Literal, override
from matplotlib import patches
from modak import Task

import numpy as np
from gluex_ksks.constants import (
    BARYON_MASS_2D_BINS,
    BARYON_MASS_BINS,
    BARYON_MASS_RANGE,
    BEAM_ENERGY_BINS,
    BEAM_ENERGY_RANGE,
    BETA_BINS,
    BETA_RANGE,
    BLACK,
    BLUE,
    CHISQDOF_BINS,
    CHISQDOF_RANGE,
    COSTHETA_BINS,
    COSTHETA_RANGE,
    DEDX_BINS,
    DEDX_RANGE,
    DELTA_BETA_BINS,
    DELTA_BETA_RANGE,
    DELTA_T_BINS,
    DELTA_T_RANGE,
    DETECTOR_THETA_DEG_BINS,
    DETECTOR_THETA_DEG_RANGE,
    E_OVER_P_BINS,
    E_OVER_P_RANGE,
    GREEN,
    LOG_PATH,
    ME_BINS,
    ME_RANGE,
    MESON_MASS_2D_BINS,
    MESON_MASS_BINS,
    MESON_MASS_RANGE,
    MESON_PARTICLES,
    MM2_BINS,
    MM2_RANGE,
    ORANGE,
    P_BINS,
    P_RANGE,
    PARTICLE_TO_LATEX,
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
    add_alt_hypos,
    add_hx_angles,
    add_ksb_costheta,
    add_m_baryon,
    add_m_meson,
    select_mesons_tag,
    custom_colormap,
)
import matplotlib.pyplot as plt

CMAP, NORM = custom_colormap()


class PlotTask(Task):
    def __init__(
        self,
        name: str,
        output_names: str | list[str],
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        **kwargs,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.nspec = nspec
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
                for run_period in RUN_PERIODS
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
                for run_period in RUN_PERIODS
            ]
        else:
            raise ValueError('Unsupported data type + cut combination')
        self.tag = select_mesons_tag(select_mesons)
        if isinstance(output_names, str):
            output_names = [output_names]
        outputs = [
            PLOTS_PATH
            / f'{name}_{data_type}{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{method}_{nspec}.png'
            for name in output_names
        ]
        task_name = f'plot_{name}_{data_type}{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{method}_{nspec}'
        self.short_name = name
        super().__init__(
            name=task_name,
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
            **kwargs,
        )

    def read_inputs(self) -> pl.LazyFrame:
        return pl.concat(
            [
                pl.scan_parquet(inp.outputs[0], low_memory=True)
                for inp in self.inputs[: len(RUN_PERIODS)]
            ],
            how='diagonal',
            rechunk=True,
        )

    def log_plot_start(
        self,
    ) -> None:
        self.logger.info(
            f'Plotting {self.short_name} for {self.data_type} (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )

    def log_plot_end(self) -> None:
        self.logger.info(
            f'Finished plotting {self.short_name} for {self.data_type} (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )


class DetectorPlotTask(Task):
    def __init__(
        self,
        name: str,
        output_names: str | list[str],
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        **kwargs,
    ):
        self.data_type = data_type
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.nspec = nspec
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
                for run_period in RUN_PERIODS
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
                for run_period in RUN_PERIODS
            ]
        else:
            raise ValueError('Unsupported data type + cut combination')
        self.tag = select_mesons_tag(select_mesons)
        if isinstance(output_names, str):
            output_names = [output_names]
        outputs = [
            PLOTS_PATH
            / f'{name}_{data_type}{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{method}_{nspec}.png'
            for name in output_names
        ]
        task_name = f'plot_{name}_{data_type}{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{method}_{nspec}'
        self.short_name = name
        super().__init__(
            name=task_name,
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
            **kwargs,
        )

    def read_inputs(self) -> pl.LazyFrame:
        return pl.concat(
            [
                pl.scan_parquet(inp.outputs[0], low_memory=True)
                for inp in self.inputs[: len(RUN_PERIODS)]
            ],
            how='diagonal',
            rechunk=True,
        )

    def log_plot_start(
        self,
    ) -> None:
        self.logger.info(
            f'Plotting {self.short_name} for {self.data_type} (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )

    def log_plot_end(self) -> None:
        self.logger.info(
            f'Finished plotting {self.short_name} for {self.data_type} (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )


class CombinedPlotTask(Task):
    def __init__(
        self,
        name: str,
        output_names: str | list[str],
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        **kwargs,
    ):
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        inputs: list[Task] = [
            *[
                FiducialCuts(
                    data_type='data',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
                    mass_cut=mass_cut,
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
                    mass_cut=mass_cut,
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
                    mass_cut=mass_cut,
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                )
                for run_period in RUN_PERIODS
            ],
        ]
        self.tag = select_mesons_tag(select_mesons)
        if isinstance(output_names, str):
            output_names = [output_names]
        outputs = [
            PLOTS_PATH
            / f'{name}_combined{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{None}_{None}.png'
            for name in output_names
        ]
        task_name = f'plot_{name}_combined{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{self.tag}_{None}_{None}'
        self.short_name = name
        super().__init__(
            name=task_name,
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
            **kwargs,
        )

    def read_inputs(self) -> tuple[pl.LazyFrame, pl.LazyFrame, pl.LazyFrame]:
        return (
            pl.concat(
                [
                    pl.scan_parquet(inp.outputs[0], low_memory=True)
                    for inp in self.inputs[: len(RUN_PERIODS)]
                ],
                how='diagonal',
                rechunk=True,
            ),
            pl.concat(
                [
                    pl.scan_parquet(inp.outputs[0], low_memory=True)
                    for inp in self.inputs[len(RUN_PERIODS) : 2 * len(RUN_PERIODS)]
                ],
                how='diagonal',
                rechunk=True,
            ),
            pl.concat(
                [
                    pl.scan_parquet(inp.outputs[0], low_memory=True)
                    for inp in self.inputs[2 * len(RUN_PERIODS) : 3 * len(RUN_PERIODS)]
                ],
                how='diagonal',
                rechunk=True,
            ),
        )

    def log_plot_start(
        self,
    ) -> None:
        self.logger.info(
            f'Plotting {self.short_name} for combined datasets with pz={self.protonz_cut}, mass_cut={self.mass_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )

    def log_plot_end(self) -> None:
        self.logger.info(
            f'Finished plotting {self.short_name} for combined datasets with pz={self.protonz_cut}, mass_cut={self.mass_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )


class PlotMesonMass(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'meson_mass',
            ['meson_mass', 'ksb_costheta_v_meson_mass', 'meson_pdg'],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = (
            add_ksb_costheta(
                add_m_meson(
                    self.read_inputs(),
                )
            )
            .select('m_meson', 'weight', 'ksb_costheta')
            .collect()
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

        plt.style.use('gluex_ksks.thesis')
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
        _, _, _, im = ax.hist2d(
            df_data['m_meson'],
            df_data['ksb_costheta'],
            bins=(MESON_MASS_2D_BINS, COSTHETA_BINS),
            range=(MESON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
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
        hist_ax.stairs(counts, edges, color=BLUE)
        hist_ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)

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

        self.log_plot_end()


class PlotBaryonMass(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'baryon_mass',
            ['baryon_mass', 'ksb_costheta_v_baryon_mass'],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = (
            add_ksb_costheta(
                add_m_baryon(
                    self.read_inputs(),
                )
            )
            .select('m_baryon', 'weight', 'ksb_costheta')
            .collect()
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

        plt.style.use('gluex_ksks.thesis')
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
        _, _, _, im = ax.hist2d(
            df_data['m_baryon'],
            df_data['ksb_costheta'],
            bins=(BARYON_MASS_2D_BINS, COSTHETA_BINS),
            range=(BARYON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel('Invariant Mass of $K_{S,B}^0 p$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\cos\left(\theta\right)$ of $K_{S,B}^0$')
        plt.savefig(self.outputs[1])
        plt.close()
        self.log_plot_end()


class PlotAngles(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'angle_plots',
            ['costheta_hx_v_meson_mass', 'phi_hx_v_meson_mass'],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = (
            add_hx_angles(
                add_m_meson(
                    self.read_inputs(),
                )
            )
            .select('m_meson', 'weight', 'hx_costheta', 'hx_phi')
            .collect()
        )

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data['m_meson'],
            df_data['hx_costheta'],
            bins=(MESON_MASS_2D_BINS, COSTHETA_BINS),
            range=(MESON_MASS_RANGE, COSTHETA_RANGE),
            weights=df_data['weight'],
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel('Invariant Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\cos\left(\theta_{\text{HX}}\right)$')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data['m_meson'],
            df_data['hx_phi'],
            bins=(MESON_MASS_2D_BINS, PHI_BINS),
            range=(MESON_MASS_RANGE, PHI_RANGE),
            weights=df_data['weight'],
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel('Invariant Mass of $K_S^0 K_S^0$ (GeV/$c^2$)')
        ax.set_ylabel(r'$\phi_{\text{HX}}$ (rad)')
        plt.savefig(self.outputs[1])
        plt.close()
        self.log_plot_end()


class PlotRF(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'rf',
            ['rf_weighted', 'rf_unweighted'],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = self.read_inputs().select('RF', 'weight').collect()
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

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_xlabel(r'$\Delta t_{\text{RF}}$ (ns)')
        bin_width = int((RF_RANGE[1] - RF_RANGE[0]) / RF_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width} (ps)')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(counts_unweighted, edges, color=BLUE)
        ax.errorbar(
            bin_centers,
            counts_unweighted,
            yerr=np.sqrt(counts_unweighted),
            fmt='none',
            color=BLUE,
        )
        ax.set_xlabel(r'$\Delta t_{\text{RF}}$ (ns)')
        bin_width = int((RF_RANGE[1] - RF_RANGE[0]) / RF_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width} (ps)')
        ax.set_ylim(0)
        plt.savefig(self.outputs[1])
        plt.close()
        self.log_plot_end()


class PlotChiSqDOF(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'chisqdof',
            'chisqdof',
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = self.read_inputs().select('ChiSqDOF', 'weight').collect()
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

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        bin_width = round((CHISQDOF_RANGE[1] - CHISQDOF_RANGE[0]) / CHISQDOF_BINS, 1)
        ax.set_ylabel(f'Counts / {bin_width}')
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotChiSqDOFCombined(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'chisqdof',
            'chisqdof',
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data, df_sigmc, df_bkgmc = self.read_inputs()
        df_data = df_data.select('ChiSqDOF', 'weight').collect()
        df_sigmc = df_sigmc.select('ChiSqDOF', 'weight').collect()
        df_bkgmc = df_bkgmc.select('ChiSqDOF', 'weight').collect()

        plt.style.use('gluex_ksks.thesis')
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
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotYoudenJAndROC(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'chisqdof_youdenj_and_roc',
            ['chisqdof_youdenj', 'chisqdof_roc'],
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        _, df_sigmc, df_bkgmc = self.read_inputs()
        df_sigmc = df_sigmc.select('ChiSqDOF', 'weight').collect()
        df_bkgmc = df_bkgmc.select('ChiSqDOF', 'weight').collect()

        h_sigmc, edges = np.histogram(
            df_sigmc['ChiSqDOF'],
            bins=YOUDENJ_BINS,
            range=YOUDENJ_RANGE,
            weights=df_sigmc['weight'],
            density=True,
        )
        h_bkgmc, _ = np.histogram(
            df_bkgmc['ChiSqDOF'],
            bins=YOUDENJ_BINS,
            range=YOUDENJ_RANGE,
            weights=df_bkgmc['weight'],
            density=True,
        )

        sigmc_cumsum = np.cumsum(h_sigmc)
        bkgmc_cumsum = np.cumsum(h_bkgmc)
        sigmc_eff = sigmc_cumsum / sigmc_cumsum[-1]
        bkgmc_eff = bkgmc_cumsum / bkgmc_cumsum[-1]
        youden_j = sigmc_eff - bkgmc_eff
        max_j_index = np.argmax(youden_j)
        cut_values = edges[:-1]
        max_j = cut_values[max_j_index]

        plt.style.use('gluex_ksks.thesis')
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
        self.log_plot_end()


class PlotProtonZ(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'protonz',
            'protonz',
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = self.read_inputs().select('Proton_Z', 'weight').collect()
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

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel(r'Proton $z$-vertex (cm)')
        bin_width = int((PROTONZ_RANGE[1] - PROTONZ_RANGE[0]) / PROTONZ_BINS)
        ax.set_ylabel(f'Counts / {bin_width} (cm)')
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotProtonZCombined(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'protonz',
            'protonz',
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data, df_sigmc, df_bkgmc = self.read_inputs()
        df_data = df_data.select('Proton_Z', 'weight').collect()
        df_sigmc = df_sigmc.select('Proton_Z', 'weight').collect()
        df_bkgmc = df_bkgmc.select('Proton_Z', 'weight').collect()

        plt.style.use('gluex_ksks.thesis')
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
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotRFLCombined(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'rfl',
            'rfl',
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data, df_sigmc, df_bkgmc = self.read_inputs()
        df_data = df_data.select('RFL1', 'weight').collect()
        df_sigmc = df_sigmc.select('RFL1', 'weight').collect()
        df_bkgmc = df_bkgmc.select('RFL1', 'weight').collect()

        plt.style.use('gluex_ksks.thesis')
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
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotMM2Combined(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'mm2',
            'mm2',
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data, df_sigmc, df_bkgmc = self.read_inputs()
        df_data = df_data.select('MM2', 'weight').collect()
        df_sigmc = df_sigmc.select('MM2', 'weight').collect()
        df_bkgmc = df_bkgmc.select('MM2', 'weight').collect()

        plt.style.use('gluex_ksks.thesis')
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
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotMECombined(CombinedPlotTask):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        super().__init__(
            'me',
            'me',
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data, df_sigmc, df_bkgmc = self.read_inputs()
        df_data = df_data.select('ME', 'weight').collect()
        df_sigmc = df_sigmc.select('ME', 'weight').collect()
        df_bkgmc = df_bkgmc.select('ME', 'weight').collect()

        plt.style.use('gluex_ksks.thesis')
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
        ax.set_xlabel(r'Missing Energy ($\text{{GeV}}$)')
        bin_width = round((ME_RANGE[1] - ME_RANGE[0]) / ME_BINS, 3)
        ax.set_ylabel(rf'Normalized Counts / {bin_width} ($\text{{GeV}}$)')
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotBeamEnergy(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'beam_energy',
            'beam_energy',
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = self.read_inputs().select('p4_0_E', 'weight').collect()
        counts, edges = np.histogram(
            df_data['p4_0_E'],
            bins=BEAM_ENERGY_BINS,
            range=BEAM_ENERGY_RANGE,
            weights=df_data['weight'],
        )
        weights_squared, _ = np.histogram(
            df_data['p4_0_E'],
            bins=BEAM_ENERGY_BINS,
            range=BEAM_ENERGY_RANGE,
            weights=df_data['weight'].pow(2),
        )
        bin_centers = (edges[:-1] + edges[1:]) / 2
        errors = np.sqrt(weights_squared)

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.stairs(counts, edges, color=BLUE)
        ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
        ax.set_ylim(0)
        ax.set_xlabel('Beam Energy (GeV)')
        bin_width = int(
            (BEAM_ENERGY_RANGE[1] - BEAM_ENERGY_RANGE[0]) / BEAM_ENERGY_BINS * 1000
        )
        ax.set_ylabel(f'Counts / {bin_width} (MeV)')
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotAltHypos(PlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        super().__init__(
            'alt_hypos',
            [
                'hypo_ks12',
                'hypo_ks21',
                'hypo_deltaplusplus1',
                'hypo_deltaplusplus2',
                'hypo_lambda1',
                'hypo_lambda2',
                'hypo_deltaplus11',
                'hypo_deltaplus12',
                'hypo_deltaplus21',
                'hypo_deltaplus22',
                'hypo_deltaminus',
                'hypo_ks12_v_ks21',
            ],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        df_data = (
            add_alt_hypos(self.read_inputs())
            .select(
                'piplus1_piminus2_m',
                'piplus2_piminus1_m',
                'p_piplus1_m',
                'p_piplus2_m',
                'p_piminus1_m',
                'p_piminus2_m',
                'p_piplus1_piminus1_m',
                'p_piplus1_piminus2_m',
                'p_piplus2_piminus1_m',
                'p_piplus2_piminus2_m',
                'p_piminus1_piminus2_m',
                'weight',
            )
            .collect()
        )
        self.logger.info('Read data, plotting...')
        column_names = [
            'piplus1_piminus2_m',
            'piplus2_piminus1_m',
            'p_piplus1_m',
            'p_piplus2_m',
            'p_piminus1_m',
            'p_piminus2_m',
            'p_piplus1_piminus1_m',
            'p_piplus1_piminus2_m',
            'p_piplus2_piminus1_m',
            'p_piplus2_piminus2_m',
            'p_piminus1_piminus2_m',
        ]
        latex_particles_strings = [
            r'\pi^+_1\pi^-_2',
            r'\pi^+_2\pi^-_1',
            r'p \pi^+_1',
            r'p \pi^+_2',
            r'p \pi^-_1',
            r'p \pi^-_2',
            r'p \pi^+_1 \pi^-_1',
            r'p \pi^+_1 \pi^-_2',
            r'p \pi^+_2 \pi^-_1',
            r'p \pi^+_2 \pi^-_2',
            r'p \pi^-_1 \pi^-_2',
        ]
        ranges = [
            (0.25, 1.7),
            (0.25, 1.7),
            (1.0, 3.3),
            (1.0, 3.3),
            (1.0, 3.3),
            (1.0, 3.3),
            (1.3, 5.0),
            (1.3, 5.0),
            (1.3, 5.0),
            (1.3, 5.0),
            (1.3, 5.0),
        ]
        binnings = [200] * len(column_names)
        for i, (column_name, latex_particles_string, bins, range_) in enumerate(
            zip(column_names, latex_particles_strings, binnings, ranges)
        ):
            self.logger.info(f'Plotting {column_name}')
            counts, edges = np.histogram(
                df_data[column_name],
                bins=bins,
                range=range_,
                weights=df_data['weight'],
            )
            weights_squared, _ = np.histogram(
                df_data[column_name],
                bins=bins,
                range=range_,
                weights=df_data['weight'].pow(2),
            )
            bin_centers = (edges[:-1] + edges[1:]) / 2
            errors = np.sqrt(weights_squared)

            plt.style.use('gluex_ksks.thesis')
            _, ax = plt.subplots()
            ax.stairs(counts, edges, color=BLUE)
            ax.errorbar(bin_centers, counts, yerr=errors, fmt='none', color=BLUE)
            ax.set_ylim(0)
            ax.set_xlabel(f'Invariant Mass of ${latex_particles_string}$ (GeV/$c^2$)')
            bin_width = int((range_[1] - range_[0]) / bins * 1000)
            ax.set_ylabel(f'Counts / {bin_width} (MeV)')
            plt.savefig(self.outputs[i])
            plt.close()
        self.logger.info('Plotting 2d pi+pi- correlation')
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        range_ = (0.25, 1.7)
        bins = 200
        _, _, _, im = ax.hist2d(
            df_data['piplus1_piminus2_m'],
            df_data['piplus2_piminus1_m'],
            bins=(bins, bins),
            range=(range_, range_),
            weights=df_data['weight'],
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(r'Invariant Mass of $\pi^+_1\pi^-_2$ (GeV/$c^2$)')
        ax.set_ylabel(r'Invariant Mass of $\pi^+_2\pi^-_1$ (GeV/$c^2$)')
        plt.savefig(self.outputs[-1])
        plt.close()
        self.log_plot_end()


class PlotDeltaTVP(DetectorPlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        particle: Literal['Proton', 'PiPlus1', 'PiMinus1', 'PiPlus2', 'PiMinus2'] | str,
        detector: Literal['BCAL', 'TOF', 'FCAL'] | str,
    ):
        self.particle = particle
        self.detector = detector
        super().__init__(
            f'delta_t_v_p_{str(detector).lower()}_{str(particle).lower()}',
            f'delta_t_v_p_{str(detector).lower()}_{str(particle).lower()}',
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        delta_t_column = f'{self.particle}_DeltaT_{self.detector}'
        p_column = f'{self.particle}_P'
        df_data = (
            self.read_inputs()
            .select(delta_t_column, p_column, 'weight')
            .filter(pl.col(delta_t_column).ne(0.0))
            .collect()
        )
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data[p_column],
            df_data[delta_t_column],
            weights=df_data['weight'],
            bins=(P_BINS, DELTA_T_BINS),
            range=(P_RANGE, DELTA_T_RANGE),
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(f'${PARTICLE_TO_LATEX[self.particle]}$ Momentum (GeV/c)')
        ax.set_ylabel(rf'{self.detector} $\Delta t$ (ns)')
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotBetaVP(DetectorPlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        particle: Literal['Proton', 'PiPlus1', 'PiMinus1', 'PiPlus2', 'PiMinus2'] | str,
        detector: Literal['BCAL', 'TOF', 'FCAL'] | str,
    ):
        self.particle = particle
        self.detector = detector
        super().__init__(
            f'beta_v_p_{str(detector).lower()}_{str(particle).lower()}',
            [
                f'beta_v_p_{str(detector).lower()}_{str(particle).lower()}',
                f'delta_beta_v_p_{str(detector).lower()}_{str(particle).lower()}',
            ],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        beta_column = f'{self.particle}_Beta_{self.detector}'
        p_column = f'{self.particle}_P'
        e_column = f'{self.particle}_E_{self.detector}'
        if self.detector != 'TOF':
            df_data = (
                self.read_inputs()
                .select(beta_column, p_column, e_column, 'weight')
                .filter(pl.col(beta_column).ne(0.0))
                .filter(pl.col(e_column).ne(0.0))
                .collect()
            )
        else:
            df_data = (
                self.read_inputs()
                .select(beta_column, p_column, 'weight')
                .filter(pl.col(beta_column).ne(0.0))
                .collect()
            )
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data[p_column],
            df_data[beta_column],
            weights=df_data['weight'],
            bins=(P_BINS, BETA_BINS[self.particle]),
            range=(P_RANGE, BETA_RANGE[self.particle]),
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(f'${PARTICLE_TO_LATEX[self.particle]}$ Momentum (GeV/c)')
        ax.set_ylabel(rf'{self.detector} $\beta$')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        if self.detector != 'TOF':
            _, _, _, im = ax.hist2d(
                df_data[p_column],
                df_data[p_column] / df_data[e_column] - df_data[beta_column],
                weights=df_data['weight'],
                bins=(P_BINS, DELTA_BETA_BINS),
                range=(P_RANGE, DELTA_BETA_RANGE),
                cmap=CMAP,
                norm=NORM,
            )
            plt.colorbar(im)
        else:
            ax.text(
                0.5,
                0.5,
                r'No $\delta\beta$ Plot for TOF',
                ha='center',
                va='center',
                transform=ax.transAxes,
            )
        ax.set_xlabel(f'${PARTICLE_TO_LATEX[self.particle]}$ Momentum (GeV/c)')
        ax.set_ylabel(rf'{self.detector} $\Delta\beta$')
        plt.savefig(self.outputs[1])
        plt.close()
        self.log_plot_end()


class PlotDEDXVP(DetectorPlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        particle: Literal['Proton', 'PiPlus1', 'PiMinus1', 'PiPlus2', 'PiMinus2'] | str,
        detector: Literal['CDC', 'CDC_integral', 'FDC', 'ST', 'TOF'] | str,
    ):
        self.particle = particle
        self.detector = detector
        super().__init__(
            f'dedx_v_p_{str(detector).lower()}_{str(particle).lower()}',
            f'dedx_v_p_{str(detector).lower()}_{str(particle).lower()}',
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        dedx_column = f'{self.particle}_dEdx_{self.detector}'
        p_column = f'{self.particle}_P'
        df_data = (
            self.read_inputs()
            .select(dedx_column, p_column, 'weight')
            .filter(pl.col(dedx_column).ne(0.0))
            .collect()
        )
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data[p_column],
            df_data[dedx_column],
            weights=df_data['weight'],
            bins=(P_BINS, DEDX_BINS[self.particle]),
            range=(P_RANGE, DEDX_RANGE[self.particle]),
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(f'${PARTICLE_TO_LATEX[self.particle]}$ Momentum (GeV/c)')
        ax.set_ylabel(rf'{self.detector} $\mathrm{{d}}E/\mathrm{{d}}x$ (keV/cm)')
        plt.savefig(self.outputs[0])
        plt.close()
        self.log_plot_end()


class PlotEoverPVPandTheta(DetectorPlotTask):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
        particle: Literal['Proton', 'PiPlus1', 'PiMinus1', 'PiPlus2', 'PiMinus2'] | str,
        detector: Literal['BCAL', 'FCAL'] | str,
    ):
        self.particle = particle
        self.detector = detector
        super().__init__(
            f'e_over_p_v_p_and_theta_{str(detector).lower()}_{str(particle).lower()}',
            [
                f'e_over_p_v_p_{str(detector).lower()}_{str(particle).lower()}',
                f'e_over_p_v_theta_{str(detector).lower()}_{str(particle).lower()}',
            ],
            data_type=data_type,
            protonz_cut=protonz_cut,
            mass_cut=mass_cut,
            chisqdof=chisqdof,
            select_mesons=select_mesons,
            method=method,
            nspec=nspec,
        )

    @override
    def run(self) -> None:
        self.log_plot_start()
        e_column = f'{self.particle}_E_{self.detector}'
        p_column = f'{self.particle}_P'
        theta_column = f'{self.particle}_Theta'
        df_data = (
            self.read_inputs()
            .select(e_column, p_column, theta_column, 'weight')
            .filter(pl.col(e_column).ne(0.0))
            .collect()
        )
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data[p_column],
            df_data[e_column] / df_data[p_column],
            weights=df_data['weight'],
            bins=(P_BINS, E_OVER_P_BINS[self.particle]),
            range=(P_RANGE, E_OVER_P_RANGE[self.particle]),
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(f'${PARTICLE_TO_LATEX[self.particle]}$ Momentum (GeV/c)')
        ax.set_ylabel(rf'{self.detector} $E/p$')
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        _, _, _, im = ax.hist2d(
            df_data[theta_column] * 180 / np.pi,
            df_data[e_column] / df_data[p_column],
            weights=df_data['weight'],
            bins=(DETECTOR_THETA_DEG_BINS[self.detector], E_OVER_P_BINS[self.particle]),
            range=(
                DETECTOR_THETA_DEG_RANGE[self.detector],
                E_OVER_P_RANGE[self.particle],
            ),
            cmap=CMAP,
            norm=NORM,
        )
        plt.colorbar(im)
        ax.set_xlabel(rf'${PARTICLE_TO_LATEX[self.particle]}$ $\theta$ (deg)')
        ax.set_ylabel(rf'{self.detector} $E/p$')
        plt.savefig(self.outputs[1])
        plt.close()
        self.log_plot_end()


class PlotDetectors(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        single_plot_kwargs = {
            'data_type': data_type,
            'protonz_cut': protonz_cut,
            'mass_cut': mass_cut,
            'chisqdof': chisqdof,
            'select_mesons': select_mesons,
            'method': method,
            'nspec': nspec,
        }
        tag = select_mesons_tag(select_mesons)
        super().__init__(
            name=f'plot_detectors_{data_type}{"_pz" if protonz_cut else ""}{"masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}',
            inputs=[
                *[
                    PlotDeltaTVP(
                        **single_plot_kwargs, particle=particle, detector=detector
                    )
                    for particle in [
                        'Proton',
                        'PiPlus1',
                        'PiMinus1',
                        'PiPlus2',
                        'PiMinus2',
                    ]
                    for detector in ['BCAL', 'TOF', 'FCAL']
                ],
                *[
                    PlotBetaVP(
                        **single_plot_kwargs, particle=particle, detector=detector
                    )
                    for particle in [
                        'Proton',
                        'PiPlus1',
                        'PiMinus1',
                        'PiPlus2',
                        'PiMinus2',
                    ]
                    for detector in ['BCAL', 'TOF', 'FCAL']
                ],
                *[
                    PlotDEDXVP(
                        **single_plot_kwargs, particle=particle, detector=detector
                    )
                    for particle in [
                        'Proton',
                        'PiPlus1',
                        'PiMinus1',
                        'PiPlus2',
                        'PiMinus2',
                    ]
                    for detector in ['CDC', 'CDC_integral', 'FDC', 'ST', 'TOF']
                ],
                *[
                    PlotEoverPVPandTheta(
                        **single_plot_kwargs, particle=particle, detector=detector
                    )
                    for particle in [
                        'Proton',
                        'PiPlus1',
                        'PiMinus1',
                        'PiPlus2',
                        'PiMinus2',
                    ]
                    for detector in ['BCAL', 'FCAL']
                ],
            ],
            outputs=[],
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        pass


class PlotAll(Task):
    def __init__(
        self,
        *,
        data_type: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | None,
        nspec: int | None,
    ):
        single_plot_kwargs = {
            'data_type': data_type,
            'protonz_cut': protonz_cut,
            'mass_cut': mass_cut,
            'chisqdof': chisqdof,
            'select_mesons': select_mesons,
            'method': method,
            'nspec': nspec,
        }
        combined_plot_kwargs = {
            'protonz_cut': protonz_cut,
            'mass_cut': mass_cut,
            'chisqdof': chisqdof,
            'select_mesons': select_mesons,
        }
        tag = select_mesons_tag(select_mesons)
        super().__init__(
            name=f'plot_all_{data_type}{"_pz" if protonz_cut else ""}{"masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}_{method}_{nspec}',
            inputs=[
                PlotMesonMass(**single_plot_kwargs),
                PlotBaryonMass(**single_plot_kwargs),
                PlotAngles(**single_plot_kwargs),
                PlotChiSqDOF(**single_plot_kwargs),
                PlotChiSqDOFCombined(**combined_plot_kwargs),
                PlotYoudenJAndROC(**combined_plot_kwargs),
                PlotProtonZ(**single_plot_kwargs),
                PlotProtonZCombined(**combined_plot_kwargs),
                PlotRFLCombined(**combined_plot_kwargs),
                PlotMM2Combined(**combined_plot_kwargs),
                PlotMECombined(**combined_plot_kwargs),
                PlotRF(**single_plot_kwargs),
                PlotBeamEnergy(**single_plot_kwargs),
            ],
            outputs=[],
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        pass
