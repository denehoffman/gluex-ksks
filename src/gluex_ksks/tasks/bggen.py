from typing import override

import matplotlib.pyplot as plt
from modak import Task
import polars as pl

from gluex_ksks.constants import (
    BLUE,
    CHISQDOF_BINS,
    CHISQDOF_RANGE,
    GRAY,
    GREEN,
    LOG_PATH,
    ME_BINS,
    ME_RANGE,
    MM2_BINS,
    MM2_RANGE,
    ORANGE,
    PINK,
    PLOTS_PATH,
    PROTONZ_BINS,
    PROTONZ_RANGE,
    PURPLE,
    RED,
    RFL_BINS,
    RFL_RANGE,
    RUN_PERIODS_BGGEN,
)
from gluex_ksks.tasks.cuts import FiducialCuts
from gluex_ksks.utils import select_mesons_tag


def topo_to_latex(topo: str) -> str:
    topo = topo.replace('#plus', '+')
    topo = topo.replace('#minus', '-')
    topo = topo.replace('#', '\\')
    return f'${topo}$'


class PlotBGGEN(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        tag = select_mesons_tag(select_mesons)
        inputs: list[Task] = [
            FiducialCuts(
                data_type='bggen',
                run_period=run_period,
                protonz_cut=protonz_cut,
                mass_cut=mass_cut,
                chisqdof=chisqdof,
                select_mesons=select_mesons,
            )
            for run_period in RUN_PERIODS_BGGEN
        ]
        outputs = [
            PLOTS_PATH
            / f'{name}_bggen{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}.png'
            for name in ['chisqdof_large', 'chisqdof', 'protonz', 'rfl', 'mm2', 'me']
        ]
        task_name = f'bggen_plots{"_pz" if protonz_cut else ""}{"_masscut" if mass_cut else ""}{f"_chisqdof_{chisqdof}" if chisqdof is not None else ""}_{tag}'
        super().__init__(
            task_name, inputs=inputs, outputs=outputs, log_directory=LOG_PATH
        )

    @override
    def run(self):
        data_df = pl.concat(
            [pl.scan_parquet(inp.outputs[0]) for inp in self.inputs],
            how='diagonal',
            rechunk=True,
        ).select('ChiSqDOF', 'Proton_Z', 'RFL1', 'MM2', 'ME', 'Topology', 'weight')
        target_topo = '2#pi^{#plus}2#pi^{#minus}p[K^{0}_{S}]'

        top_five_topos = (
            data_df.filter(pl.col('Topology').ne(target_topo))
            .group_by('Topology')
            .agg(pl.count().alias('count'))
            .sort('count', descending=True)
            .limit(5)
            .select('Topology')
            .collect()
            .get_column('Topology')
            .to_list()
        )
        top_topos = [target_topo] + top_five_topos
        data_df = data_df.with_columns(
            [
                pl.when(pl.col('Topology').is_in(top_topos))
                .then(pl.col('Topology'))
                .otherwise(pl.lit('Other'))
                .alias('TopologyGroup')
            ]
        ).collect()

        data_chisqdof = []
        data_protonz = []
        data_rfl = []
        data_mm2 = []
        data_me = []
        weights = []

        for group in top_topos + ['Other']:
            group_df = data_df.filter(pl.col('TopologyGroup') == group)
            data_chisqdof.append(group_df['ChiSqDOF'].to_numpy())
            data_protonz.append(group_df['Proton_Z'].to_numpy())
            data_rfl.append(group_df['RFL1'].to_numpy())
            data_mm2.append(group_df['MM2'].to_numpy())
            data_me.append(group_df['ME'].to_numpy())
            weights.append(group_df['weight'].to_numpy())

        n_groups = len(top_topos) + 1

        color_sequence = [
            BLUE,
            RED,
            GREEN,
            PURPLE,
            ORANGE,
            PINK,
            GRAY,
        ][:n_groups]

        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.hist(
            data_chisqdof,
            weights=weights,
            bins=200,
            range=(0.0, 200.0),
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        bin_width = 1.0 / 200
        ax.set_ylabel(f'Counts / {bin_width:.2f}')
        ax.legend()
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.hist(
            data_chisqdof,
            weights=weights,
            bins=CHISQDOF_BINS,
            range=CHISQDOF_RANGE,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_ylim(0)
        ax.set_xlabel(r'$\chi^2_{\nu}$')
        bin_width = round((CHISQDOF_RANGE[1] - CHISQDOF_RANGE[0]) / CHISQDOF_BINS, 1)
        ax.set_ylabel(f'Counts / {bin_width}')
        ax.legend()
        plt.savefig(self.outputs[1])
        plt.close()

        _, ax = plt.subplots()
        ax.hist(
            data_protonz,
            weights=weights,
            bins=PROTONZ_BINS,
            range=PROTONZ_RANGE,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(r'Proton $z$-vertex (cm)')
        bin_width = int((PROTONZ_RANGE[1] - PROTONZ_RANGE[0]) / PROTONZ_BINS)
        ax.set_ylabel(f'Counts / {bin_width} (cm)')
        ax.set_ylim(1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[2])
        plt.close()

        _, ax = plt.subplots()
        ax.hist(
            data_rfl,
            weights=weights,
            bins=RFL_BINS,
            range=RFL_RANGE,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(r'$K_S^0$ Rest Frame Lifetime (ns)')
        bin_width = int((RFL_RANGE[1] - RFL_RANGE[0]) / RFL_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width} (ps)')
        ax.set_ylim(1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[3])
        plt.close()

        _, ax = plt.subplots()
        ax.hist(
            data_mm2,
            weights=weights,
            bins=MM2_BINS,
            range=MM2_RANGE,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(r'Missing Mass Squared ($\text{{GeV}}^2/c^4$)')
        bin_width = round((MM2_RANGE[1] - MM2_RANGE[0]) / MM2_BINS, 3)
        ax.set_ylabel(rf'Counts / {bin_width} ($\text{{GeV}}^2/c^4$)')
        ax.set_ylim(1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[4])
        plt.close()

        _, ax = plt.subplots()
        ax.hist(
            data_me,
            weights=weights,
            bins=ME_BINS,
            range=ME_RANGE,
            color=color_sequence,
            stacked=True,
            label=[
                'Signal',
                *[topo_to_latex(topo) for topo in top_five_topos],
                'Other',
            ],
        )
        ax.set_xlabel(r'Missing Energy ($\text{{GeV}}$)')
        bin_width = round((ME_RANGE[1] - ME_RANGE[0]) / ME_BINS, 3)
        ax.set_ylabel(rf'Counts / {bin_width} ($\text{{GeV}}$)')
        ax.set_ylim(1)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[5])
        plt.close()
