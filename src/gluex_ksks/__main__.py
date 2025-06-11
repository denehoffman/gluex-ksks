from modak import TaskQueue

from gluex_ksks.constants import (
    MAX_FITS,
    N_WORKERS,
    NBOOT,
    STATE_PATH,
    mkdirs,
    LOG_PATH,
)
import laddu as ld
from itertools import product

from gluex_ksks.tasks.aux_plots import MakeAuxiliaryPlots
from gluex_ksks.tasks.bggen import PlotBGGEN
from gluex_ksks.tasks.clas import PlotCLASComparison
from gluex_ksks.tasks.factorization import FactorizationReport
from gluex_ksks.tasks.fits import (
    ProcessBinned,
    ProcessUnbinned,
)
from gluex_ksks.tasks.flux import PlotFlux
from gluex_ksks.tasks.plot import PlotAll, PlotDetectors, PlotAltHypos
from gluex_ksks.tasks.splot import SPlotReport
from gluex_ksks.wave import Wave

import click

BINNED_WAVESETS = [
    [Wave(0, 0, '+'), Wave(2, 2, '+')],
    [Wave(0, 0, '+'), Wave(2, 1, '+')],
    [Wave(0, 0, '+'), Wave(2, 0, '+')],
    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')],
    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+'), Wave(2, 2, '-')],
]
UNBINNED_WAVESETS = [
    [Wave(0, 0, '+'), Wave(2, 2, '+')],
    [Wave(0, 0, '+'), Wave(0, 0, '-'), Wave(2, 2, '+')],
]

DEFAULT_CHISQDOF = 3.0


@click.command()
@click.option('--chisqdof', default=DEFAULT_CHISQDOF, type=float)
def main(chisqdof: float):
    mkdirs()
    state_file_path = STATE_PATH
    if chisqdof != DEFAULT_CHISQDOF:
        state_file_path = state_file_path.parent / (
            state_file_path.name + f'-{chisqdof:.2f}'
        )
    tq = TaskQueue(
        workers=N_WORKERS,
        resources={'fit': MAX_FITS, 'fitplot': 1},
        state_file_path=state_file_path,
        log_path=LOG_PATH / 'all.log',
    )
    tasks = [
        PlotFlux(),
        MakeAuxiliaryPlots(),
        *[
            PlotDetectors(
                data_type=data_type,
                protonz_cut=False,
                mass_cut=False,
                chisqdof=None,
                select_mesons=None,
                method=None,
                nspec=None,
            )
            for data_type in ['data', 'sigmc']
        ],
        *[
            PlotAltHypos(
                data_type=data_type,
                protonz_cut=False,
                mass_cut=False,
                chisqdof=None,
                select_mesons=None,
                method=None,
                nspec=None,
            )
            for data_type in ['data', 'sigmc', 'bkgmc']
        ],
        *[
            PlotAll(
                data_type='data',
                protonz_cut=cuts[0],
                mass_cut=cuts[1],
                chisqdof=cuts[2],
                select_mesons=cuts[3],
                method=None,
                nspec=None,
            )
            for cuts in list(
                product(
                    [True, False], [True, False], [None, chisqdof], [True, False, None]
                )
            )
        ],
        *[
            PlotBGGEN(
                protonz_cut=cuts[0],
                mass_cut=cuts[1],
                chisqdof=cuts[2],
                select_mesons=cuts[3],
            )
            for cuts in list(
                product(
                    [True, False], [True, False], [None, chisqdof], [True, False, None]
                )
            )
        ],
        FactorizationReport(
            protonz_cut=True,
            mass_cut=True,
            chisqdof=chisqdof,
            select_mesons=True,
            max_nspec=10,
        ),
        SPlotReport(
            protonz_cut=True,
            mass_cut=True,
            chisqdof=chisqdof,
            select_mesons=True,
            max_nspec=10,
        ),
        *[
            ProcessBinned(
                protonz_cut=True,
                mass_cut=True,
                chisqdof=chisqdof,
                select_mesons=True,
                method='free',
                nspec=2,
                waves=waves,
            )
            for waves in BINNED_WAVESETS
        ],
        *[
            ProcessUnbinned(
                protonz_cut=True,
                mass_cut=True,
                chisqdof=chisqdof,
                select_mesons=True,
                method='free',
                nspec=2,
                waves=waves,
            )
            for waves in UNBINNED_WAVESETS
        ],
    ]

    if chisqdof == DEFAULT_CHISQDOF:
        tasks.extend(
            [
                PlotCLASComparison(
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=chisqdof,
                    select_mesons=True,
                    method='free',
                    nspec=2,
                    nboot=NBOOT,
                    bootstrap_mode='CI-BC',
                ),
                PlotAll(
                    data_type='data',
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=chisqdof,
                    select_mesons=True,
                    method=None,
                    nspec=None,
                ),
                *[
                    PlotDetectors(
                        data_type=data_type,
                        protonz_cut=True,
                        mass_cut=True,
                        chisqdof=chisqdof,
                        select_mesons=True,
                        method=None,
                        nspec=None,
                    )
                    for data_type in ['data', 'sigmc']
                ],
                *[
                    PlotAltHypos(
                        data_type=data_type,
                        protonz_cut=True,
                        mass_cut=True,
                        chisqdof=chisqdof,
                        select_mesons=True,
                        method=None,
                        nspec=None,
                    )
                    for data_type in ['data', 'sigmc', 'bkgmc']
                ],
                PlotAll(
                    data_type='data',
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=chisqdof,
                    select_mesons=True,
                    method='free',
                    nspec=2,
                ),
                PlotDetectors(
                    data_type='data',
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=chisqdof,
                    select_mesons=True,
                    method='free',
                    nspec=2,
                ),
                PlotAltHypos(
                    data_type='data',
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=chisqdof,
                    select_mesons=True,
                    method='free',
                    nspec=2,
                ),
                *[
                    ProcessBinned(
                        protonz_cut=True,
                        mass_cut=True,
                        chisqdof=chisqdof,
                        select_mesons=None,
                        method='free',
                        nspec=2,
                        waves=waves,
                    )
                    for waves in BINNED_WAVESETS
                ],
                *[
                    ProcessUnbinned(
                        protonz_cut=True,
                        mass_cut=True,
                        chisqdof=chisqdof,
                        select_mesons=None,
                        method='free',
                        nspec=2,
                        waves=waves,
                    )
                    for waves in UNBINNED_WAVESETS
                ],
            ]
        )
    tq.run(tasks)


if __name__ == '__main__':
    main()
