from modak import TaskQueue

from gluex_ksks.constants import mkdirs, LOG_PATH
import laddu as ld
from itertools import product

from gluex_ksks.tasks.aux_plots import MakeAuxiliaryPlots
from gluex_ksks.tasks.bggen import PlotBGGEN
from gluex_ksks.tasks.factorization import FactorizationReport
from gluex_ksks.tasks.fits import (
    ProcessBinned,
    ProcessUnbinned,
)
from gluex_ksks.tasks.plot import PlotAll, PlotDetectors, PlotAltHypos
from gluex_ksks.tasks.splot import SPlotReport
from gluex_ksks.wave import Wave

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

if __name__ == '__main__':
    mkdirs()
    tq = TaskQueue(workers=ld.available_parallelism(), log_path=LOG_PATH / 'all.log')
    tq.run(
        [
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
                for data_type in ['data', 'sigmc']
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
                    product([True, False], [True, False], [None], [True, False, None])
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
                    product([True, False], [True, False], [None], [True, False, None])
                )
            ],
            PlotAll(
                data_type='data',
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                method=None,
                nspec=None,
            ),
            *[
                PlotDetectors(
                    data_type=data_type,
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=2.75,
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
                    chisqdof=2.75,
                    select_mesons=True,
                    method=None,
                    nspec=None,
                )
                for data_type in ['data', 'sigmc']
            ],
            FactorizationReport(
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                max_nspec=10,
            ),
            SPlotReport(
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                max_nspec=10,
            ),
            PlotAll(
                data_type='data',
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                method='free',
                nspec=2,
            ),
            PlotDetectors(
                data_type='data',
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                method='free',
                nspec=2,
            ),
            PlotAltHypos(
                data_type='data',
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                method='free',
                nspec=2,
            ),
            *[
                ProcessBinned(
                    protonz_cut=True,
                    mass_cut=True,
                    chisqdof=2.75,
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
                    chisqdof=2.75,
                    select_mesons=True,
                    method='free',
                    nspec=2,
                    waves=waves,
                )
                for waves in UNBINNED_WAVESETS
            ],
        ]
    )
