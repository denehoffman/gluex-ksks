from modak import TaskQueue

from gluex_ksks.constants import (
    mkdirs,
)
import laddu as ld
from itertools import product

from gluex_ksks.tasks.factorization import FactorizationReport
from gluex_ksks.tasks.plot import PlotAll

if __name__ == '__main__':
    mkdirs()
    tq = TaskQueue(workers=ld.available_parallelism())
    tq.run(
        [
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
            PlotAll(
                data_type='data',
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                method=None,
                nspec=None,
            ),
            FactorizationReport(
                protonz_cut=True,
                mass_cut=True,
                chisqdof=2.75,
                select_mesons=True,
                max_nspec=4,
            ),
        ]
    )
