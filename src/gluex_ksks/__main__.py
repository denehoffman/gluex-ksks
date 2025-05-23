from modak import TaskQueue

from gluex_ksks.constants import (
    mkdirs,
)
import laddu as ld
from itertools import product

from gluex_ksks.tasks.plot import PlotAll, PlotProtonZCombined

if __name__ == '__main__':
    mkdirs()
    tq = TaskQueue(workers=ld.available_parallelism())
    tq.run(
        [
            PlotProtonZCombined(protonz_cut=False, chisqdof=None, select_mesons=None),
            *[
                PlotAll(
                    data_type='data',
                    protonz_cut=True,
                    chisqdof=cuts[0],
                    select_mesons=cuts[1],
                    method='free',
                    nspec=2,
                )
                for cuts in list(product([None, 2.4, 3.4, 4.4], [True, False, None]))
            ],
        ]
    )
