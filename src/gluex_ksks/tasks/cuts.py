from typing import override

import polars as pl
from modak import Task

from gluex_ksks.constants import DATASETS_PATH
from gluex_ksks.utils import root_to_parquet


class ConvertToParquet(Task):
    def __init__(self, *, data_type: str, run_period: str):
        self.data_type = data_type
        self.run_period = run_period
        super().__init__(
            f'convert_to_parquet_{run_period}_{data_type}',
            outputs=[DATASETS_PATH / data_type / f'{run_period}.parquet'],
        )

    @override
    def run(self):
        root_to_parquet(DATASETS_PATH / self.data_type / f'{self.run_period}.root')


def compute_accidental_weight(row):
    run_number, e_beam, rf, weight = row
    return


class AccidentalsAndPolarization(Task):
    def __init__(self, *, data_type: str, run_period: str):
        self.data_type = data_type
        super().__init__(
            f'accidentals_and_polarization_{run_period}_{data_type}',
            inputs=[ConvertToParquet(data_type=data_type, run_period=run_period)],
            outputs=[DATASETS_PATH / data_type / f'{run_period}_ap.parquet'],
        )

    @override
    def run(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        is_mc = self.data_type != 'data'
        best_combo_map = {}
        best_combo_chi2_map = {}
        src_data = pl.read_parquet(src)
        no_amo_src_data = (
            src_data.sort(['RunNumber', 'EventNumber', 'ChiSqDOF']).group_by(['RunNumber', 'EventNumber']).first()
        )


# class ChiSqDOF(Task):
#     def __init__(self,*, data_type: str,  run_period: str, chisqdof: float):
#         self.data_type = data_type
#         self.run_period = run_period
#         self.chisqdof = chisqdof
#         super().__init__(f'chisqdof_{run_period}_{data_type}', outputs=[DATASETS_PATH / data_type / f'{run_period}.chi_sq_dof']
