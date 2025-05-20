from __future__ import annotations

from typing import override

import polars as pl
from modak import Task

from gluex_ksks.constants import DATASETS_PATH
from gluex_ksks.utils import get_ccdb, get_rcdb, root_to_parquet


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
        root_path = DATASETS_PATH / self.data_type / f'{self.run_period}.root'
        self.logger.info(f'Converting {root_path} to parquet')
        root_to_parquet(root_path, self.outputs[0])
        self.logger.info(f'Result written to {self.outputs[0]}')


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
        self.logger.info(
            f'Adding beam polarization info and accidental weights to {src}'
        )
        is_mc = self.data_type != 'data'
        src_data = pl.scan_parquet(src)
        rcdb = get_rcdb()
        ccdb = get_ccdb()

        def process(struct) -> dict[str, float | None]:
            run_number = struct['RunNumber']
            e_beam = struct['p4_0_E']
            rf = struct['RF']
            weight = struct['weight']
            new_weight = ccdb.get_accidental_weight(
                run_number, e_beam, rf, weight, is_mc=is_mc
            )
            aux_0_x, aux_0_y, polarized = rcdb.get_eps_xy(run_number, e_beam)
            if not polarized or new_weight == 0:
                aux_0_x = None
                aux_0_y = None
            return {
                'weight': new_weight,
                'aux_0_x': aux_0_x,
                'aux_0_y': aux_0_y,
                'aux_0_z': 0.0,
            }

        dst_data = (
            src_data.sort(['RunNumber', 'EventNumber', 'ChiSqDOF'])
            .group_by(['RunNumber', 'EventNumber'])
            .first()
            .with_columns(
                pl.struct('RunNumber', 'p4_0_E', 'RF', 'weight')
                .map_elements(
                    process,
                    return_dtype=pl.Struct(
                        {
                            'aux_0_x': pl.Float64,
                            'aux_0_y': pl.Float64,
                            'aux_0_z': pl.Float64,
                            'weight': pl.Float64,
                        }
                    ),
                )
                .alias('aux_struct')
            )
            .drop('weight')
            .unnest('aux_struct')
            .filter(pl.col('aux_0_x').is_not_null())
            .with_columns(
                [
                    pl.col('aux_0_x').cast(pl.Float32),
                    pl.col('aux_0_y').cast(pl.Float32),
                    pl.col('aux_0_z').cast(pl.Float32),
                    pl.col('weight').cast(pl.Float32),
                ]
            )
        )
        dst_data = dst_data.collect()
        dst_data.write_parquet(dst)
        self.logger.info(f'Result written to {dst}')


class ChiSqDOF(Task):
    def __init__(self, *, data_type: str, run_period: str, chisqdof: float):
        self.data_type = data_type
        self.run_period = run_period
        self.chisqdof = chisqdof
        super().__init__(
            f'chisqdof_{run_period}_{data_type}',
            inputs=[
                AccidentalsAndPolarization(data_type=data_type, run_period=run_period)
            ],
            outputs=[
                DATASETS_PATH
                / data_type
                / f'{run_period}_ap_chisqdof_{self.chisqdof:.1f}.parquet'
            ],
        )

    @override
    def run(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        self.logger.info(f'Cutting χ²/DOF at {self.chisqdof} for {src}')
        src_data = pl.scan_parquet(src)
        dst_data = src_data.filter(pl.col('ChiSqDOF') < self.chisqdof)
        dst_data = dst_data.collect()
        dst_data.write_parquet(dst)
        self.logger.info(f'Result written to {dst}')
