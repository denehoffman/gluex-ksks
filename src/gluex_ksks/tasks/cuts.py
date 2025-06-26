from __future__ import annotations

from typing import override

import polars as pl
from modak import Task

from gluex_ksks.constants import DATASET_PATH, LOG_PATH, RAW_DATASET_PATH
from gluex_ksks.tasks.databases import CCDB, RCDB
from gluex_ksks.utils import (
    add_ksb_costheta,
    add_m_meson,
    get_all_polarized_run_numbers,
    get_ccdb,
    get_rcdb,
    select_mesons_tag,
    root_to_dataframe,
)


class AccPol(Task):
    """
    Accidental subtraction and polarization
    """

    def __init__(self, *, data_type: str, run_period: str):
        self.data_type = data_type
        self.run_period = run_period
        super().__init__(
            f'accidentals_and_polarization_{self.data_type}_{self.run_period}',
            inputs=[
                RCDB(),
                CCDB(),
            ],
            outputs=[DATASET_PATH[self.data_type] / f'{self.run_period}.parquet'],
            log_directory=LOG_PATH,
        )

    @override
    def run(self):
        srcs = (RAW_DATASET_PATH[self.data_type] / self.run_period).glob('*.root')
        dst = self.outputs[0]
        self.logger.info(
            f'Adding beam polarization info and accidental weights to {self.run_period} {self.data_type}'
        )
        is_mc = self.data_type != 'data'
        rcdb = get_rcdb()
        ccdb = get_ccdb()
        polarized_runs = get_all_polarized_run_numbers()

        def process(struct) -> dict[str, float | None]:
            run_number = struct['RunNumber']
            e_beam = struct['p4_0_E']
            rf = struct['RF']
            weight = struct['weight']
            new_weight = ccdb.get_accidental_weight(
                run_number, e_beam, rf, weight, is_mc=is_mc
            )
            aux_0_x, aux_0_y, polarized = rcdb.get_eps_xy(run_number, e_beam)
            if not polarized or new_weight == 0 or run_number not in polarized_runs:
                aux_0_x = None
                aux_0_y = None
            return {
                'weight': new_weight,
                'aux_0_x': aux_0_x,
                'aux_0_y': aux_0_y,
                'aux_0_z': 0.0,
            }

        for src in srcs:
            dst_path = src.with_suffix('.parquet')
            if dst_path.exists():
                self.logger.info(f'Skipping {src} ({dst_path} already exists)')
                continue
            self.logger.info(f'Converting {src} to polarized parquet')
            src_data = root_to_dataframe(src)
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
                .drop('RunNumber', 'EventNumber')
                # don't need these anymore
            )
            if dst_data.is_empty():
                continue
            dst_data.write_parquet(dst_path)
        self.logger.info(f'Merging parquet files to {dst}')
        dst_data = pl.concat(
            [
                pl.scan_parquet(f)
                for f in (RAW_DATASET_PATH[self.data_type] / self.run_period).glob(
                    '*.parquet'
                )
            ],
            how='diagonal',
            rechunk=True,
        )
        dst_data.sink_parquet(dst)
        self.logger.info(f'Result written to {dst}')


class FiducialCuts(Task):
    def __init__(
        self,
        *,
        data_type: str,
        run_period: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
    ):
        self.data_type = data_type
        self.run_period = run_period
        self.cut_type = None
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = select_mesons_tag(self.select_mesons)
        if self.select_mesons is not None:
            inputs: list[Task] = [
                FiducialCuts(
                    data_type=self.data_type,
                    run_period=self.run_period,
                    protonz_cut=self.protonz_cut,
                    mass_cut=self.mass_cut,
                    chisqdof=self.chisqdof,
                    select_mesons=None,
                )
            ]
            outputs = [
                inputs[0].outputs[0].parent
                / f'{inputs[0].outputs[0].stem}_{self.tag}.parquet'
            ]
        elif self.chisqdof is not None:
            inputs = [
                FiducialCuts(
                    data_type=self.data_type,
                    run_period=self.run_period,
                    protonz_cut=self.protonz_cut,
                    mass_cut=self.mass_cut,
                    chisqdof=None,
                    select_mesons=self.select_mesons,
                )
            ]
            outputs = [
                inputs[0].outputs[0].parent
                / f'{inputs[0].outputs[0].stem}_chisqdof_{self.chisqdof}.parquet'
            ]
        elif self.mass_cut:
            inputs = [
                FiducialCuts(
                    data_type=self.data_type,
                    run_period=self.run_period,
                    protonz_cut=self.protonz_cut,
                    mass_cut=False,
                    chisqdof=self.chisqdof,
                    select_mesons=self.select_mesons,
                )
            ]
            outputs = [
                inputs[0].outputs[0].parent
                / f'{inputs[0].outputs[0].stem}_masscut.parquet'
            ]
        elif self.protonz_cut:
            inputs = [
                FiducialCuts(
                    data_type=self.data_type,
                    run_period=self.run_period,
                    protonz_cut=False,
                    mass_cut=self.mass_cut,
                    chisqdof=self.chisqdof,
                    select_mesons=self.select_mesons,
                )
            ]
            outputs = [
                inputs[0].outputs[0].parent / f'{inputs[0].outputs[0].stem}_pz.parquet'
            ]
        else:  # no cuts
            inputs = [
                AccPol(
                    data_type=self.data_type,
                    run_period=self.run_period,
                )
            ]
            outputs = [inputs[0].outputs[0]]
        super().__init__(
            f'fiducial_cut_{self.data_type}_{self.run_period}_{self.protonz_cut}_{self.mass_cut}_{self.chisqdof}_{self.tag}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        if self.select_mesons is not None:
            self.cut_baryons()
        elif self.chisqdof is not None:
            self.cut_chisqdof()
        elif self.mass_cut:
            self.cut_mass()
        elif self.protonz_cut:
            self.cut_protonz()
        else:
            pass  # no-op

    def cut_protonz(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        self.logger.info(f'Cutting Proton-z for {src}')
        src_data = pl.scan_parquet(src)
        dst_data = src_data.filter(pl.col('Proton_Z').is_between(50, 80))
        dst_data.sink_parquet(dst)
        self.logger.info(f'Result written to {dst}')

    def cut_mass(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        self.logger.info(f'Cutting KsKs Mass for {src}')
        src_data = add_m_meson(pl.scan_parquet(src))
        dst_data = src_data.filter(pl.col('m_meson').lt(2.0)).drop('m_meson')
        dst_data.sink_parquet(dst)
        self.logger.info(f'Result written to {dst}')

    def cut_chisqdof(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        self.logger.info(f'Cutting χ²/DOF at {self.chisqdof} for {src}')
        src_data = pl.scan_parquet(src)
        dst_data = src_data.filter(pl.col('ChiSqDOF') < self.chisqdof)
        dst_data.sink_parquet(dst)
        self.logger.info(f'Result written to {dst}')

    def cut_baryons(self):
        src = self.inputs[0].outputs[0]
        dst = self.outputs[0]
        self.logger.info(f'Cutting Backward K_S^0 cos(θ) at 0 for {src}')
        src_data = add_ksb_costheta(pl.scan_parquet(src))

        if self.select_mesons:
            dst_data = src_data.filter(pl.col('ksb_costheta') >= 0.0)
        else:
            dst_data = src_data.filter(pl.col('ksb_costheta') < 0.0)
        dst_data = dst_data.drop('ksb_costheta')
        dst_data.sink_parquet(dst)
        self.logger.info(f'Result written to {dst}')
