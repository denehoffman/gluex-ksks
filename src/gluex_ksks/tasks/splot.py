import itertools
from typing import Literal, override

import pickle
import polars as pl
from modak import Task
from scipy.optimize import minimize
from gluex_ksks.constants import FITS_PATH, LOG_PATH, RUN_PERIODS
from gluex_ksks.tasks.cuts import FiducialCuts
from gluex_ksks.types import FloatArray
from gluex_ksks.utils import (
    FitResult,
    SPlotArrays,
    SPlotFitResult,
    add_m_meson,
    exp_pdf,
    get_bkgmc_lda0s_list,
    get_quantile_edges,
    get_sigmc_fit_components,
    select_mesons_tag,
)
import numpy as np


def run_splot_fit(
    arrays_data: SPlotArrays,
    arrays_sigmc: SPlotArrays,
    arrays_bkgmc: SPlotArrays,
    *,
    nspec: int,
    fixed: bool,
    logger,
) -> SPlotFitResult:
    quantile_edges = get_quantile_edges(
        arrays_data.control, bins=nspec, weights=arrays_data.weight
    )
    sigmc_fit_components = get_sigmc_fit_components(
        arrays=arrays_sigmc,
    )
    sigmc_pdf_evaluated = sigmc_fit_components.pdf(arrays_data.rfl1, arrays_data.rfl2)

    bkgmc_lda0s = get_bkgmc_lda0s_list(
        arrays=arrays_bkgmc,
        mass_bins=quantile_edges,
    )

    nevents = arrays_data.nevents
    yield_sig0 = [nevents / (nspec + 1)]
    yields_bkg0 = [nevents / (nspec + 1)] * nspec

    def nll(args: FloatArray) -> float:
        yield_sig = [args[0]]
        yields_bkg = list(args[1::2])
        ldas = list(args[2::2])
        yields = yield_sig + yields_bkg
        likelihoods: FloatArray = arrays_data.weight * np.log(
            (yield_sig[0] * sigmc_pdf_evaluated)
            + np.sum(
                [
                    yields_bkg[i]
                    * exp_pdf(rfl1=arrays_data.rfl1, rfl2=arrays_data.rfl2, lda=ldas[i])
                    for i in range(nspec)
                ],
                axis=0,
            )
            + np.finfo(float).tiny
        )
        return -2 * (np.sum(np.sort(likelihoods)) - np.sum(yields))

    def nll_fixed(args: FloatArray) -> float:
        yield_sig = [args[0]]
        yields_bkg = list(args[1:])
        ldas = bkgmc_lda0s
        yields = yield_sig + yields_bkg
        likelihoods: FloatArray = arrays_data.weight * np.log(
            (yield_sig[0] * sigmc_pdf_evaluated)
            + np.sum(
                [
                    yields_bkg[i]
                    * exp_pdf(rfl1=arrays_data.rfl1, rfl2=arrays_data.rfl2, lda=ldas[i])
                    for i in range(nspec)
                ],
                axis=0,
            )
            + np.finfo(float).tiny
        )
        return -2 * (np.sum(np.sort(likelihoods)) - np.sum(yields))

    yield_sig_bounds = [(0.0, None)]
    yields_bkg_bounds = [(0.0, None)] * nspec
    ldas_bounds = [
        (max(bkgmc_lda0 - 50.0, 0.01), bkgmc_lda0 + 50.0) for bkgmc_lda0 in bkgmc_lda0s
    ]
    if fixed:
        x0 = yield_sig0 + yields_bkg0
        bounds = [(0.0, None)] * (nspec + 1)
        opt = minimize(nll_fixed, x0, bounds=bounds)
        yields = list(opt.x[:])
        ldas = bkgmc_lda0s
    else:
        x0 = yield_sig0 + list(itertools.chain(*zip(yields_bkg0, bkgmc_lda0s)))
        bounds = yield_sig_bounds + list(
            itertools.chain(*zip(yields_bkg_bounds, ldas_bounds))
        )
        opt = minimize(nll, x0, bounds=bounds)
        yields: list[float] = [opt.x[0]] + list(opt.x[1::2])
        ldas: list[float] = list(opt.x[2::2])
    pdfs = [sigmc_pdf_evaluated] + [
        exp_pdf(rfl1=arrays_data.rfl1, rfl2=arrays_data.rfl2, lda=ldas[i])
        for i in range(nspec)
    ]
    denom: FloatArray = np.sum([yields[k] * pdfs[k] for k in range(nspec + 1)], axis=0)
    inds = np.argwhere(
        np.power(denom, 2) == 0.0
    )  # if a component is very small, this can happen
    denom[inds] += np.sqrt(
        np.finfo(float).eps
    )  # push these values just lightly away from zero
    v_inv = np.array(
        [
            [
                np.sum((arrays_data.weight * pdfs[i] * pdfs[j]) / np.power(denom, 2))
                for j in range(nspec + 1)
            ]
            for i in range(nspec + 1)
        ]
    )
    logger.info(f'v_inv =\n{v_inv}')
    v: FloatArray = np.linalg.inv(v_inv)
    logger.info(f'v =\n{v}')
    logger.info(f'λs = {ldas}')
    return SPlotFitResult(
        sigmc_fit_components, yields, ldas, FitResult.from_opt(opt, len(arrays_data)), v
    )


class SPlotFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'],
        nspec: int,
    ):
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
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
        self.tag = select_mesons_tag(self.select_mesons)
        outputs = [
            FITS_PATH
            / f'splot_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}{self.tag}_{self.method}_{self.nspec}.pkl'
        ]
        super().__init__(
            name=f'factorization_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}{self.tag}_{self.method}_{self.nspec}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info(
            f'Running sPlot fit (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, mass_cut={self.mass_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )
        arrays_data = SPlotArrays.from_polars(
            add_m_meson(
                pl.concat(
                    [
                        pl.scan_parquet(inp.outputs[0])
                        for inp in self.inputs[: len(RUN_PERIODS)]
                    ],
                    how='diagonal',
                    rechunk=True,
                ),
            ),
            control='m_meson',
        )
        arrays_sigmc = SPlotArrays.from_polars(
            add_m_meson(
                pl.concat(
                    [
                        pl.scan_parquet(inp.outputs[0])
                        for inp in self.inputs[len(RUN_PERIODS) : 2 * len(RUN_PERIODS)]
                    ],
                    how='diagonal',
                    rechunk=True,
                ),
            ),
            control='m_meson',
        )
        arrays_bkgmc = SPlotArrays.from_polars(
            add_m_meson(
                pl.concat(
                    [
                        pl.scan_parquet(inp.outputs[0])
                        for inp in self.inputs[
                            2 * len(RUN_PERIODS) : 3 * len(RUN_PERIODS)
                        ]
                    ],
                    how='diagonal',
                    rechunk=True,
                ),
            ),
            control='m_meson',
        )
        fit_result = run_splot_fit(
            arrays_data,
            arrays_sigmc,
            arrays_bkgmc,
            nspec=self.nspec,
            fixed=self.fixed,
            logger=self.logger,
        )
        pickle.dump(fit_result, self.outputs[0].open('wb'))


def get_sweights(
    fit_result: SPlotFitResult,
    *,
    rfl1: FloatArray,
    rfl2: FloatArray,
    weight: FloatArray,
    nspec: int,
    logger,
) -> FloatArray:
    yields = fit_result.yields
    pdfs = fit_result.pdfs(rfl1, rfl2)
    denom: FloatArray = np.sum([yields[k] * pdfs[k] for k in range(nspec + 1)], axis=0)
    inds = np.argwhere(
        np.power(denom, 2) == 0.0
    )  # if a component is very small, this can happen
    denom[inds] += np.sqrt(
        np.finfo(float).eps
    )  # push these values just lightly away from zero
    v = fit_result.v
    logger.debug(f'sPlot Result: v = {v}')
    return (
        np.sum([weight * v[0, j] * pdfs[j] for j in range(nspec + 1)], axis=0) / denom
    )


class SPlotWeights(Task):
    def __init__(
        self,
        *,
        run_period: str,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'],
        nspec: int,
    ):
        self.run_period = run_period
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = 'None'
        if self.select_mesons is not None:
            self.tag = '_mesons' if self.select_mesons else '_baryons'
        self.method = method
        self.fixed = method == 'fixed'
        self.nspec = nspec
        inputs: list[Task] = [
            SPlotFit(
                protonz_cut=self.protonz_cut,
                mass_cut=self.mass_cut,
                chisqdof=self.chisqdof,
                select_mesons=self.select_mesons,
                method=self.method,
                nspec=self.nspec,
            ),
            FiducialCuts(
                data_type='data',
                run_period=self.run_period,
                protonz_cut=self.protonz_cut,
                mass_cut=self.mass_cut,
                chisqdof=self.chisqdof,
                select_mesons=self.select_mesons,
            ),
        ]

        outputs = [
            inputs[1].outputs[0].parent
            / f'{inputs[1].outputs[0].stem}_{self.method}_{self.nspec}.parquet'
        ]
        super().__init__(
            name=f'splot_weights_{self.run_period}_{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}{self.tag}_{self.method}_{self.nspec}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info(
            f'Running sPlot weights (method={self.method}, nspec={self.nspec}) with pz={self.protonz_cut}, mass_cut={self.mass_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )
        fit_result: SPlotFitResult = pickle.load(self.inputs[0].outputs[0].open('rb'))
        data = pl.read_parquet(self.inputs[1].outputs[0])
        rfl1 = data['RFL1'].to_numpy()
        rfl2 = data['RFL2'].to_numpy()
        weight = data['weight'].to_numpy()
        new_weights = get_sweights(
            fit_result,
            rfl1=rfl1,
            rfl2=rfl2,
            weight=weight,
            nspec=self.nspec,
            logger=self.logger,
        )
        data = data.with_columns(
            pl.Series(name='weight', values=new_weights, dtype=pl.Float32)
        )
        data.write_parquet(self.outputs[0])
