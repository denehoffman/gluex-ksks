from __future__ import annotations
import pickle
from typing import override
from modak import Task
from scipy.optimize import minimize, OptimizeResult

from gluex_ksks.constants import FITS_PATH, LOG_PATH, RUN_PERIODS
from gluex_ksks.tasks.cuts import FiducialCuts
import numpy as np

from gluex_ksks.types import FloatArray
from gluex_ksks.utils import (
    FactorizationResult,
    FitResult,
    SPlotArrays,
    add_m_meson,
    exp_pdf,
    get_bkgmc_lda0s_list,
    get_quantile_edges,
    get_quantile_indices,
    get_sigmc_fit_components,
    select_mesons_tag,
)
import polars as pl


def run_factorization_fits(
    arrays_data: SPlotArrays,
    arrays_sigmc: SPlotArrays,
    arrays_bkgmc: SPlotArrays,
    *,
    nspec: int,
    nbins: int,
) -> FactorizationResult:
    quantile_edges = get_quantile_edges(
        arrays_data.control, bins=nspec, weights=arrays_data.weight
    )
    quantile_indices = get_quantile_indices(
        arrays_data.control, bins=nspec, weights=arrays_data.weight
    )

    sigmc_fit_components = get_sigmc_fit_components(
        arrays=arrays_sigmc,
        nbins=nbins,
    )

    bkgmc_lda0s = get_bkgmc_lda0s_list(
        arrays=arrays_bkgmc,
        mass_bins=quantile_edges,
    )

    def generate_nll(
        rfl1: FloatArray,
        rfl2: FloatArray,
        weight: FloatArray,
    ):
        def nll(args: FloatArray) -> float:
            z = args[0]
            lda_b = args[1]
            likelihoods: FloatArray = weight * np.log(
                z * sigmc_fit_components.pdf(rfl1, rfl2)
                + (1 - z) * exp_pdf(rfl1=rfl1, rfl2=rfl2, lda=lda_b)
                + np.finfo(float).tiny
            )
            return float(
                -2.0 * np.sum(np.sort(likelihoods))
            )  # the integral term doesn't matter here since we are using this in a ratio where it cancels

        return nll

    nlls = [
        generate_nll(
            arrays_data.rfl1[quantile_indices[i]],
            arrays_data.rfl2[quantile_indices[i]],
            arrays_data.weight[quantile_indices[i]],
        )
        for i in range(nspec)
    ]

    # arguments are (z_0, z_1, ..., z_{n-1}, lda_b)
    def nll0(args: FloatArray) -> float:
        return np.sum(
            np.array([nlls[i](np.array([args[i], args[-1]])) for i in range(nspec)])
        )

    h0_x0 = ([0.5] * nspec) + [90.0]
    h0_bounds = ([(0.0, 1.0)] * nspec) + [(70.0, 130.0)]

    opt_h0: OptimizeResult = minimize(nll0, x0=h0_x0, bounds=h0_bounds)

    opt_h1s: list[OptimizeResult] = []
    for i in range(nspec):
        h1_x0 = [0.5, bkgmc_lda0s[i]]
        h1_bounds = [(0.0, 1.0), (70.0, 130.0)]
        opt_h1: OptimizeResult = minimize(nlls[i], x0=h1_x0, bounds=h1_bounds)
        opt_h1s.append(opt_h1)

    return FactorizationResult(
        FitResult.from_opt(opt_h0, len(arrays_data)),
        [
            FitResult.from_opt(opt_h1s[i], len(quantile_indices[i]))
            for i in range(nspec)
        ],
        nspec - 1,
    )


class FactorizationFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        nspec: int,
    ):
        self.protonz_cut = protonz_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.nspec = nspec
        inputs: list[Task] = [
            *[
                FiducialCuts(
                    data_type='data',
                    run_period=run_period,
                    protonz_cut=protonz_cut,
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
                    chisqdof=chisqdof,
                    select_mesons=select_mesons,
                )
                for run_period in RUN_PERIODS
            ],
        ]
        self.tag = select_mesons_tag(select_mesons)
        outputs = [
            FITS_PATH
            / f'factorization_fit{"_pz" if self.protonz_cut else ""}{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{self.tag}.pkl'
        ]
        super().__init__(
            name=f'factorization_fit{"_pz" if self.protonz_cut else ""}{f"_chisqdof_{self.chisqdof:.1f}" if self.chisqdof is not None else ""}{self.tag}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        self.logger.info(
            f'Running factorization fit (nspec={self.nspec}) with pz={self.protonz_cut}, chisqdof={self.chisqdof}, select={self.tag}'
        )
        arrays_data = SPlotArrays.from_polars(
            add_m_meson(
                pl.concat(
                    [
                        pl.read_parquet(inp.outputs[0])
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
                        pl.read_parquet(inp.outputs[0])
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
                        pl.read_parquet(inp.outputs[0])
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
        fit_result = run_factorization_fits(
            arrays_data,
            arrays_sigmc,
            arrays_bkgmc,
            nspec=self.nspec,
            nbins=200,
        )
        pickle.dump(fit_result, self.outputs[0].open('wb'))
