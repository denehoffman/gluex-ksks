import itertools
from typing import Literal, override
import matplotlib.pyplot as plt

import pickle
import polars as pl
from modak import Task
from num2words import num2words
from scipy.optimize import minimize
from gluex_ksks.constants import (
    BLUE,
    FITS_PATH,
    GREEN,
    LOG_PATH,
    PLOTS_PATH,
    PURPLE,
    RED,
    REPORTS_PATH,
    RFL_BINS,
    RFL_RANGE,
    RUN_PERIODS,
)
from gluex_ksks.tasks.cuts import FiducialCuts
from gluex_ksks.types import FloatArray
from gluex_ksks.utils import (
    FitResult,
    SPlotArrays,
    SPlotFitResult,
    add_m_meson,
    exp_pdf,
    exp_pdf_single,
    get_bkgmc_lda0s_list,
    get_quantile_edges,
    get_sigmc_fit_components,
    select_mesons_tag,
    to_latex,
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
    weighted_total = np.sum(
        np.sum(
            [arrays_data.weight * v[0, j] * pdfs[j] for j in range(nspec + 1)], axis=0
        )
        / denom
    )
    return SPlotFitResult(
        sigmc_fit_components,
        yields,
        ldas,
        FitResult.from_opt(opt, len(arrays_data)),
        v,
        weighted_total,
    )


class SPlotFit(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        method: Literal['fixed', 'free'] | str,
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
            / f'splot_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}_{self.tag}_{self.method}_{self.nspec}.pkl',
            REPORTS_PATH
            / f'splot_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}_{self.tag}_{self.method}_{self.nspec}.tex',
            PLOTS_PATH
            / f'splot_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}_{self.tag}_{self.method}_{self.nspec}.png',
        ]
        super().__init__(
            name=f'splot_fit{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}_{self.tag}_{self.method}_{self.nspec}',
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

        if self.fixed:
            par_names = ['s'] + [f'b{i + 1}' for i in range(self.nspec)]
        else:
            par_names = ['s'] + list(
                itertools.chain(
                    *zip(
                        [f'b{i + 1}' for i in range(self.nspec)],
                        [f'l{i + 1}' for i in range(self.nspec)],
                    )
                )
            )
        output_str = r"""
\begin{table}[ht]
    \begin{center}
        \begin{tabular}{lr}\toprule
            Parameter & Value \\\midrule"""
        for par_name, value, error in zip(
            par_names,
            fit_result.total_fit.values,
            fit_result.total_fit.errors,
        ):
            if par_name.startswith('s'):
                normalized_name = 'Signal Yield'
            elif par_name.startswith('b'):
                normalized_name = rf'Background Yield $\#{par_name[1:]}$'
            else:
                normalized_name = rf'Background $\lambda$ $\#{par_name[1:]}$'
            output_str += f"""
            {normalized_name} & {to_latex(value, error)} \\\\"""
        output_str += rf"""\bottomrule
        \end{{tabular}}
        \caption{{The parameter values and uncertainties for the sPlot fit of data with $\chi^2_\nu < {self.chisqdof:.2f}$ using {num2words(self.nspec)} {self.method} background slope{'s' if self.nspec > 1 else ''}. Uncertainties are calculated using the covariance matrix of the fit.{r' All $\lambda$ parameters have units of $\si{\nano\second}^{-1}$.' if self.method == 'free' else ''}}}\label{{tab:splot-fit-results-chisqdof-{self.chisqdof:.2f}-{self.method}-{self.nspec}}}
    \end{{center}}
\end{{table}}
% {fit_result.weighted_total} weighted events"""
        self.outputs[1].write_text(output_str)
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.hist(
            arrays_data.rfl1,
            weights=arrays_data.weight,
            bins=RFL_BINS,
            range=RFL_RANGE,
            histtype='step',
            color=BLUE,
        )
        bin_width = (RFL_RANGE[1] - RFL_RANGE[0]) / RFL_BINS
        rfls = np.linspace(*RFL_RANGE, 1000)
        sigmc_fit_components = get_sigmc_fit_components(
            arrays=arrays_sigmc,
        )
        sig_line = sigmc_fit_components.pdf1(rfls) * bin_width * fit_result.sig_yield
        ax.plot(
            rfls,
            sig_line,
            color=GREEN,
            label='Signal Component',
        )
        bkg_lines = []
        for i, (bkg_yield, bkg_lda) in enumerate(
            zip(fit_result.bkg_yields, fit_result.bkg_ldas)
        ):
            bkg_line = exp_pdf_single(rfl=rfls, lda=bkg_lda) * bkg_yield * bin_width
            bkg_lines.append(bkg_line)
            ax.plot(
                rfls,
                bkg_line,
                color=RED,
                label='Background Component' if i == 0 else None,
            )
        total_line = sig_line + sum(bkg_lines)
        ax.plot(
            rfls,
            total_line,
            color=PURPLE,
            label='Fit Total',
        )
        ax.set_xlabel(r'$K_S^0$ Rest Frame Lifetime (ns)')
        bin_width_ps = int((RFL_RANGE[1] - RFL_RANGE[0]) / RFL_BINS * 1000)
        ax.set_ylabel(f'Counts / {bin_width_ps} (ps)')
        ax.set_ylim(10)
        ax.set_yscale('log')
        ax.legend()
        plt.savefig(self.outputs[2])
        plt.close()


class SPlotReport(Task):
    def __init__(
        self,
        *,
        protonz_cut: bool,
        mass_cut: bool,
        chisqdof: float | None,
        select_mesons: bool | None,
        max_nspec: int,
    ):
        self.protonz_cut = protonz_cut
        self.mass_cut = mass_cut
        self.chisqdof = chisqdof
        self.select_mesons = select_mesons
        self.tag = 'None'
        if self.select_mesons is not None:
            self.tag = '_mesons' if self.select_mesons else '_baryons'
        self.max_nspec = max_nspec
        inputs: list[Task] = [
            SPlotFit(
                protonz_cut=self.protonz_cut,
                mass_cut=self.mass_cut,
                chisqdof=self.chisqdof,
                select_mesons=self.select_mesons,
                method=method,
                nspec=nspec,
            )
            for method in ['free', 'fixed']
            for nspec in range(1, self.max_nspec + 1)
        ]

        outputs = [
            REPORTS_PATH
            / f'splot_report{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}{self.tag}_max_nspec_{self.max_nspec}.tex'
        ]
        super().__init__(
            name=f'splot_report{"_pz" if self.protonz_cut else ""}{"_masscut" if self.mass_cut else ""}{f"_chisqdof_{self.chisqdof}" if self.chisqdof is not None else ""}{self.tag}_max_nspec_{self.max_nspec}',
            inputs=inputs,
            outputs=outputs,
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        fit_results: list[SPlotFitResult] = [
            pickle.load(inp.outputs[0].open('rb')) for inp in self.inputs
        ]
        min_aic = np.inf
        min_bic = np.inf
        for fit_result in fit_results:
            if fit_result.aic < min_aic:
                min_aic = fit_result.aic
            if fit_result.bic < min_bic:
                min_bic = fit_result.bic
        output_str = r"""\begin{table}[ht]
    \begin{center}
        \begin{tabular}{ccccc}\toprule
        Background Slope Parameters & Number of Background Components & $r\text{AIC}$ & $r\text{BIC}$\\\midrule"""
        current_method = None
        for (method, nspec), fit_result in zip(
            [(m, n) for m in ['Free', 'Fixed'] for n in range(1, self.max_nspec + 1)],
            fit_results,
        ):
            method_string = ''
            if method != current_method:
                current_method = method
                method_string = current_method
            output_str += f"""
        {method_string} & {nspec} & {fit_result.aic - min_aic:.3f} & {fit_result.bic - min_bic:.3f} \\\\"""
        output_str += r"""\bottomrule
        \end{tabular}
        \caption{Relative AIC and BIC values for each sPlot fitting method. ``Fixed'' and ``free'' refer to whether the background slope parameters are fixed to values obtained from background Monte Carlo or are free parameters in the fit.}\label{tab:splot-model-results}
    \end{center}
\end{table}"""
        self.outputs[0].write_text(output_str)


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
        method: Literal['fixed', 'free'] | str,
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
