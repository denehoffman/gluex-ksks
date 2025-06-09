import pickle
from typing import override
from modak import Task
import numpy as np
import matplotlib.pyplot as plt

from gluex_ksks.constants import (
    BEAM_ENERGY_BINS,
    BEAM_ENERGY_RANGE,
    BLUE,
    LOG_PATH,
    PLOTS_PATH,
)
from gluex_ksks.tasks.databases import PSFlux
from gluex_ksks.utils import (
    Histogram,
    PSFluxTable,
    get_all_polarized_run_numbers,
    get_coherent_peak,
)


class PlotFlux(Task):
    def __init__(self):
        super().__init__(
            'plot_flux',
            inputs=[PSFlux()],
            outputs=[
                PLOTS_PATH / 'psflux_tagged_flux.png',
                PLOTS_PATH / 'psflux_tagm_e_flux.png',
                PLOTS_PATH / 'psflux_tagm_flux.png',
                PLOTS_PATH / 'psflux_tagh_e_flux.png',
                PLOTS_PATH / 'psflux_tagh_flux.png',
                PLOTS_PATH / 'psflux_tagged_lumi.png',
            ],
            log_directory=LOG_PATH,
        )

    @override
    def run(self) -> None:
        def ps_accept(x: float, p0: float, p1: float, p2: float) -> float:
            if x > 2.0 * p1 and x < p1 + p2:
                return p0 * (1.0 - 2.0 * p1 / x)
            elif x >= p1 + p2:
                return p0 * (2.0 * p2 / x - 1.0)
            return 0.0

        ps_flux_table: PSFluxTable = pickle.load(self.inputs[0].outputs[0].open('rb'))
        run_numbers = get_all_polarized_run_numbers()
        h_tagged_flux = Histogram.empty(BEAM_ENERGY_BINS, BEAM_ENERGY_RANGE)
        h_tagm_e_flux = Histogram.empty(BEAM_ENERGY_BINS, BEAM_ENERGY_RANGE)
        h_tagm_flux = Histogram.empty(102, (1, 103))
        h_tagh_e_flux = Histogram.empty(BEAM_ENERGY_BINS, BEAM_ENERGY_RANGE)
        h_tagh_flux = Histogram.empty(274, (1, 275))
        h_tagged_lumi = Histogram.empty(BEAM_ENERGY_BINS, BEAM_ENERGY_RANGE)
        self.logger.info('Filling Histograms')
        for run_number in run_numbers:
            coherent_peak = get_coherent_peak(run_number)
            scale = ps_flux_table.df_scale[run_number]
            photon_endpoint_delta_e = (
                ps_flux_table.df_photon_endpoint[run_number]
                - ps_flux_table.df_photon_endpoint_calib[run_number]
                if run_number > 60000
                else 0.0
            )
            ps_accept_pars = ps_flux_table.df_ps_accept[run_number]
            for tagm_flux, tagm_scaled_energy in zip(
                ps_flux_table.df_tagm_tagged_flux_table[run_number],
                ps_flux_table.df_tagm_scaled_energy_table[run_number],
            ):
                tagm_energy = (
                    ps_flux_table.df_photon_endpoint[run_number]
                    * (float(tagm_scaled_energy[1]) + float(tagm_scaled_energy[2]))
                    / 2.0
                ) + photon_endpoint_delta_e
                if not (coherent_peak[0] < tagm_energy < coherent_peak[1]):
                    continue
                psa = ps_accept(tagm_energy, *ps_accept_pars)
                if psa <= 0.0:
                    continue
                ibin = h_tagged_flux.get_bin_index(tagm_energy)
                if ibin is None:
                    continue
                c = float(tagm_flux[1]) * scale / psa
                e = float(tagm_flux[2]) * scale / psa
                h_tagged_flux.counts[ibin] += c
                h_tagged_flux.errors[ibin] = np.sqrt(
                    h_tagged_flux.errors[ibin] ** 2 + e**2
                )
                h_tagm_e_flux.counts[ibin] += c
                h_tagm_e_flux.errors[ibin] = np.sqrt(
                    h_tagged_flux.errors[ibin] ** 2 + e**2
                )
                ibin_htagm = h_tagm_flux.get_bin_index(float(tagm_flux[0]))
                if ibin_htagm is None:
                    continue
                h_tagm_flux.counts[ibin_htagm] += c

            for tagh_flux, tagh_scaled_energy in zip(
                ps_flux_table.df_tagh_tagged_flux_table[run_number],
                ps_flux_table.df_tagh_scaled_energy_table[run_number],
            ):
                tagh_energy = (
                    ps_flux_table.df_photon_endpoint[run_number]
                    * (float(tagh_scaled_energy[1]) + float(tagh_scaled_energy[2]))
                    / 2.0
                ) + photon_endpoint_delta_e
                if not (coherent_peak[0] < tagh_energy < coherent_peak[1]):
                    continue
                psa = ps_accept(tagh_energy, *ps_accept_pars)
                if psa <= 0.0:
                    continue
                ibin = h_tagged_flux.get_bin_index(tagh_energy)
                if ibin is None:
                    continue
                c = float(tagh_flux[1]) * scale / psa
                e = float(tagh_flux[2]) * scale / psa
                h_tagged_flux.counts[ibin] += c
                h_tagged_flux.errors[ibin] = np.sqrt(
                    h_tagged_flux.errors[ibin] ** 2 + e**2
                )
                h_tagh_e_flux.counts[ibin] += c
                h_tagh_e_flux.errors[ibin] = np.sqrt(
                    h_tagged_flux.errors[ibin] ** 2 + e**2
                )
                ibin_htagh = h_tagh_flux.get_bin_index(float(tagh_flux[0]))
                if ibin_htagh is None:
                    continue
                h_tagh_flux.counts[ibin_htagh] += c

            target_scattering_centers = ps_flux_table.df_target_scattering_centers[
                run_number
            ]
            for i in range(h_tagged_flux.nbins):
                if h_tagged_flux.counts[i] <= 0.0:
                    continue
                lumi = h_tagged_flux.counts[i] * target_scattering_centers[0] / 1e12
                flux_err = h_tagged_flux.errors[i] / h_tagged_flux.counts[i]
                target_err = target_scattering_centers[1] / target_scattering_centers[0]
                lumi_err = lumi * np.sqrt(flux_err**2 + target_err**2)
                h_tagged_lumi.counts[i] = lumi
                h_tagged_lumi.errors[i] = lumi_err
        self.logger.info('Plotting Histograms')
        plt.style.use('gluex_ksks.thesis')
        _, ax = plt.subplots()
        ax.stairs(h_tagged_flux.counts, h_tagged_flux.bins, color=BLUE)
        ax.errorbar(
            h_tagged_flux.centers,
            h_tagged_flux.counts,
            yerr=h_tagged_flux.errors,
            fmt='none',
            color=BLUE,
        )
        bin_width = int(
            (BEAM_ENERGY_RANGE[1] - BEAM_ENERGY_RANGE[0]) / BEAM_ENERGY_BINS * 1000
        )
        ax.set_xlabel('Tagged Flux (GeV)')
        ax.set_ylabel(f'Counts / {bin_width} (MeV)')
        ax.set_ylim(0)
        plt.savefig(self.outputs[0])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(h_tagm_e_flux.counts, h_tagm_e_flux.bins, color=BLUE)
        ax.errorbar(
            h_tagm_e_flux.centers,
            h_tagm_e_flux.counts,
            yerr=h_tagm_e_flux.errors,
            fmt='none',
            color=BLUE,
        )
        bin_width = int(
            (BEAM_ENERGY_RANGE[1] - BEAM_ENERGY_RANGE[0]) / BEAM_ENERGY_BINS * 1000
        )
        ax.set_xlabel('Tagged Flux in Microscope (GeV)')
        ax.set_ylabel(f'Counts / {bin_width} (MeV)')
        ax.set_ylim(0)
        plt.savefig(self.outputs[1])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(h_tagm_flux.counts, h_tagm_flux.bins, color=BLUE)
        ax.errorbar(
            h_tagm_flux.centers,
            h_tagm_flux.counts,
            yerr=h_tagm_flux.errors,
            fmt='none',
            color=BLUE,
        )
        ax.set_xlabel('Tagged Flux in each TAGM Column')
        ax.set_ylabel('Counts / Column')
        ax.set_ylim(0)
        plt.savefig(self.outputs[2])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(h_tagh_e_flux.counts, h_tagh_e_flux.bins, color=BLUE)
        ax.errorbar(
            h_tagh_e_flux.centers,
            h_tagh_e_flux.counts,
            yerr=h_tagh_e_flux.errors,
            fmt='none',
            color=BLUE,
        )
        bin_width = int(
            (BEAM_ENERGY_RANGE[1] - BEAM_ENERGY_RANGE[0]) / BEAM_ENERGY_BINS * 1000
        )
        ax.set_xlabel('Tagged Flux in Hodoscope (GeV)')
        ax.set_ylabel(f'Counts / {bin_width} (MeV)')
        ax.set_ylim(0)
        plt.savefig(self.outputs[3])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(h_tagh_flux.counts, h_tagh_flux.bins, color=BLUE)
        ax.errorbar(
            h_tagh_flux.centers,
            h_tagh_flux.counts,
            yerr=h_tagh_flux.errors,
            fmt='none',
            color=BLUE,
        )
        ax.set_xlabel('Tagged Flux in each TAGH Counter')
        ax.set_ylabel('Counts / Counter')
        ax.set_ylim(0)
        plt.savefig(self.outputs[4])
        plt.close()

        _, ax = plt.subplots()
        ax.stairs(h_tagged_lumi.counts, h_tagged_lumi.bins, color=BLUE)
        ax.errorbar(
            h_tagged_lumi.centers,
            h_tagged_lumi.counts,
            yerr=h_tagged_lumi.errors,
            fmt='none',
            color=BLUE,
        )
        bin_width = int(
            (BEAM_ENERGY_RANGE[1] - BEAM_ENERGY_RANGE[0]) / BEAM_ENERGY_BINS * 1000
        )
        ax.set_xlabel('Tagged Luminosity Binned by Beam Energy (GeV)')
        ax.set_ylabel(f'Luminosity (pb${{}}^{{-1}}$ / {bin_width} MeV)')
        ax.set_ylim(0)
        plt.savefig(self.outputs[5])
        plt.close()
