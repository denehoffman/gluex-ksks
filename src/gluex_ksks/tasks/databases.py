import pickle
import shutil
import sqlite3
from pathlib import Path
from typing import override

import uproot
from modak import Task
from uproot.behaviors.TBranch import HasBranches
from uproot.reading import ReadOnlyDirectory

from gluex_ksks.constants import (
    LOG_PATH,
    MISC_PATH,
    RCDB_SELECTION_PREFIX,
    RCDB_SELECTION_SUFFIX,
)
from gluex_ksks.utils import (
    CCDBData,
    Histogram,
    PSFluxTable,
    RCDBData,
    ScalingFactors,
    get_ccdb_table,
    get_pol_angle,
    get_rcdb_text_condition,
    get_run_period,
)


class CCDB(Task):
    def __init__(self):
        MISC_PATH.mkdir(exist_ok=True, parents=True)
        super().__init__(
            'ccdb',
            outputs=[MISC_PATH / 'ccdb.pkl'],
            log_directory=LOG_PATH,
        )

    @override
    def run(self):
        with sqlite3.connect(str(MISC_PATH / 'ccdb.sqlite')) as ccdb:
            cursor = ccdb.cursor()
            query = """
            SELECT rr.runMin, rr.runMax, cs.vault
            FROM directories d
            JOIN typeTables tt ON d.id = tt.directoryId
            JOIN constantSets cs ON tt.id = cs.constantTypeId
            JOIN assignments a ON cs.id = a.constantSetId
            JOIN runRanges rr ON a.runRangeId = rr.id
            LEFT JOIN variations v ON a.variationId = v.id
            WHERE d.name = 'ANALYSIS'
            AND tt.name = 'accidental_scaling_factor'
            AND v.name IS 'default'
            ORDER BY rr.runMin, a.created DESC
            """
            cursor.execute(query)
            asf_results = cursor.fetchall()
            factors = {}
            for run_min, run_max, vault in asf_results:
                data = [float(v) for v in vault.split('|')]
                fb = tuple(data[:8])
                scale_factors = ScalingFactors(fb[0], fb[2], fb[4], fb[6], fb[7])
                for run in range(run_min, run_max + 1):
                    factors[run] = scale_factors
            pickle.dump(
                CCDBData(factors),
                (self.outputs[0]).open('wb'),
            )


class RCDB(Task):
    def __init__(self):
        MISC_PATH.mkdir(exist_ok=True, parents=True)
        super().__init__(
            'rcdb',
            outputs=[MISC_PATH / 'rcdb.pkl'],
            log_directory=LOG_PATH,
        )

    @override
    def run(self):
        angles = {}
        with sqlite3.connect(str(MISC_PATH / 'rcdb.sqlite')) as rcdb:
            cursor = rcdb.cursor()
            query = f"""
            {RCDB_SELECTION_PREFIX}
            SELECT r.number, c.float_value
            FROM conditions c
            JOIN condition_types ct ON c.condition_type_id = ct.id
            JOIN runs r ON c.run_number = r.number
            WHERE ct.name = 'polarization_angle'
            {RCDB_SELECTION_SUFFIX}
            ORDER BY r.number
            """
            cursor.execute(query)
            pol_angle_results = cursor.fetchall()
            for run_number, angle_deg in pol_angle_results:
                run_period = get_run_period(run_number)
                pol_angle = get_pol_angle(run_period, str(angle_deg))
                if pol_angle:
                    angles[run_number] = (
                        run_period,
                        str(angle_deg).split('.')[0],
                        pol_angle,
                    )
        magnitudes = {}
        pol_hists = {
            's17': MISC_PATH / 's17.root',
            's18': MISC_PATH / 's18.root',
            'f18': MISC_PATH / 'f18.root',
            's20': MISC_PATH / 's20.root',
        }
        for rp, hist_path in pol_hists.items():
            hists = {}
            tfile = uproot.open(hist_path)  # pyright:ignore[reportUnknownVariableType]
            for pol in ['0', '45', '90', '135']:
                hist = tfile[f'hPol{pol}']
                if isinstance(hist, HasBranches | ReadOnlyDirectory):
                    self.logger.error(f'Error reading histograms from {hist_path}')
                    msg = f'Error reading histograms from {hist_path}'
                    raise OSError(msg)
                mags, edges = hist.to_numpy()
                hists[pol] = Histogram(mags, edges)
            magnitudes[rp] = hists
        pickle.dump(
            RCDBData(angles, magnitudes),
            (self.outputs[0]).open('wb'),
        )


class PSFlux(Task):
    def __init__(self):
        MISC_PATH.mkdir(exist_ok=True, parents=True)
        super().__init__(
            'psflux',
            outputs=[MISC_PATH / 'psflux.pkl'],
            log_directory=LOG_PATH,
        )

    @override
    def run(self):
        self.logger.info('Querying PS-Flux data')
        df_livetime_ratio_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/trig_live'
        )
        self.logger.info('Received livetime ratio data')
        df_livetime_ratio: dict[int, float] = {}
        for k, v in df_livetime_ratio_table.items():
            if float(v[3][1]) > 0.0:
                df_livetime_ratio[k] = float(v[0][1]) / float(v[3][1])
            else:
                df_livetime_ratio[k] = 1.0
        converter_thickness = get_rcdb_text_condition('polarimeter_converter')
        self.logger.info('Received converter thickness data')
        berillium_radiation_length = 35.28e-2
        df_radiation_length: dict[int, float] = {
            k: (75e-6 if v == 'Be 75um' else 750e-6) / berillium_radiation_length
            for k, v in converter_thickness.items()
        }
        df_scale = {
            k: df_livetime_ratio.get(k, 1.0) * 1.0 / (7.0 / 9.0 * v)
            for k, v in df_radiation_length.items()
        }
        df_ps_accept_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/PS_accept'
        )
        self.logger.info('Received PS acceptance parameters')
        df_ps_accept: dict[int, tuple[float, float, float]] = {}
        for k, v in df_ps_accept_table.items():
            df_ps_accept[k] = (float(v[0][0]), float(v[0][1]), float(v[0][2]))

        df_photon_endpoint: dict[int, float] = {
            k: float(v[0][0])
            for k, v in get_ccdb_table('PHOTON_BEAM/endpoint_energy').items()
        }
        self.logger.info('Received endpoint energy')
        df_tagm_tagged_flux_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/tagm/tagged'
        )
        self.logger.info('Received tagged microscope luminosity')
        df_tagm_scaled_energy_table = get_ccdb_table(
            'PHOTON_BEAM/microscope/scaled_energy_range'
        )
        self.logger.info('Received microscope scaled energy')
        df_tagh_tagged_flux_table = get_ccdb_table(
            'PHOTON_BEAM/pair_spectrometer/lumi/tagh/tagged'
        )
        self.logger.info('Received tagged hodoscope luminosity')
        df_tagh_scaled_energy_table = get_ccdb_table(
            'PHOTON_BEAM/hodoscope/scaled_energy_range'
        )
        self.logger.info('Received hodoscope scaled energy')
        df_photon_endpoint_calib: dict[int, float] = {
            k: float(v[0][0])
            for k, v in get_ccdb_table('PHOTON_BEAM/hodoscope/endpoint_calib').items()
        }
        self.logger.info('Received calibrated energy endpoints for Phase-II data')
        target_length = 29.5
        target_factor = target_length * 6.02214e23 * 1e-24 * 1e-3
        df_target_scattering_centers: dict[int, tuple[float, float]] = {
            k: (float(v[0][0]) * target_factor, float(v[0][1]) * target_factor)
            for k, v in get_ccdb_table('TARGET/density', use_timestamp=False).items()
        }
        self.logger.info('Received target density data')
        pickle.dump(
            PSFluxTable(
                df_scale,
                df_ps_accept,
                df_photon_endpoint,
                df_tagm_tagged_flux_table,
                df_tagm_scaled_energy_table,
                df_tagh_tagged_flux_table,
                df_tagh_scaled_energy_table,
                df_photon_endpoint_calib,
                df_target_scattering_centers,
            ),
            (self.outputs[0]).open('wb'),
        )

        # loop over runs
        #   select PARA/PERP/ANGLE
        #   get livetime_ratio from CCDB by run number = PHOTON_BEAM/pair_spectrometer/lumi/trig_live
        # for num_cols, run_min, run_max, vault in asf_results:
        #     data = [float(vault.split("|")[i: i + num_cols]) for i in range(0, len(vault.split("|"), num_cols))]
        #     for run in range(run_min, run_max + 1):
        #         factors[run] = data[0][1] / data[3][1] if data[3][1] > 0.0 else 1.0
        # in RCDB:
        #   SELECT r.number, c.float_value
        #   FROM conditions c
        #   JOIN condition_types ct ON c.condition_type_id = ct.id
        #   JOIN runs r ON c.run_number = r.number
        #   WHERE ct.name = 'polarization_angle'
        #   ORDER BY r.number
        #   converter_length = 75e-6 if "Be 75um"
        #                    = 750e-6 if "Be 750um"
        #   berillium_radlen = 35.28e-2
        #   radlen = converter_length / berillium_radlen
        #   scale = livetime_ratio * 1 / (7/9 * radlen)
        #
        #   Get PHOTON_BEAM/pair_spectrometer/lumi/PS_accept -> function(t[0][0], t[0][1], t[0][2])
        #   def psaccept(x, p0, p1, p2):
        #       if x > 2 * p1 and x < p1 + p2:
        #           return p0 * (1 - 2 * p1 / x)
        #       elif x >= p1 + p2:
        #           return p0 * (2 * p2 / x - 1)
        #       return 0
        #
        #   photon_endpoint = PHOTON_BEAM/endoint_energy
        #   tagm_tagged_flux = PHOTON_BEAM/pair_spectrometer/lumi/tagm/tagged
        #   tagm_scaled_energy = PHOTON_BEAM/microscope/scaled_energy_range
        #   tagh_tagged_flux = PHOTON_BEAM/pair_spectrometer/lumi/tagh/tagged
        #   tagh_scaled_energy = PHOTON_BEAM/hodoscope/scaled_energy_range
        #   calibrated_endpoint = False
        #   if run number > 60000 (phase II):
        #       photon_endpoint_calib = PHOTON_BEAM/hodoscope/endpoint_calib
        #       photon_endpoint_delta_E = float(photon_endpoint[0][0]) - float(photon_endpoint_calib[0][0])
        #       calibrated_endpoint = True
        #   for tagm_flux, tagm_scaled_energy:
        #       tagm_energy = photon_endpoint[0][0] * (tagm_scaled_energy[1] + tagm_scaled_energy[2]) / 2
        #       if calibrated_endpoint:
        #           tagm_energy += photon_endpoint_delta_E
        # ps_accept = func(tagm_energy)
        # if ps_accept <= 0:
        #   continue
        # Then get the bin for htagged_flux_err and get the previous content and error
        # Calc the new content and error from:
        # content = tagm_flux[1] * scale / ps_accept
        # err = tagm_flux[2] * scale / ps_accept
        # Add bin content and add error in quadrature
        # Set these values for htagged_flux_err AND for htagmE_flux_err
        # fill htagm_flux_err with x = tagm_flux[0] with weight content
        #
        # Repeat for tagh histograms
        #
        # Get density from CCDB:
        # density_table = TARGET/density
        # density = density_table[0][0]
        # density_err = density_table[0][1]
        # target_length = 29.5 (cm)
        # target_scattering_centers = density * target_length * n_avagadro(6.02214e23 atoms/mol) * 1e-24 (cm^2 to barn) * 1e-3 (g to mg)
        # target_cattering_centers_err = target_scattering_center * density_err / density
        # for i in range(1, htagged_flux_err.nbins + 1):
        #   if htagged_flux_err.counts[i] <= 0.0: continue
        #   lumi = htagged_flux_err.counts[i] * target_scattering_centers / 1e12
        #   flux_err = htagged_flux_err.errors[i] / htagged_flux_err.counts[i]
        #   target_err = target_cattering_centers_err / target_scattering_centers
        #   lumi_err = lumi * sqrt(flux_err**2 + target_err**2)
        #   htagged_lumi_err.counts[i] = lumi
        #   htagged_lumi_err.errors[i] = lumi_err


def cli():
    path_map: dict[Path, Path] = {
        Path('/home/gluex2/gluexdb/ccdb_2024_05_08.sqlite'): MISC_PATH / 'ccdb.sqlite',
        Path('/home/gluex2/gluexdb/rcdb_2024_05_08.sqlite'): MISC_PATH / 'rcdb.sqlite',
        Path('/raid3/nhoffman/analysis/pol_hists/S17.root'): MISC_PATH / 's17.root',
        Path('/raid3/nhoffman/analysis/pol_hists/S18.root'): MISC_PATH / 's18.root',
        Path('/raid3/nhoffman/analysis/pol_hists/F18.root'): MISC_PATH / 'f18.root',
        Path('/raid3/nhoffman/analysis/pol_hists/S20.root'): MISC_PATH / 's20.root',
    }
    for src, dst in path_map.items():
        shutil.copy(src, dst)
