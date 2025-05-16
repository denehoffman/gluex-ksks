import pickle
import sqlite3
from pathlib import Path
from typing import override

import uproot
from modak import Task
from uproot.behaviors.TBranch import HasBranches
from uproot.reading import ReadOnlyDirectory

from gluex_ksks.constants import MISC_PATH
from gluex_ksks.utils import CCDBData, Histogram, RCDBData, ScalingFactors, get_pol_angle, get_run_period


class CCDB(Task):
    def __init__(self):
        MISC_PATH.mkdir(exist_ok=True, parents=True)
        super().__init__('ccdb', outputs=[MISC_PATH / 'ccdb.pkl'])

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
                (MISC_PATH / 'ccdb.pkl').open('wb'),
            )


class RCDB(Task):
    def __init__(self):
        MISC_PATH.mkdir(exist_ok=True, parents=True)
        super().__init__('ccdb', outputs=[MISC_PATH / 'ccdb.pkl'])

    @override
    def run(self):
        angles = {}
        with sqlite3.connect(str(MISC_PATH / 'rcdb.sqlite')) as rcdb:
            cursor = rcdb.cursor()
            query = """
            SELECT r.number, c.float_value
            FROM conditions c
            JOIN condition_types ct ON c.condition_type_id = ct.id
            JOIN runs r ON c.run_number = r.number
            WHERE ct.name = 'polarization_angle'
            ORDER BY r.number
            """
            cursor.execute(query)
            pol_angle_results = cursor.fetchall()
            for run_number, angle_deg in pol_angle_results:
                run_period = get_run_period(run_number)
                pol_angle = get_pol_angle(run_period, angle_deg)
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
                    self.log_error(f'Error reading histograms from {hist_path}')
                    msg = f'Error reading histograms from {hist_path}'
                    raise OSError(msg)
                mags, edges = hist.to_numpy()
                hists[pol] = Histogram(mags, edges)
            magnitudes[rp] = hists
        pickle.dump(
            RCDBData(angles, magnitudes),
            (MISC_PATH / 'rcdb.pkl').open('wb'),
        )


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
        dst.hardlink_to(src)
