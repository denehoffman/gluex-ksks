from pathlib import Path

import polars as pl
import uproot
from uproot.behaviors.TBranch import HasBranches


def root_to_parquet(path: Path, tree: str = 'kin'):
    tt = uproot.open(
        f'{path}:{tree}',
    )
    assert isinstance(tt, HasBranches)  # noqa: S101
    root_data = tt.arrays(library='np')
    keys = list(root_data.keys())
    for key in keys:
        if key.startswith('P4_'):
            root_data[key.replace('P4_', 'p4_')] = root_data.pop(key)
        if key.startswith('Weight'):
            root_data[key.replace('Weight', 'weight')] = root_data.pop(key)
    dataframe = pl.from_dict(root_data)
    dataframe.write_parquet(path.with_suffix('.parquet'))
