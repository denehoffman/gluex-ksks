default:
  just --list

venv:
  [ -d .venv ] || uv venv --python=3.13
  source .venv/bin/activate

setup: venv
  uv pip install --reinstall --no-cache -e .
  ./.venv/bin/gluex-init

run-dselector queue='blue': setup
  ./.venv/bin/gluex-run-dselectors --queue {{queue}}

[confirm]
log-reset:
  rm -rf analysis/logs

[confirm]
raw-data-reset:
  rm -rf analysis/raw_datasets

[confirm]
data-reset:
  rm -rf analysis/datasets

[confirm]
all-data-reset:
  rm -rf analysis/raw_datasets
  rm -rf analysis/datasets

[confirm]
analysis-reset:
  rm -rf analysis/fits
  rm -rf analysis/plots
  rm -rf analysis/reports

[confirm]
hard-reset: analysis-reset all-data-reset log-reset
  rm -rf analysis/misc

run queue='blue': setup
  sbatch slurm_job.sh --partition={{queue}}

tail:
  tail -f slurm.log
