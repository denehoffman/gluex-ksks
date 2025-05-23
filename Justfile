default:
  just --list

venv:
  [ -d .venv ] || uv venv --python=3.13
  source .venv/bin/activate

setup: venv
  uv pip install --reinstall --no-cache -e .
  ./.venv/bin/gluex-init

get-data queue:
  ./.venv/bin/gluex-run-dselectors --queue {{queue}}

clean-raw-data:
  rm -rf analysis/raw_datasets

run queue: setup
  sbatch slurm_job.sh --partition={{queue}}
