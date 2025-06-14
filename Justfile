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

run chisqdof='3.00' queue='blue':
  sbatch slurm_job.sh {{chisqdof}} --partition={{queue}}

run-all queue='blue':
  sbatch slurm_job.sh 2.00 --partition={{queue}}
  sbatch slurm_job.sh 3.00 --partition={{queue}}
  sbatch slurm_job.sh 4.00 --partition={{queue}}
  sbatch slurm_job.sh 5.00 --partition={{queue}}

tail:
  tail -f analysis/logs/all.log
