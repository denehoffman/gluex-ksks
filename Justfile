default:
  just --list

setup:
  uv pip install --reinstall --no-cache -e .
  gluex-init

run-dselector queue='blue': setup
  gluex-run-dselectors --queue {{queue}}

[confirm]
log-reset:
  rm -rf analysis/logs

run chisqdof='3.00' queue='blue' waves='all':
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} {{waves}}

run-others queue='blue' waves='all':
  sbatch slurm_job_{{queue}}.sh 2.00 {{waves}}
  sbatch slurm_job_{{queue}}.sh 4.00 {{waves}}
  sbatch slurm_job_{{queue}}.sh 5.00 {{waves}}

# WARNING: use caution here, these jobs may all write to the same file in some instances, which can cause corruptions
run-separate chisqdof='3.00' queue='blue':
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} spd2p
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} spnd2p
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} spd0p
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} spd1p
  sbatch slurm_job_{{queue}}.sh {{chisqdof}} spnd2pn

tail:
  tail -f analysis/logs/all.log
