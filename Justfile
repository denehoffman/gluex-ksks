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

run chisqdof='3.00' queue='blue':
  sbatch slurm_job_{{queue}}.sh {{chisqdof}}

run-others queue='blue':
  sbatch slurm_job_{{queue}}.sh 2.00
  sbatch slurm_job_{{queue}}.sh 4.00
  sbatch slurm_job_{{queue}}.sh 5.00

tail:
  tail -f analysis/logs/all.log
