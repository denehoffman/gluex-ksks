#!/bin/bash
#SBATCH --job-name=dene-thesis
#SBATCH --output=slurm.log
#SBATCH --error=slurm.log
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=120G

source /home/nhoffman/.venv/bin/activate
source /raid3/nhoffman/root/root_install_313/bin/thisroot.sh

cd "$PWD" || exit
uv pip install --reinstall .

which python3
python3 -m gluex_ksks --chisqdof="$1"
echo "Done!"
