#!/bin/bash
#SBATCH --job-name=dene-thesis
#SBATCH --output=slurm.log
#SBATCH --error=slurm.log
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --partition=green

source /home/nhoffman/.venv/bin/activate
cd "$PWD" || exit
python3 -m gluex_ksks --chisqdof="$1" --waves="$2"
echo "Done!"
