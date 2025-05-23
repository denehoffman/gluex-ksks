import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import click

from gluex_ksks.constants import RAW_DATASET_PATH, mkdirs


def generate_slurm_job(
    output_name: str,
    *,
    raw_output_dir: Path,
    input_dir: Path,
    job_name: str,
    queue_name: str,
    env_path: Path,
    version_path: Path,
    scratch_dir: Path,
    dselector_c_path: Path,
    dselector_h_path: Path,
    tree_name: str,
) -> str:
    return f"""#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=100
#SBATCH --partition={queue_name}
#SBATCH --output={raw_output_dir}/logs/log_dispatch_%A.out
#SBATCH --error={raw_output_dir}/logs/log_dispatch_%A.err

echo "Starting Job"
pwd

source {env_path} {version_path}
echo "Sourced env file {env_path}"

for inputpath in {input_dir}/*.root
do
    echo "Submitting job for $inputpath..."
    inputname="$(basename $inputpath)"
    inputstem="$(basename $inputname .root)"
    cat << 'EOS' > {scratch_dir}/${{inputstem}}_${{SLURM_JOB_ID}}.sh
#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --partition={queue_name}
#SBATCH --output={raw_output_dir}/logs/log_%A.out
#SBATCH --error={raw_output_dir}/logs/log_%A.err

WORKINGDIR="/scratch/slurm_$SLURM_JOB_ID"
scp="/usr/bin/scp"

echo "Sourcing env"
inputpath=$1
unset $1
inputname=$2
unset $2
echo $1
source {env_path} {version_path}
cd "$WORKINGDIR"
pwd
echo "Contents of working directory:"
ls -lh

echo "Input file: $inputname"
inputname_hist=$(echo "$inputname" | sed 's/tree/hist/')
inputname_flattree=$(echo "$inputname" | sed 's/tree/flattree/')

echo "Copying files..."
scp "{dselector_c_path}" ./
scp "{dselector_h_path}" ./
scp -l 10000 "$inputpath" ./

echo "Contents of working directory:"
ls -lh

echo "Creating a ROOT macro called run.C"
cat << EOF > run.C
void run() {{
    gROOT->ProcessLine(".x $ROOT_ANALYSIS_HOME/scripts/Load_DSelector.C");
    cout << "Checking path name '$inputname': " << !gSystem->AccessPathName("$WORKINGDIR/$inputname") << endl;
    if(!gSystem->AccessPathName("$WORKINGDIR/$inputname")) {{
        gROOT->ProcessLine("TChain *chain = new TChain(\\"{tree_name}\\");");
        gROOT->ProcessLine("chain->Add(\\"$WORKINGDIR/$inputname\\");");
        gROOT->ProcessLine("chain->Process(\\"{dselector_c_path.name}+\\");");
    }}
}}
EOF

    echo "run.C:"
    cat run.C
    echo "Calling run.C"
    echo $ROOTSYS
    root -l -b -q run.C
    echo "Script complete!"

    echo "Contents of working directory:"
    ls -lh

    echo "Copying files out of scratch:"

    scp "$WORKINGDIR/flat.root" "{raw_output_dir}/{output_name}/$inputname_flattree"
    echo "DONE"
EOS
    ssh -o StrictHostKeyChecking=no ${{USER}}@ernest.phys.cmu.edu "sbatch {scratch_dir}/${{inputstem}}_${{SLURM_JOB_ID}}.sh $inputpath $inputname"
    rm {scratch_dir}/${{inputstem}}_${{SLURM_JOB_ID}}.sh
    sleep 2
    done
echo "DONE"
"""


def run_slurm_script(content: str):
    with tempfile.NamedTemporaryFile('w+', suffix='.sh', delete=True) as temp:
        temp.write(content)
        temp.flush()
        subprocess.run(['sbatch', temp.name], check=False)  # noqa: S603, S607


queues = {'blue': 64, 'green': 32}
data_analysis_sets = [
    {
        'output_name': output_name,
        'data_type': 'data',
        'input_dir': input_dir,
        'job_name': f'data_{output_name}',
        'dselector_c_path': Path.cwd()
        / 'analysis'
        / 'dselectors'
        / ('DSelector_phase_1.C' if output_name != 's20' else 'DSelector_phase_2.C'),
        'tree_name': 'ksks__B4_Tree',
    }
    for output_name, input_dir in [
        (
            's17',
            Path(
                '/raid3/nhoffman/RunPeriod-2017-01/analysis/ver52/tree_ksks__B4/merged'
            ),
        ),
        (
            's18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-01/analysis/ver19/tree_ksks__B4/merged'
            ),
        ),
        (
            'f18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-08/analysis/ver19/tree_ksks__B4/merged'
            ),
        ),
        (
            's20',
            Path(
                '/raid3/nhoffman/RunPeriod-2019-11/analysis/ver04/tree_ksks__B4/merged'
            ),
        ),
    ]
]
sigmc_analysis_sets = [
    {
        'output_name': output_name,
        'data_type': 'sigmc',
        'input_dir': input_dir,
        'job_name': f'sigmc_{output_name}',
        'dselector_c_path': Path.cwd()
        / 'analysis'
        / 'dselectors'
        / ('DSelector_phase_1.C' if output_name != 's20' else 'DSelector_phase_2.C'),
        'tree_name': 'ksks__B4_Tree',
    }
    for output_name, input_dir in [
        (
            's17',
            Path(
                '/raid3/nhoffman/RunPeriod-2017-01/flat_MC/ver52/ksks/tree_ksks__B4_gen_amp_large'
            ),
        ),
        (
            's18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-01/flat_MC/ver19/ksks/tree_ksks__B4_gen_amp_large'
            ),
        ),
        (
            'f18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-08/flat_MC/ver19/ksks/tree_ksks__B4_gen_amp_large'
            ),
        ),
        (
            's20',
            Path(
                '/raid3/nhoffman/RunPeriod-2019-11/flat_MC/ver04/ksks/tree_ksks__B4_gen_amp_large'
            ),
        ),
    ]
]
bkgmc_analysis_sets = [
    {
        'output_name': output_name,
        'data_type': 'bkgmc',
        'input_dir': input_dir,
        'job_name': f'bkgmc_{output_name}',
        'dselector_c_path': Path.cwd()
        / 'analysis'
        / 'dselectors'
        / ('DSelector_phase_1.C' if output_name != 's20' else 'DSelector_phase_2.C'),
        'tree_name': 'ksks__B4_Tree',
    }
    for output_name, input_dir in [
        (
            's17',
            Path(
                '/raid3/nhoffman/RunPeriod-2017-01/flat_MC/ver52/4pi/tree_ksks__B4_gen_amp/'
            ),
        ),
        (
            's18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-01/flat_MC/ver19/4pi/tree_ksks__B4_gen_amp/'
            ),
        ),
        (
            'f18',
            Path(
                '/raid3/nhoffman/RunPeriod-2018-08/flat_MC/ver19/4pi/tree_ksks__B4_gen_amp/'
            ),
        ),
        (
            's20',
            Path(
                '/raid3/nhoffman/RunPeriod-2019-11/flat_MC/ver04/4pi/tree_ksks__B4_gen_amp/'
            ),
        ),
    ]
]
bggen_analysis_sets = [
    {
        'output_name': output_name,
        'data_type': 'bggen',
        'input_dir': input_dir,
        'job_name': f'bggen_{output_name}',
        'dselector_c_path': Path.cwd()
        / 'analysis'
        / 'dselectors'
        / ('DSelector_phase_1.C' if output_name != 's20' else 'DSelector_phase_2.C'),
        'tree_name': 'ksks__B4_Tree',
    }
    for output_name, input_dir in [
        (
            's18',
            Path(
                '/raid2/nhoffman/RunPeriod-2018-01/analysis/bggen/ver11/batch01/tree_ksks__B4/merged'
            ),
        ),
    ]
]


def run_analysis(
    output_name: str,
    *,
    data_type: str,
    input_dir: Path,
    job_name: str,
    queue_name: str,
    env_path: Path,
    version_path: Path,
    dselector_c_path: Path,
    tree_name: str,
):
    scratch_dir = Path.cwd() / 'tmp'
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mkdirs()
    run_slurm_script(
        generate_slurm_job(
            output_name,
            raw_output_dir=RAW_DATASET_PATH[data_type],
            input_dir=input_dir,
            job_name=job_name,
            queue_name=queue_name,
            env_path=env_path,
            version_path=version_path,
            scratch_dir=scratch_dir,
            dselector_c_path=dselector_c_path,
            dselector_h_path=dselector_c_path.with_suffix('.h'),
            tree_name=tree_name,
        )
    )


def run_on_slurm(
    data_type: Literal['data', 'sigmc', 'bkgmc', 'bggen'],
    queue_name: Literal['blue', 'green'],
):
    if data_type == 'data':
        analysis_sets = data_analysis_sets
    elif data_type == 'sigmc':
        analysis_sets = sigmc_analysis_sets
    elif data_type == 'bkgmc':
        analysis_sets = bkgmc_analysis_sets
    else:
        analysis_sets = bggen_analysis_sets

    for analysis_set in analysis_sets:
        run_analysis(
            **analysis_set,
            queue_name=queue_name,
            env_path=Path.cwd() / 'analysis' / 'env.sh',
            version_path=Path.cwd() / 'analysis' / 'version.xml',
        )


@click.command()
@click.option(
    '--queue',
    type=click.Choice(['blue', 'green'], case_sensitive=False),
    default='blue',
    show_default=True,
    help='slurm queue to use',
)
def cli(queue):
    run_on_slurm('data', queue)
    run_on_slurm('sigmc', queue)
    run_on_slurm('bkgmc', queue)
    run_on_slurm('bggen', queue)
