import subprocess
import tempfile
from pathlib import Path
from typing import Literal

import click
import polars as pl
import uproot
from uproot.behaviors.TBranch import HasBranches


def root_to_parquet(path: Path, tree: str = "kin"):
    tt = uproot.open(
        f"{path}:{tree}",
    )
    assert isinstance(tt, HasBranches)  # noqa: S101
    root_data = tt.arrays(library="np")
    for key in root_data:
        if key.startswith("P4_"):
            root_data[key.replace("P4_", "p4_")] = root_data.pop(key)
        if key.startswith("Weight"):
            root_data[key.replace("Weight", "weight")] = root_data.pop(key)
    dataframe = pl.from_dict(root_data)
    dataframe.write_parquet(path.with_suffix(".parquet"))


def generate_slurm_job(
    output_name: str,
    *,
    raw_output_dir: Path,
    input_dir: Path,
    job_name: str,
    queue_name: str,
    env_path: Path,
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

source {env_path}
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
source {env_path}
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


def generate_parallel_hadd_job(
    output_name: str,
    *,
    output_dir: Path,
    raw_output_dir: Path,
    job_name: str,
    queue_name: str,
    env_path: Path,
    max_n_tasks: int,
):
    return f"""#!/bin/sh
#SBATCH --job-name={job_name}
#SBATCH --dependency=singleton
#SBATCH --nodes=1
#SBATCH --ntasks={max_n_tasks}
#SBATCH --partition={queue_name}
#SBATCH --output={raw_output_dir}/logs/merge_log_%A.out
#SBATCH --error={raw_output_dir}/logs/merge_log_%A.err
#SBATCH --time=1:00:00

source {env_path}

echo "Merging flat trees..."
echo "hadd -O -f -j {max_n_tasks} {output_dir}/{output_name}.root {raw_output_dir}/{output_name}/*.root"
hadd -O -f -j {max_n_tasks} {output_dir}/{output_name}.root {raw_output_dir}/{output_name}/*.root

echo "DONE"
"""


def mkdirs(output_name: str, output_dir: Path, raw_output_dir: Path):
    (output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (output_dir / output_name).mkdir(parents=True, exist_ok=True)
    (raw_output_dir / "logs").mkdir(parents=True, exist_ok=True)
    (raw_output_dir / output_name).mkdir(parents=True, exist_ok=True)


def run_slurm_script(content: str):
    with tempfile.NamedTemporaryFile("w+", suffix=".sh", delete=True) as temp:
        temp.write(content)
        temp.flush()
        subprocess.run(["sbatch", temp.name], check=False)  # noqa: S603, S607


queues = {"blue": 64, "green": 32}
data_analysis_sets = [
    {
        "output_name": output_name,
        "output_dir": Path.cwd() / "analysis" / "datasets" / "data",
        "raw_output_dir": Path.cwd() / "analysis" / "raw_datasets" / "data",
        "input_dir": input_dir,
        "job_name": f"data_{output_name}",
        "dselector_c_path": Path.cwd()
        / "analysis"
        / "dselectors"
        / ("DSelector_phase_1.C" if output_name != "s20" else "DSelector_phase_2.C"),
        "tree_name": "ksks__B4_Tree",
    }
    for output_name, input_dir in [
        ("s17", Path("/raid3/nhoffman/RunPeriod-2017-01/analysis/ver52/tree_ksks__B4/merged")),
        ("s18", Path("/raid3/nhoffman/RunPeriod-2018-01/analysis/ver19/tree_ksks__B4/merged")),
        ("f18", Path("/raid3/nhoffman/RunPeriod-2018-08/analysis/ver19/tree_ksks__B4/merged")),
        ("s20", Path("/raid3/nhoffman/RunPeriod-2019-11/analysis/ver04/tree_ksks__B4/merged")),
    ]
]
sigmc_analysis_sets = [
    {
        "output_name": output_name,
        "output_dir": Path.cwd() / "analysis" / "datasets" / "sigmc",
        "raw_output_dir": Path.cwd() / "analysis" / "raw_datasets" / "sigmc",
        "input_dir": input_dir,
        "job_name": f"sigmc_{output_name}",
        "dselector_c_path": Path.cwd()
        / "analysis"
        / "dselectors"
        / ("DSelector_phase_1.C" if output_name != "s20" else "DSelector_phase_2.C"),
        "tree_name": "ksks__B4_Tree",
    }
    for output_name, input_dir in [
        ("s17", Path("/raid3/nhoffman/RunPeriod-2017-01/flat_MC/ver52/ksks/tree_ksks__B4_gen_amp_large")),
        ("s18", Path("/raid3/nhoffman/RunPeriod-2018-01/flat_MC/ver19/ksks/tree_ksks__B4_gen_amp_large")),
        ("f18", Path("/raid3/nhoffman/RunPeriod-2018-08/flat_MC/ver19/ksks/tree_ksks__B4_gen_amp_large")),
        ("s20", Path("/raid3/nhoffman/RunPeriod-2019-11/flat_MC/ver04/ksks/tree_ksks__B4_gen_amp_large")),
    ]
]
bkgmc_analysis_sets = [
    {
        "output_name": output_name,
        "output_dir": Path.cwd() / "analysis" / "datasets" / "bkgmc",
        "raw_output_dir": Path.cwd() / "analysis" / "raw_datasets" / "bkgmc",
        "input_dir": input_dir,
        "job_name": f"bkgmc_{output_name}",
        "dselector_c_path": Path.cwd()
        / "analysis"
        / "dselectors"
        / ("DSelector_phase_1.C" if output_name != "s20" else "DSelector_phase_2.C"),
        "tree_name": "ksks__B4_Tree",
    }
    for output_name, input_dir in [
        ("s17", Path("/raid3/nhoffman/RunPeriod-2017-01/flat_MC/ver52/4pi/tree_ksks__B4_gen_amp/")),
        ("s18", Path("/raid3/nhoffman/RunPeriod-2018-01/flat_MC/ver19/4pi/tree_ksks__B4_gen_amp/")),
        ("f18", Path("/raid3/nhoffman/RunPeriod-2018-08/flat_MC/ver19/4pi/tree_ksks__B4_gen_amp/")),
        ("s20", Path("/raid3/nhoffman/RunPeriod-2019-11/flat_MC/ver04/4pi/tree_ksks__B4_gen_amp/")),
    ]
]
bggen_analysis_sets = [
    {
        "output_name": output_name,
        "output_dir": Path.cwd() / "analysis" / "datasets" / "bggen",
        "raw_output_dir": Path.cwd() / "analysis" / "raw_datasets" / "bggen",
        "input_dir": input_dir,
        "job_name": f"bggen_{output_name}",
        "dselector_c_path": Path.cwd()
        / "analysis"
        / "dselectors"
        / ("DSelector_phase_1.C" if output_name != "s20" else "DSelector_phase_2.C"),
        "tree_name": "ksks__B4_Tree",
    }
    for output_name, input_dir in [
        ("s18", Path("/raid2/nhoffman/RunPeriod-2018-01/analysis/bggen/ver11/batch01/tree_ksks__B4/merged")),
    ]
]


def run_analysis(
    output_name: str,
    *,
    output_dir: Path,
    raw_output_dir: Path,
    input_dir: Path,
    job_name: str,
    queue_name: str,
    env_path: Path,
    dselector_c_path: Path,
    tree_name: str,
    max_n_tasks: int,
):
    scratch_dir = Path.cwd() / "tmp"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    mkdirs(output_name, output_dir, raw_output_dir)
    run_slurm_script(
        generate_slurm_job(
            output_name,
            raw_output_dir=raw_output_dir,
            input_dir=input_dir,
            job_name=job_name,
            queue_name=queue_name,
            env_path=env_path,
            scratch_dir=scratch_dir,
            dselector_c_path=dselector_c_path,
            dselector_h_path=dselector_c_path.with_suffix(".h"),
            tree_name=tree_name,
        )
    )
    run_slurm_script(
        generate_parallel_hadd_job(
            output_name,
            output_dir=output_dir,
            raw_output_dir=raw_output_dir,
            job_name=job_name,
            queue_name=queue_name,
            env_path=env_path,
            max_n_tasks=max_n_tasks,
        )
    )
    output_file_path = output_dir / f"{output_name}.root"
    root_to_parquet(output_file_path)


def run_on_slurm(data_type: Literal["data", "sigmc", "bkgmc", "bggen"], queue_name: Literal["blue", "green"]):
    if data_type == "data":
        analysis_sets = data_analysis_sets
    elif data_type == "sigmc":
        analysis_sets = sigmc_analysis_sets
    elif data_type == "bkgmc":
        analysis_sets = bkgmc_analysis_sets
    else:
        analysis_sets = bggen_analysis_sets

    for analysis_set in analysis_sets:
        run_analysis(
            **analysis_set,
            queue_name=queue_name,
            max_n_tasks=queues[queue_name],
            env_path=Path.cwd() / "analysis" / "env.sh",
        )


@click.command()
@click.option(
    "--queue",
    type=click.Choice(["blue", "green"], case_sensitive=False),
    defualt="blue",
    show_default=True,
    help="slurm queue to use",
)
def cli(queue):
    run_on_slurm("data", queue)
    run_on_slurm("sigmc", queue)
    run_on_slurm("bkgmc", queue)
    run_on_slurm("bggen", queue)
