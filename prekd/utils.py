import subprocess
import pickle as pk
import gzip


def get_job_params_str(parameters):
    job_str = ("_mp" + str(parameters.num_messages) +
               "_f" + str(parameters.mol_features) +
               "_b" + str(parameters.batch_size) +
               "_lr" + f"{parameters.learning_rate:1.0e}" +
               "_dr" + f"{parameters.dropout * 100:1.0f}" +
               "_dc" + f"{parameters.decay:1.0e}" +
               "_e" + str(parameters.epochs)
              )
    return job_str


# load the relevant training and validation data without reading in the TF GNNs
def read_model_data(loc_models):
    mm_data = []
    for i in range(10):
        data_path = loc_models / f"model_{i}/model_{i}_data.pk"
        with open(data_path, "rb") as f:
            model_data = pk.load(f)
            mm_data += [model_data]
    return mm_data


def pickle_read(file):
    """

    Args:
        file (string): location and filename with file extension

    Returns:
        _type_: stored object
    """
    open_func = gzip.open if str(file).endswith('.gz') else open
    with open_func(file, 'rb') as f:
        return pk.load(f)


def write_submit_kestrel(out_dir,
                         mm_data_file,
                         params,
                         job_name,
                         base_mm_data_file=None,
                         node_idx=0,
                         n_runs_per_node=5,
                         user="jlaw",
                         submit=False):
    """ Create a slurm script file that will run five of the 10 CV jobs on each kestrel GPU
    *node_idx*: the index of the node / GPU this job will run on
    """
    start_idx = node_idx * n_runs_per_node
    end_idx = (node_idx + 1) * n_runs_per_node
    # num_cpus_per_task = 4
    # mem_per_cpu_per_task = int(80 / (n_runs * num_cpus_per_task))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"log_n{node_idx}_nruns{n_runs_per_node}.txt"
    python_script = f"python train_solute_solvent.py"
    transfer_learning_opt = ""
    if base_mm_data_file is not None:
        # setup the transfer learning option
        transfer_learning_opt = f" --base_model {base_mm_data_file}"
        python_script = f"python train_solute_solvent_TL.py" 

    submit_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --account=bpms
##SBATCH --partition=debug
#SBATCH --time=4:00:00
##SBATCH --time=2-00
#SBATCH --gres=gpu:h100:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
# Reserve 1/4 of the GPU node's CPUs and memory for a single GPU
#SBATCH --cpus-per-task=26
#SBATCH --mem=80GB
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --open-mode=append
##SBATCH --mail-type=ALL
##SBATCH --mail-user=jlaw@nrel.gov

module load mamba cuda/12.4 apptainer
conda activate /home/jlaw/.conda-envs/prot

echo "Job started at `date`"
for ((i = {start_idx}; i < {end_idx} ; i++)); do
apptainer run --bind $PWD:/workspace --nv \\
    /projects/bpms/jlaw/envs/tensorflow_24_05.sif \\
    {python_script} \\
        --kfolds $i \\
        --save_folder {out_dir} \\
        --mm_dump {mm_data_file} \\
        {transfer_learning_opt} \\
        --n_messages {params.num_messages} \\
        --af {params.atom_features} \\
        --bf {params.bond_features} \\
        --mf {params.mol_features} \\
        --epochs {params.epochs} \\
        --batch_size {params.batch_size} \\
        --dropout {params.dropout} \\
        --learning_rate {params.learning_rate} \\
        --decay {params.decay} \\
        --pred_cols {','.join(params.prediction_columns)} \\
        --solute_col {params.solute_col} \\
        --solvents_col {params.solvents_col} \\
        --solvent_fracs_col {params.solvent_fracs_col} \\
        &
done

wait
echo "Job finished at `date`"
"""

    submit_file = out_dir / f"gpu_submit_n{node_idx}_nruns{n_runs_per_node}.sh"
    print(submit_file)
    with open(submit_file, 'w') as out:
        out.write(submit_str)

    if submit:
        cmd = f"sbatch {submit_file}"
        print(cmd)
        subprocess.check_call(cmd, shell=True)

    return submit_file
