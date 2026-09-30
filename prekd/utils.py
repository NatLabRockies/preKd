import os
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
    if getattr(parameters, "use_hybrid_loss", False):
        bce_weight = getattr(parameters, "hybrid_bce_weight", 0.5)
        job_str += "_hb" + f"{bce_weight:.2f}".replace(".", "p")
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


def write_slurm_submit(out_dir,
                       mm_data_file,
                       params,
                       job_name,
                       base_mm_data_file=None,
                       node_idx=0,
                       n_runs_per_node=5,
                       account=None,
                       partition=None,
                       time_limit="4:00:00",
                       gres="gpu:1",
                       cpus_per_task=26,
                       mem="80GB",
                       env_setup=None,
                       submit=False):
    """Write a SLURM script that runs ``n_runs_per_node`` of the CV folds on one GPU node.

    Cluster-specific settings are parameters rather than hard-coded values, so this
    helper is portable. Each may also be supplied through an environment variable:

    ======================  ==============================  =========================
    argument                environment variable            example
    ======================  ==============================  =========================
    ``account``             ``PREKD_SLURM_ACCOUNT``         ``my_allocation``
    ``partition``           ``PREKD_SLURM_PARTITION``       ``gpu``
    ``gres``                ``PREKD_SLURM_GRES``            ``gpu:h100:1``
    ``env_setup``           ``PREKD_ENV_SETUP``             ``module load cuda; conda activate prekd``
    ======================  ==============================  =========================

    Parameters
    ----------
    out_dir : Path
        Directory the submit script, logs and trained models are written to.
    mm_data_file : Path
        The training-data dump produced by ``prepare_data.py``.
    params : Parameters
        Model/training parameters used to build the command line.
    job_name : str
        SLURM job name.
    base_mm_data_file : Path, optional
        If given, run transfer learning from this base model instead.
    node_idx : int, optional
        Index of the node this script runs on, by default 0.
    n_runs_per_node : int, optional
        How many folds to run concurrently on the node, by default 5.
    env_setup : str or list of str, optional
        Shell lines placed before the training command, e.g. ``module load`` and
        the activation of your Python environment.
    submit : bool, optional
        Submit the script with ``sbatch`` after writing it, by default False.

    Returns
    -------
    Path
        The submit script that was written.
    """
    account = account if account is not None else os.environ.get("PREKD_SLURM_ACCOUNT")
    partition = partition if partition is not None else os.environ.get("PREKD_SLURM_PARTITION")
    gres = os.environ.get("PREKD_SLURM_GRES", gres)
    if env_setup is None:
        env_setup = os.environ.get("PREKD_ENV_SETUP", "")
    if isinstance(env_setup, (list, tuple)):
        env_setup = "\n".join(env_setup)

    start_idx = node_idx * n_runs_per_node
    end_idx = (node_idx + 1) * n_runs_per_node
    out_dir.mkdir(parents=True, exist_ok=True)
    log_file = out_dir / f"log_n{node_idx}_nruns{n_runs_per_node}.txt"
    python_script = "python train_solute_solvent.py"
    transfer_learning_opt = ""
    hybrid_opt = ""
    if base_mm_data_file is not None:
        # setup the transfer learning option
        transfer_learning_opt = f" --base_model {base_mm_data_file}"
        python_script = "python train_solute_solvent_TL.py"

    if getattr(params, "use_hybrid_loss", False):
        hybrid_opt = " --use_hybrid_loss"

    # Only emit the optional directives that were actually configured
    optional_directives = ""
    if account:
        optional_directives += f"#SBATCH --account={account}\n"
    if partition:
        optional_directives += f"#SBATCH --partition={partition}\n"

    submit_str = f"""#!/bin/bash
#SBATCH --job-name={job_name}
{optional_directives}#SBATCH --time={time_limit}
#SBATCH --gres={gres}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus_per_task}
#SBATCH --mem={mem}
#SBATCH --output={log_file}
#SBATCH --error={log_file}
#SBATCH --open-mode=append

# Environment setup (see the env_setup argument / PREKD_ENV_SETUP)
{env_setup}

echo "Job started at `date`"
for ((i = {start_idx}; i < {end_idx} ; i++)); do
mkdir -p {out_dir}/model_$i
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
    --hybrid_cutoff {getattr(params, 'hybrid_cutoff', 1.5)} \\
    --hybrid_bce_weight {getattr(params, 'hybrid_bce_weight', 0.5)} \\
    --seed {getattr(params, 'seed', 0)} \\
    {hybrid_opt} \\
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


# Backwards-compatible alias for the previous cluster-specific name
write_submit_kestrel = write_slurm_submit
