# Adaptive Mechanism Design (AMD) in Sequential Social Dilemma (SSD)

This is the code for project of cource Foundations of Reinforcement Learning (FoRL) Spring Semester 2023. The major contributions of this project is
- A full implementation of AMD algorithm [[original paper]](https://arxiv.org/abs/1806.04067) for arbitrary environments, using `ray==2.3.1`. 
  - migrating to higher version of `ray` needs additional effort
- Two RL environments, Wolfpack and Gathering. [[original paper]](https://arxiv.org/abs/1702.03037)

# Instruction for local and Euler environment setup
I decide to use Python 3.10.4 and CUDA 11.8 as standard version. This is the default version on Euler.

These versions can be modifyed. Depending on the repo we are migrating.

## Euler setup
On Euler, for each time you need to load the module.
```sh
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy cudnn/8.4.0.27
```
Package `gcc/8.2.0` is necessary. Only this module is loaded, then can you search out result about python and cuda. You can search for the version of python and cuda you want by command 
```shell
module avail ${package name: cuda, python, etc}
```

create a virtual env with venv
```sh
py_venv_dir="${SCRATCH}/.python_venv"
if [ ! -d ${py_venv_dir} ]; then
    mkdir ${py_venv_dir}
fi
python -m venv ${py_venv_dir}/forl-proj --upgrade-deps
# actiavte
source "${SCRATCH}/.python_venv/forl-proj/bin/activate"
# deactivate
deactivate
```
To install python packages, after activation, run
```sh
pip3 install -r requirements.txt --cache-dir ${SCRATCH}/pip_cache
```

## Local setup
On local machine, to install this exact python version I use conda (you can also use venv).
```sh
conda create --name=forl-proj python=3.10
# activate
conda activate forl-proj
# deactivate
conda deactivate
```

# Tips for using LSF or Slurm
There are some links here 
- [Migration from LSF to SLURM](https://scicomp.ethz.ch/wiki/LSF_to_Slurm_quick_reference)
- [How to use GPU](https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs)

Most importantly, [this](https://scicomp.ethz.ch/public/lsla/index2.html) interactive website can generate sbatch scripts.


Here (`run.bash`) is an example of my sbatch scripts, you can submit job by command
```shell
sbatch run.bash
```
`run.bash` is like this, only the beginning commented part begin with `#SBATCH` are effective for configuration.
```bash
#!/usr/bin/bash
#SBATCH --job-name="snt-vae"
#SBATCH --output=%j.out
#SBATCH --time=60:00:00
#SBATCH --ntasks=2
#SBATCH --mem-per-cpu=32G
#SBATCH --tmp=4G
#SBATCH --gpus=rtx_3090:1

# for normal computer:
# work_dir="$(readlink -f $(dirname -- "$0";)/..;)"
# for SBATCH use this:
# work_dir="${SCRATCH}/transformer-vae"
if [ -z ${SCRATCH+x} ]; then work_dir="$(readlink -f $(dirname -- "$0")/..)"; else work_dir="${SCRATCH}/transformer-vae"; fi
echo ${work_dir}
experiement_dir="${work_dir}/experiments"
experiment_name="baseline"
logging_dir="${work_dir}/logs"
if [ ! -d ${logging_dir} ]; then
    mkdir ${logging_dir}
fi
if [ ! -d ${experiement_dir} ]; then
    mkdir ${experiement_dir}
fi
# check if current experiment name is in use
tmp_name=${experiment_name}
x=1
while [ -d "${experiement_dir}/${tmp_name}" ]; do
    tmp_name="${experiment_name}_${x}"
    x=$(( $x + 1 ))
done
experiment_name="${tmp_name}"
${work_dir}/.venv/bin/python main.py fit \
    --config ./config/sentence_vae.yaml \
    --trainer.default_root_dir "${experiement_dir}/${experiment_name}" \
    --trainer.logger+=pytorch_lightning.loggers.WandbLogger \
    --trainer.logger.project="transformer-vae" \
    --trainer.logger.entity="nlp-ethz-semester-project-constraint-optim" \
    --trainer.logger.name="${experiment_name}" \
    --trainer.logger.save_dir="${logging_dir}/wandb" \
    --trainer.logger.dir="${logging_dir}" \
    --trainer.logger+=pytorch_lightning.loggers.TensorBoardLogger \
    --trainer.logger.name="${experiment_name}" \
    --trainer.logger.save_dir="${logging_dir}/tensorboard" \
    --trainer.logger+=pytorch_lightning.loggers.CSVLogger \
    --trainer.logger.name="${experiment_name}" \
    --trainer.logger.save_dir="${logging_dir}/csv" \
    --trainer.callbacks+=core.callbacks.SaveCLIConfigToWandB \
    --trainer.callbacks+=pytorch_lightning.callbacks.LearningRateMonitor \
    --trainer.callbacks+=pytorch_lightning.callbacks.ModelCheckpoint \
    --trainer.callbacks.dirpath="${experiement_dir}/${experiment_name}/checkpoints" \
    --trainer.callbacks.monitor="valid/ELBO" \
    --trainer.callbacks.mode="min" \
    --data.init_args.tokenizer_dir "${experiement_dir}/${experiment_name}"

```

## Ray on SLURM
See [here](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray).
