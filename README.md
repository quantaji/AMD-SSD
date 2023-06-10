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
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy
```
Package `gcc/8.2.0` is necessary. Only this module is loaded, then can you search out result about python and cuda. You can search for the version of python and cuda you want by command 
```shell
module avail ${package name: cuda, python, etc}
```

create a virtual env with venv
```sh
py_venv_dir="${SCRATCH}/.python_venv"
python -m venv ${py_venv_dir}/forl-proj --upgrade-deps
# To install python packages, run
${SCRATCH}/.python_venv/forl-proj/bin/pip install -r requirements.txt --cache-dir ${SCRATCH}/pip_cache
# actiavte
source "${SCRATCH}/.python_venv/forl-proj/bin/activate"
# deactivate
deactivate
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



## Ray on SLURM
See the following link:
- https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#slurm-network-ray
- https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm-basic.html
- https://github.com/NERSC/slurm-ray-cluster/blob/master/submit-ray-cluster.sbatch
- https://github.com/pengzhenghao/use-ray-with-slurm
- https://github.com/klieret/ray-tune-slurm-demo
