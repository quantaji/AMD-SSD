#!/usr/bin/bash

#SBATCH --job-name="forl-proj"
#SBATCH --output=%j.out
#SBATCH --time=15:00

### Give all resources to a single Ray task, ray can manage the resources internally
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=3G
#SBATCH --gpus=gtx_1080_ti:1

### storage
#SBATCH --tmp=4G

# testing, check the environment variable
echo $SLURM_NTASKS

# Load modules or your own conda environment here
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy cudnn/8.4.0.27

# update environment !!! this is important
source "${SCRATCH}/.python_venv/forl-proj/bin/activate"

# configure ray temp dir
ray_temp_dir="${SCRATCH}/.tmp_ray"
if [ ! -d ${ray_temp_dir} ]; then
    mkdir ${ray_temp_dir}
fi

# python sbatch_scripts/amd_wolfpack.py --num_workers ${worker_num}
python sbatch_scripts/amd_wolfpack.py
