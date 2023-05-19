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

# Load modules or your own conda environment here
module load gcc/8.2.0 python_gpu/3.10.4 cuda/11.8.0 git-lfs/2.3.0 git/2.31.1 eth_proxy cudnn/8.4.0.27

# update environment !!! this is important
source "${SCRATCH}/.python_venv/forl-proj/bin/activate"

# configure ray temp dir
ray_temp_dir="${SCRATCH}/.tmp_ray"
if [ ! -d ${ray_temp_dir} ]; then
    mkdir ${ray_temp_dir}
fi

redis_password=$(uuidgen)
export redis_password

head_node_ip=$(srun --exact --ntasks=1 hostname --ip-address)

# if we detect a space character in the head node IP, we'll
# convert it to an ipv4 address. This step is optional.
if [[ "$head_node_ip" == *" "* ]]; then
    IFS=' ' read -ra ADDR <<<"$head_node_ip"
    if [[ ${#ADDR[0]} -gt 16 ]]; then
        head_node_ip=${ADDR[1]}
    else
        head_node_ip=${ADDR[0]}
    fi
    echo "IPV6 address detected. We split the IPV4 address as $head_node_ip"
fi
# __doc_head_address_end__

# __doc_head_ray_start__
port=6379
ip_head=$head_node_ip:$port
export ip_head
echo "IP Head: $ip_head"

echo "STARTING HEAD"
srun ray start --head --node-ip-address=$head_node_ip --port=$port --redis-password=$redis_password --temp-dir $ray_temp_dir --block --disable-usage-stats &
# __doc_head_ray_end__

# __doc_script_start__
echo "STARTING ALGORITHM with num of works of $worker_num ."
python sbatch_scripts/amd_matrix_game.py
