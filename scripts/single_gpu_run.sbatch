#!/bin/bash
#SBATCH --account nvr_asicvlsi_chipnemo
#SBATCH --partition grizzly,polar,polar3,polar4
#SBATCH --nodes 1
#SBATCH --time 04:00:00
#SBATCH --job-name "nvr_asicvlsi_chipnemo:mllm-4-physical-design"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1


export MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"
export MASTER_PORT="$((10000 + RANDOM % 50000))"
export WORLD_SIZE="${SLURM_NTASKS}"

echo $MASTER_ADDR
echo $MASTER_PORT
echo $WORLD_SIZE

CONTAINER=/lustre/fsw/portfolios/nvr/users/yundat/containers/mllm.sqsh
CONTAINER=/lustre/fsw/portfolios/nvr/users/yundat/containers/nemo-25-02-rc6-nofuturewarning.sqsh

read -r -d '' run <<EOF
nvidia-smi \
&& pip -V \
&& cd /lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design \
&& python -m pip install -r requirements2.txt \
&& cd ./armo \
&& python -m pip install -r requirements2.txt \
&& cd .. \
&& python ./armo/stage-3_train.py
EOF

srun --container-image=$CONTAINER \
    --export=ALL,MASTER_ADDR,MASTER_PORT,WORLD_SIZE,OMP_NUM_THREADS \
    --container-mounts /lustre:/lustre \
    bash -c "${run}"

