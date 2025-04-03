#!/bin/bash
#SBATCH --account nvr_asicvlsi_chipnemo
#SBATCH --partition grizzly,polar,polar3,polar4
#SBATCH --nodes 4
#SBATCH --time 04:00:00
#SBATCH --job-name "nvr_asicvlsi_chipnemo:mllm-4-physical-design"
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=8
#SBATCH --exclusive
#SBATCH --ntasks-per-node=1

export HYDRA_FULL_ERROR=1
export NCCL_IB_QPS_PER_CONNECTION=4;

export MASTER_ADDR=$(scontrol show hostname "${SLURM_JOB_NODELIST}" | head -n1)
export MASTER_PORT="$((${SLURM_JOB_ID} % 10000 + 10000))"
export MASTER_PORT="$((10000 + RANDOM % 50000))"
export WORLD_SIZE="${SLURM_NTASKS}"
export WORLD_SIZE="8"
export HF_TOKEN="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"
export HF_HOME="/lustre/fsw/portfolios/nvr/users/yundat/cache"

echo $MASTER_ADDR
echo $MASTER_PORT


CONTAINER=/lustre/fsw/portfolios/nvr/users/yundat/containers/nemo-25-02-rc6-nofuturewarning.sqsh


read -r -d '' run <<EOF
nvidia-smi \
&& pip -V \
&& cd /lustre/fsw/portfolios/nvr/users/yundat/mllm-physical-design \
&& python -m pip install -r requirements2.txt \
&& cd ./armo \
&& python -m pip install -r requirements2.txt \
&& cd .. \
&& torchrun --nnodes=4 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT ./armo/stage-3_train_dist.py
EOF

srun --container-image=$CONTAINER \
    --export=ALL,MASTER_ADDR,MASTER_PORT,WORLD_SIZE,OMP_NUM_THREADS \
    --container-mounts /lustre:/lustre \
    bash -c "${run}"



# srun --account nvr_asicvlsi_chipnemo --partition interactive --nodes 1 --gres=gpu:8 --container-image=/lustre/fsw/portfolios/nvr/users/yundat/containers/nemo-25-02-rc6-nofuturewarning.sqsh --container-mounts /lustre:/lustre --pty /bin/bash
