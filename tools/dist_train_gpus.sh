#!/usr/bin/env bash

CONFIG=$1
# GPUS=$2
# PORT=${PORT:-28509}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
echo $PYTHONPATH
# python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
#     $(dirname "$0")/train.py $CONFIG --launcher pytorch ${@:3} --deterministic



# NODE_RANK=$1
# # GPUS=$2
# # NNODES=${NNODES:-1}
# # NODE_RANK=${NODE_RANK:-0}
# # PORT=${PORT:-29500}
# # MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# # PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# PYTHONPATH="/nfs/projects/OccNet-1/tools/.."
# echo $PYTHONPATH
# python -m torch.distributed.launch \
#     --nproc_per_node=1 \
#     --nnodes=2 \
#     --node_rank=$NODE_RANK \
#     --master_addr="10.130.1.191" \
#     --master_port=54636 \
#     $(dirname "$0")/train.py \
#     ./projects/configs/bevformer/bev_tiny_occ_flow.py \
#     --seed 0 \
#     --launcher pytorch


# # ./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ_flow.py 1 2 0 39999 10.130.1.191
# # ./tools/dist_train.sh ./projects/configs/bevformer/bev_tiny_occ_flow.py 1 2 1 39999 10.130.1.191