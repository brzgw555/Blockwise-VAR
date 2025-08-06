#!/bin/bash

# ==============================================
# 
# ==============================================
export CUDA_VISIBLE_DEVICES=0,1,2,3  # GPU pointed
NUM_NODES=1                          # node number
GPUS_PER_NODE=4                      # GPU per node
MASTER_ADDR="localhost"              # master node address
MASTER_PORT=$((RANDOM + 10000))      # arbitrary port for master node
NUM_WORKERS=12                       # data loading processes

#  NCCL params）
if [[ "$ARNOLD_DEVICE_TYPE" == *A100* ]]; then
    IB_HCA=mlx5
    export NCCL_IB_HCA=$IB_HCA
else
    IB_HCA=$ARNOLD_RDMA_DEVICE:1
fi

if [[ "$RUNTIME_IDC_NAME" == *uswest2* ]]; then
    IDC_NAME=bond0
    export NCCL_SOCKET_IFNAME=$IDC_NAME
else
    IDC_NAME=eth0
fi

# NCCL
export NCCL_DEBUG=WARN
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3

if [[ "$*" == *"--debug"* ]]; then
    NUM_NODES=1
    GPUS_PER_NODE=1
    NUM_WORKERS=0
fi


# ==============================================
# params
# ==============================================
FIXED_ARGS="--patch_size 16 \
    --ch 160 \
    --vocab_size 4096 \
    --z_channels 32 \
    --share_quant_resi 4 \
    --optim_type AdamW --lr 4.5e-6 --disable_sch --dis_lr_multiplier 1 \
    --resolution 256 256 --batch_size 16 \
    --dataset_list imagenet --dataaug resizecrop \
    --disc_layers 3 --discriminator_iter_start 50000 \
    --l1_weight 1 --perceptual_weight 1 --image_disc_weight 1 --image_gan_weight 0.3  --gan_feat_weight 0 --lfq_weight 4 \
    --entropy_loss_weight 0.1 --diversity_gamma 1 \
    --default_root_dir block_wise_frequency \
    --new_quant --lr_drop 450000 \
    --max_steps 500000 \
    --log_every 20 --ckpt_every 10000 --visu_every 10000 \
    --remove_residual_detach --use_lecam_reg_zero --base_ch_disc 128 --dis_lr_multiplier 2.0 \
    --schedule_mode dense --use_stochastic_depth --drop_rate 0.5 --keep_last_quant --tokenizer flux"

# ==============================================
# print training configuration
# ==============================================
echo "=== training configuration ==="
echo "node num: $NUM_NODES"
echo "GPU per node: $GPUS_PER_NODE"
echo "master node: $MASTER_ADDR:$MASTER_PORT"
echo "num_workers: $NUM_WORKERS"
echo "- NCCL: $IDC_NAME"
echo "- IB device: $IB_HCA"



torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NUM_NODES \
    --master_addr=$MASTER_ADDR \
    --node_rank=0 \
    --master_port=$MASTER_PORT \
    train_vae.py --num_workers $NUM_WORKERS \
    $FIXED_ARGS \
    --max_steps 500000 \
    --log_every 10 \
    --ckpt_every 20000 \
    --visu_every 10000 \
    --ch 160 \
    --batch_size 12 \
    --discriminator_iter_start 200000