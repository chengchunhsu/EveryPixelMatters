python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --master_port=$((RANDOM + 10000)) \
    tools/train_net_da.py \
    --config-file ./configs/da_ga_ca_sim10k_R_101_FPN_4x.yaml \