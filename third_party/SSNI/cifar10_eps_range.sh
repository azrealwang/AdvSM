SEED1=$1

for seed in $SEED1; do
    python -u dataset_eps_range.py --config cifar10.yml \
        --use_cuda \
        --seed $seed \
        --batch_size 1000 \
        --dataset cifar10 \
        --single_vector_norm_flag \
        --diffusion_type sde \
        --score_type score_sde \
        --num_process_per_node 1 \
        --port 8888
        # --use_score
done