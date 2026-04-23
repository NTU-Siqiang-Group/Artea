# a quick run script

numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --dataset gist-1m --l0-radius 1.13 --beta 2.0 --candidate-queue-config 30,390,30 \
    --ul-select-nbrs-qs 64 \
    --search-nn-qs 20 \
    --ul-max-nbr-size 32 \
    --bl-max-nbr-size 64 \
    --num-build-loops 5 \
    --num-triu-iters 12 \
    --query-topk 20 \
    --test-runs 20 \
    --warmup-runs 10 \
    --routing-topk 1 \
    --routing-queue-size 50 \
    --prefill-ratio 0.34 \
    --scale-coeffs 1.00 \
    --shifted-coeffs 0.00 \
    --num-routing-loops 0 \
    # --insert-on-l0 \
    # --bl-select-nbrs-qs 32

# ----------------------------------------------------
# --dataset sift-1m --l0-radius 22031.00 --beta 2.0 --candidate-queue-config 20,200,10 \
# --dataset gist-1m --l0-radius 1.13 --beta 2.0 --candidate-queue-config 30,390,30 \