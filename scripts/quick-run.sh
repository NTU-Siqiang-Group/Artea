# a quick run script

numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --dataset sift-1m --l0-radius 45000.00 --beta 2.0 \
    --ul-select-nbrs-qs 64 \
    --search-nn-qs 20 \
    --max-nbr-size 48 \
    --refining-max-nbr-size 96 \
    --num-build-loops 5 \
    --num-triu-iters 12 \
    --query-topk 10 \
    --test-runs 20 \
    --warmup-runs 10 \
    --routing-topk 1 \
    --routing-queue-size 50 \
    --prefill-ratio 0.34 \
    --scale-coeffs 1.00 \
    --shifted-coeffs 0.00 \
    --num-routing-loops 0 \
    --candidate-queue-config 10,200,10 \
    # --insert-on-l0 \
    # --bl-select-nbrs-qs 32

# ----------------------------------------------------
# --dataset sift-1m --l0-radius 22031.00 --beta 4.0 \
# --dataset gist-1m --l0-radius 1.13 --beta 2.0 \