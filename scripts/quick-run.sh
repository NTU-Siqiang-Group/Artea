# a quick run script

numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --dataset sift-1m --l0-radius 9285.00 --beta 2.0 --candidate-queue-config 10,100,10 \
    --ul-select-nbrs-qs 64 \
    --search-nn-qs 20 \
    --ul-max-nbr-size 32 \
    --bl-max-nbr-size 64 \
    --num-build-loops 5 \
    --num-triu-iters 12 \
    --query-topk 1 \
    --test-runs 20 \
    --warmup-runs 10 \
    --prefill-ratio 0.34 \
    --scale-coeffs 1.20 \
    --shifted-coeffs 0.00 \
    --num-routing-loops 0 \
    --perform-arc \
    --aspect-ratio-constraint 9.00
    # --insert-on-l0 \
    # --bl-select-nbrs-qs 32

# ----------------------------------------------------
# --dataset sift-1m --l0-radius 4574.00 --beta 3.0 --candidate-queue-config 20,200,10 \
# --dataset gist-1m --l0-radius 1.13 --beta 2.0 --candidate-queue-config 30,390,30 \