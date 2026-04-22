# a quick run script

numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --l1-radius 2.26 \
    --dataset gist-1m  \
    --ul-select-nbrs-qs 64 \
    --search-nn-qs 20 \
    --max-nbr-size 48 \
    --refining-max-nbr-size 96 \
    --num-build-loops 5 \
    --num-triu-iters 12 \
    --query-topk 20 \
    --test-runs 20 \
    --warmup-runs 10 \
    --routing-topk 64 \
    --routing-queue-size 96 \
    --prefill-ratio 0.34 \
    --scale-coeffs 1.00 \
    --shifted-coeffs 0.0 \
    --num-routing-loops 0 \
    --candidate-queue-config 30,390,30 \
    # --insert-on-l0 \
    # --bl-select-nbrs-qs 32
