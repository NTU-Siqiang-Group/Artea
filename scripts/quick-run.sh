# a quick run script

numactl --interleave=all ./build/unit_tests/test_artea_graph \
    --l1-radius 3.00 \
    --dataset gist-1m  \
    --select-nbrs-qs 64 \
    --search-nn-qs 20 \
    --max-nbr-size 32 \
    --refining-max-nbr-size 96 \
    --insert-on-l0 \
    --num-build-loops 1 \
    --num-triu-iters 15 \
    --query-topk 20 \
    --test-runs 20 \
    --warmup-runs 10 \
    --routing-topk 96 \
    --routing-queue-size 128 \
    --prefill-ratio 0.34 \
    --shifted-coeffs 0.0 \
    --num-routing-loops 1 \
    --candidate-queue-config 30,390,30