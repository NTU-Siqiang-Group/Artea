# Workloads Directory

This directory contains bench workload files (`*.jsonc`, copied from
artea-benchmark's `workloads/`): the schema consumed by bench-artea and by
`unit_tests/test_artea_graph` (`-w/--workload`).

## File Format

Top level: `dataset-config` / `dataset` / `metric` / `warmup_runs` /
`test_runs`. Build params come from the `indexes-config.artea` base entry,
the search grid from its `throughput_search` sweeps (each entry is a
`{topk, "start,end,step"}` pair).

The `dataset-config` field is `./configs/datasets.json`, so run from this
repo's root:

```bash
./build/unit_tests/test_artea_graph -w workloads/sift1m-bench.jsonc
```
