# Agent instructions for ARTEA

## NUMA policy for experiments

All performance experiments must explicitly configure NUMA memory allocation. This includes index
construction, query throughput and latency, and microbenchmarks. Apply the policy before launching the
process so that data loading, index construction, warmup, and measurement all inherit it.

Use `numactl --interleave=all` by default, matching this repository's `scripts/quick-run.sh` and the
benchmark repository's `scripts/run_all.sh`. For example, from the ARTEA repository root:

```sh
numactl --interleave=all bash ./scripts/run_bench.sh bench_random_eg
```

Using all CPU cores or setting a thread count does not configure NUMA memory placement. An implicit
allocation policy can change memory distribution across nodes and introduce substantial variation in
multithreaded measurements.

- Check the machine's NUMA topology with `numactl --hardware` before running experiments.
- Verify that the requested policy can be applied with `numactl --interleave=all numactl --show`.
  When validating a new launcher, inspect `/proc/<pid>/numa_maps` to confirm the benchmark process's policy.
- Record the complete launch command, NUMA policy, CPU affinity, and thread settings with the results.
- Keep NUMA policy, CPU affinity, and thread settings consistent across compared runs, unless one of
  these settings is the variable being studied. Document any intentional differences.
- Check the launch scripts and recorded settings before comparing with saved or legacy results.
  If their NUMA policy is unknown, state that limitation instead of assuming the environments match.
- If `numactl` is unavailable or the policy cannot be applied, report the limitation. Do not silently
  omit the NUMA setting and present the results as a comparison under the standard configuration.

These are runtime requirements for experiments; they are not CMake compiler options.

## Temporary validation artifacts

Store temporary validation results, logs, plots, configuration snapshots, and generated indexes under
`temp/validation/` relative to the ARTEA repository root. In the benchmark checkout, this is
`third-party/artea/temp/validation/`. Do not write temporary validation artifacts to either repository's
top-level `results/` directory; those directories contain the maintained benchmark results.

Use a separate experiment directory and run directory for each validation. If a runner writes to
`results/` relative to its working directory, launch it from the temporary run directory so all outputs
remain under `temp/validation/`. Resolve dataset and workload paths before changing the working directory.
Keep these temporary artifacts ignored by Git; this repository already ignores `temp/`.
