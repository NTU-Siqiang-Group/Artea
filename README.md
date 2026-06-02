<div align="center">

<img src="README.assets/ARTEA-logo.png" alt="ARTEA logo" width="60%"/>

**Theory-Guided Hierarchical Graph Index for High-Performance ANN Search**

[![C++23](https://img.shields.io/badge/C%2B%2B-23-00599C?logo=cplusplus&logoColor=white)](https://isocpp.org/)
[![Build](https://img.shields.io/badge/build-CMake-064F8C?logo=cmake&logoColor=white)](https://cmake.org/)
[![Intel oneAPI](https://img.shields.io/badge/Intel-oneAPI-0071C5?logo=intel&logoColor=white)](https://www.intel.com/content/www/us/en/developer/tools/oneapi/overview.html)
[![SIMD](https://img.shields.io/badge/SIMD-AVX-orange)](#)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)

</div>

<div align="center">

📄 &nbsp; $\color{red}{\Large \textsf{Full technical report with complete proofs and derivations:}}$

### ➡️ [Click this link: `artea/artea-technical-report.pdf`](artea-technical-report.pdf)

</div>

ARTEA is a hierarchical proximity-graph index for approximate nearest neighbor
(ANN) search. It pairs a deterministic bottom-up *r*-net hierarchy with
**Aspect-Ratio-Constrained Pruning (ARC-Pruning)**, bounding the worst-case
search complexity to <code>O((α·τ)<sup>λ</sup> + α<sup>λ</sup> log Δ)</code> — the
first proximity graph with a *strictly logarithmic* dependence on the dataset
aspect ratio Δ — while delivering state-of-the-art query throughput and
competitive build time.

---

## Build & Compile

```sh
# 1. Activate the Intel oneAPI environment (provides the SIMD/MKL toolchain)
source $HOME/intel/oneapi/setvars.sh

# 2. Configure + build (clean rebuild optional)
rm -rf build
bash ./scripts/install.sh

# 3. Smoke-test the build
./build/unit_tests/test_artea_graph --help
```

## Quick Start

A minimal build-then-search pipeline (mirrors `bench-artea/build_and_run_artea`):

```cpp
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
using namespace artea;
using namespace artea::cpu;

// 1. Load a dataset (base / query / ground-truth vectors) and a distance fn.
vector_dataset_t dataset("configs/datasets.json", "sift-1m");
const auto& base_vecs = dataset.get_base_vecs();
dist_func_t dist_func(base_vecs.get_vec_dim());

// 2. Configure the three ARTEA stages.
artea_graph::rgraph_config_t    rgraph_cfg(/*beta=*/2.0, /*search_nn_qs=*/64);
artea_graph::propagate_config_t propagate_cfg(/*build_loops=*/15, /*triu_iters=*/4, /*prefill=*/0.4f);
artea_graph::pruning_config_t   pruning_cfg(/*scale=*/1.10, /*shift=*/1.5);

// 3. Build the hierarchical graph, then compact it for fast search.
auto graph = std::make_unique<artea_graph::index_t>(
    base_vecs.get_num_vecs(), rgraph_cfg, propagate_cfg, pruning_cfg);
artea_graph::factory_t::add_vertices(
    *graph, base_vecs.extract_subset(0, base_vecs.get_num_vecs()), dist_func,
    /*insert_on_L0=*/false, /*shuffle=*/true);
auto compact_hg = hierarchical_graph_compactor_t::compact_graph(
    graph->get_hierarchical_graph(), base_vecs, dist_func);

// 4. Search: top-k with a runtime candidate-queue budget.
hierarchical_graph_router_t router(base_vecs, dist_func, /*topk=*/100, /*queue=*/200);
router.initialize();
auto results = router.batch_query</*RandomSeeding=*/false, /*UpperBeam=*/false>(
    dataset.get_query_vecs(), compact_hg);
```

> Build parameters (`α = scale`, `τ = shift`, `β`, neighbor budgets, refinement
> loops) are tunable per dataset; the `bench-artea` workloads under
> `workloads/*.jsonc` provide recommended settings.

## Performance

**Recall@100 vs. throughput** — ARTEA traces the best QPS–recall Pareto frontier
against both flat and hierarchical baselines.

|            | SIFT-1M | YahooMusic |
|:----------:|:-------:|:----------:|
| **vs. flat** (NSG / Vamana / τ-MNG / α-CNG)  | <img src="README.assets/flat_sift1m_recall100.png" width="320"/> | <img src="README.assets/flat_yahoomusic_recall100.png" width="320"/> |
| **vs. hierarchical** (HNSW / HCNNG / MIRAGE) | <img src="README.assets/hier_sift1m_recall100.png" width="320"/> | <img src="README.assets/hier_yahoomusic_recall100.png" width="320"/> |

**Index construction time** (relative to ARTEA, lower is better) across all datasets:

<div align="center">
<img src="README.assets/build_time_all.png" alt="Build time across all datasets" width="760"/>
</div>

## Citation

If you use ARTEA, please cite the repository:

```bibtex
@software{artea_repo,
  title     = {{ARTEA}: Theory-Guided Hierarchical Graph Index for High-Performance ANN Search},
  author    = {Ye, Weitang and Mo, Dingheng and Luo, Siqiang},
  year      = {2026},
  publisher = {GitHub},
  url       = {https://github.com/NTU-Siqiang-Group/Artea}
}
```

