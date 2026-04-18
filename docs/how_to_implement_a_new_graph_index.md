# How to Implement a New Graph Index in Artea Framework

This guide uses `my_graph` as a placeholder name. The process consists of **6 steps**:

| Step | What | Where |
|------|------|-------|
| 1 | Define Configs | `include/artea/cpu/index/my_graph/configs.hpp` |
| 2 | Define Index Structure | `include/artea/cpu/index/my_graph/index_structure.hpp` |
| 3 | Implement Index Factory | `include/artea/cpu/index/my_graph/index_factory.hpp` |
| 4 | Register in the Type System | `include/artea/cpu/framework/type_traits/` |
| 5 | Export via Context and Include | `type_context/default_context.hpp` + `artea.hpp` |
| 6 | Write Tests | `unit_tests/` |

---

## Step 1 : Define Configs

**File:** `include/artea/cpu/index/my_graph/configs.hpp`

Each graph type has its own configs file. **LayerConfig** (in `index/layer_config.hpp`) is shared and mandatory for every graph type. Graph-specific configs (e.g. PruningConfig, PropagateConfig) live in `index/my_graph/configs.hpp`.

If a config you need already exists in another namespace, include it and reuse via alias:

```cpp
#include <artea/cpu/index/conv_graph/configs.hpp>

namespace artea::cpu::my_graph {

template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}
```

If you need a config with different semantics, define a new struct templated on `BaseTraitsT` with const getters and builder-pattern setters. Always extract scalar types from `BaseTraitsT`.

---

## Step 2 : Define Index Structure

**File:** `include/artea/cpu/index/my_graph/index_structure.hpp`

If the new graph shares member variables with an existing one, create a thin alias file (e.g. `knn_graph/index_structure.hpp` reuses `conv_graph::IndexStructure`):

```cpp
#include <artea/cpu/index/conv_graph/index_structure.hpp>

namespace artea::cpu::my_graph {

template <typename IndexTraitsT>
using IndexStructure = conv_graph::IndexStructure<IndexTraitsT>;

}
```

Otherwise, define a new class that **composes** a `dynamic::RefiningGraph`
(it is an ordinary class — no CRTP):

```cpp
namespace artea::cpu::my_graph {

template <typename IndexTraitsT>
class IndexStructure {
    using refining_graph_t   = typename IndexTraitsT::dynamic::refining_graph_t;
    using propagate_config_t = typename IndexTraitsT::my_graph::propagate_config_t;
    using pruning_config_t   = typename IndexTraitsT::my_graph::pruning_config_t;

public:
    IndexStructure(const vector_array_t& vecs, const layer_config_t cfg,
                   const pruning_config_t pc, const propagate_config_t pp)
        : _refining_graph(std::make_unique<refining_graph_t>(vecs, cfg)),
          _pruning_config(pc), _propagate_config(pp) {}

    // Move-only
    IndexStructure(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&) noexcept = default;

    auto pruning_config()   const -> const pruning_config_t&   { return _pruning_config; }
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

private:
    pruning_config_t   _pruning_config;
    propagate_config_t _propagate_config;
};

} // namespace artea::cpu::my_graph
```

**Rules:** inherit via CRTP, move-only semantics, extract all types from traits (never hard-code).

---

## Step 3 : Implement Index Factory

**File:** `include/artea/cpu/index/my_graph/index_factory.hpp`

The factory is a static class containing the build algorithm. Key skeleton:

```cpp
namespace artea::cpu::my_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {
    using this_index_t       = typename GraphFactoryTraitsT::my_graph::index_t;
    using propagate_config_t = typename GraphFactoryTraitsT::my_graph::propagate_config_t;
    using pruning_config_t   = typename GraphFactoryTraitsT::my_graph::pruning_config_t;
    // Parameterize edge generators on this_index_t
    using propagate_engine_t = typename GraphFactoryTraitsT::propagate_engine_t;
    using triangle_updater_t = typename GraphFactoryTraitsT::template triangle_updater_t<this_index_t>;
    using routing_updater_t  = typename GraphFactoryTraitsT::template routing_updater_t<this_index_t>;
    // ... other updater types ...

public:
    static auto construct_graph(const vector_array_t& base_vecs, ...) -> this_index_t {
        this_index_t refining_graph(base_vecs, layer_config, pruning_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());
        _build_loop(refining_graph, dist_func, pruning_config, propagate_config);
        return refining_graph;
    }

private:
    static auto _build_loop(this_index_t& refining_graph, ...) -> void {
        // 1. Initialize random edges
        // 2. Create propagate engine + updaters
        // 3. Run your build schedule  <-- THIS IS WHERE GRAPH TYPES DIVERGE
    }
};

} // namespace artea::cpu::my_graph
```

**The `_build_loop` schedule is the core differentiator.** For example:
- **conv_graph** routing loop: `routing -> truncate -> triangle -> truncate -> reverse -> truncate`
- **knn_graph** routing loop: `routing -> truncate`

---

## Step 4 : Register in the Type System

**Directory:** `include/artea/cpu/framework/type_traits/`

Register your type through the trait inheritance chain:

```
BaseTraits  -->  IndexTraits  -->  (RefinerTraits)  -->  GraphFactoryTraits
 configs          index_t           (pass-through)              factory_t
```

### 4.1 `base_traits.hpp`

**(a) Forward declaration area** (before `BaseTraits` struct):
```cpp
namespace my_graph {
    template <typename BaseTraitsT> using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;
    template <typename BaseTraitsT> using PruningConfig   = conv_graph::PruningConfig<BaseTraitsT>;
}
```

**(b) Inside `BaseTraits` struct** (nested struct):
```cpp
struct my_graph {
    my_graph() = delete;
    using propagate_config_t = cpu::my_graph::PropagateConfig<base_traits_t>;
    using pruning_config_t   = cpu::my_graph::PruningConfig<base_traits_t>;
};
```

### 4.2 `index_traits.hpp`

```cpp
struct my_graph : BaseTraitsT::my_graph {
    my_graph() = delete;
    using index_t = cpu::my_graph::IndexStructure<index_traits_t>;
    // or reuse: using index_t = cpu::conv_graph::IndexStructure<index_traits_t>;
};
```

### 4.3 `graph_factory_traits.hpp`

**(a) Forward declaration:**
```cpp
namespace my_graph { template <typename GraphFactoryTraitsT> class IndexFactory; }
```

**(b) Inside `GraphFactoryTraits`:**
```cpp
struct my_graph : RefinerTraitsT::my_graph {
    my_graph() = delete;
    using factory_t = cpu::my_graph::IndexFactory<graph_factory_traits_t>;
};
```

The nested struct chain accumulates all types:
```
BaseTraits::my_graph          = { propagate_config_t, pruning_config_t }
IndexTraits::my_graph         = { ..., index_t }
GraphFactoryTraits::my_graph  = { ..., factory_t }
```

---

## Step 5 : Export via Context and Include

### `default_context.hpp`

```cpp
namespace my_graph {
    using index_t            = typename graph_factory_traits_t::my_graph::index_t;
    using factory_t          = typename graph_factory_traits_t::my_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::my_graph::propagate_config_t;
    using pruning_config_t   = typename graph_factory_traits_t::my_graph::pruning_config_t;
}
```

### `artea.hpp`

Add `#include` for the new graph's files:
```cpp
#include <artea/cpu/index/my_graph/configs.hpp>
#include <artea/cpu/index/my_graph/index_structure.hpp>
#include <artea/cpu/index/my_graph/index_factory.hpp>
```

---

## Step 6 : Write Tests

**Directory:** `unit_tests/`

CMake auto-discovers all `.cpp` files via `file(GLOB ...)`. Create `test_my_graph_construct.cpp`:

```cpp
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>
using namespace artea::cpu;

// 1. Build
auto refining_graph = my_graph::factory_t::construct_graph(
    base_vecs, layer_config, pruning_config, propagate_config);

// 2. Convert to search graph
auto search_graph = refining_graph_compactor_t::from_refining_graph(refining_graph, extracted_nbr_size);

// 3. Grid-search QPS vs Recall
for (uint32_t qs = start; qs <= end; qs += step) {
    single_layer_router_t router(base_vecs, dist_func, topk, qs);
    router.initialize();

    auto t0 = std::chrono::high_resolution_clock::now();
    auto results = router.batch_query(query_vecs, search_graph);
    auto t1 = std::chrono::high_resolution_clock::now();

    double qps   = num_queries * 1e6 / duration_us(t1 - t0);
    float recall  = recall_estimator.calculate_recall_at_k(results, gt, topk, num_queries);
}
```

Use `argparse` to expose all parameters as CLI arguments for reproducibility.

---

## Checklist

- [ ] Configs in `index/my_graph/configs.hpp` (alias or new struct)
- [ ] IndexStructure in `index/my_graph/index_structure.hpp` (or reused via alias)
- [ ] IndexFactory in `index/my_graph/index_factory.hpp`
- [ ] `base_traits.hpp` -- forward declarations + nested struct
- [ ] `index_traits.hpp` -- nested struct with `index_t`
- [ ] `graph_factory_traits.hpp` -- forward declaration + nested struct with `factory_t`
- [ ] `default_context.hpp` -- namespace block
- [ ] `artea.hpp` -- `#include` new headers
- [ ] Test in `unit_tests/` -- build, convert, measure QPS vs Recall
- [ ] Compiles and tests pass
