# How to Implement a New Graph Index in Artea Framework

This guide walks through the complete process of adding a new graph index type to the Artea framework, using `knn_graph` as a running example. The process consists of **6 steps**:

| Step | What | Where |
|------|------|-------|
| 1 | Define Configs | `include/artea/cpu/configs/` |
| 2 | Define Index Structure | `include/artea/cpu/index/` |
| 3 | Implement Index Factory | `include/artea/cpu/graph_factory/` |
| 4 | Register in the Type System | `include/artea/cpu/framework/type_traits/` |
| 5 | Export via Context and Include | `include/artea/cpu/framework/type_context/` and `include/artea/cpu/framework/artea.hpp` |
| 6 | Write Tests | `unit_tests/` |

---

## Step 1 : Define Configs

**Directory:** `include/artea/cpu/configs/`

Every graph type needs a **PruningConfig** and a **PropagateConfig**. You can either define new config classes or reuse existing ones via aliases.

### Option A  Reuse existing configs (recommended when semantics are identical)

Add a namespace alias at the bottom of `pruning_config.hpp` and `propagate_config.hpp`:

```cpp
// pruning_config.hpp
namespace my_graph {

template <typename BaseTraitsT>
using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;

}   // namespace my_graph
```

```cpp
// propagate_config.hpp
namespace my_graph {

template <typename BaseTraitsT>
using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;

}   // namespace my_graph
```

### Option B  Define a brand-new config

Create a new struct in the corresponding config file under `namespace my_graph`. The struct must:

- Be templated on `BaseTraitsT`
- Extract scalar types (e.g. `ratio_t`, `iter_t`) from `BaseTraitsT`
- Provide **const getters** and **builder-pattern setters** (chainable)

```cpp
namespace my_graph {

template <typename BaseTraitsT>
struct PropagateConfig {
    using iter_t = typename BaseTraitsT::iter_t;
    // ... members, constructor, getters, setters ...
};

}   // namespace my_graph
```

### Key points

- Always place config definitions inside `namespace artea::cpu::<graph_name>`.
- If reusing, the alias must appear after the original definition.

---

## Step 2 : Define Index Structure

**Directory:** `include/artea/cpu/index/`

The index structure holds the graph data and any graph-type-specific configurations.

### Option A  Reuse an existing index

If your new graph type shares the same member variables as an existing one (e.g. `conv_graph::IndexStructure`), you can simply alias it in the type system (Step 4) without creating a new file. For example, `knn_graph` reuses `conv_graph::IndexStructure` directly.

### Option B  Define a new IndexStructure

Create a new header (e.g. `my_graph_index.hpp`) following the CRTP pattern:

```cpp
// include/artea/cpu/index/my_graph_index.hpp
namespace artea {
namespace cpu {
namespace my_graph {

template <typename IndexTraitsT>
class IndexStructure :
    public IndexTraitsT::template flat_graph_t<IndexStructure<IndexTraitsT>>
{
    using base_t = typename IndexTraitsT::template flat_graph_t<IndexStructure<IndexTraitsT>>;
    using vector_array_t   = typename IndexTraitsT::vector_array_t;
    using layer_config_t   = typename IndexTraitsT::layer_config_t;
    using propagate_config_t = typename IndexTraitsT::my_graph::propagate_config_t;
    using pruning_config_t   = typename IndexTraitsT::my_graph::pruning_config_t;

public:
    IndexStructure(
        const vector_array_t& vecs_data,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) : base_t(vecs_data, layer_config),
        _pruning_config(pruning_config),
        _propagate_config(propagate_config)
    {}

    // Move-only semantics
    IndexStructure(const IndexStructure&) = delete;
    IndexStructure& operator=(const IndexStructure&) = delete;
    IndexStructure(IndexStructure&&) noexcept = default;
    IndexStructure& operator=(IndexStructure&&) noexcept = default;

    // Config accessors
    auto pruning_config()   const -> const pruning_config_t&   { return _pruning_config; }
    auto propagate_config() const -> const propagate_config_t& { return _propagate_config; }

    // Metadata serialization (for persistence)
    auto get_metadata() const -> nlohmann::json { /* ... */ }
    static auto from_metadata(const nlohmann::json&, const vector_array_t&, const layer_config_t&) -> IndexStructure { /* ... */ }

private:
    pruning_config_t   _pruning_config;
    propagate_config_t _propagate_config;
};

}   // namespace my_graph
}   // namespace cpu
}   // namespace artea
```

### Key points

- Use CRTP: inherit from `IndexTraitsT::template flat_graph_t<DerivedClass>`.
- Delete copy constructor/assignment; default move constructor/assignment.
- Extract all types from traits, never hard-code (e.g. never write `using distance_t = float`).

---

## Step 3 : Implement Index Factory

**Directory:** `include/artea/cpu/graph_factory/`

The factory is a static class that constructs the graph. Create `my_graph_factory.hpp`:

```cpp
namespace artea {
namespace cpu {
namespace my_graph {

template <typename GraphFactoryTraitsT>
class IndexFactory {

    // Extract types from traits
    using index_t            = typename GraphFactoryTraitsT::my_graph::index_t;
    using propagate_config_t = typename GraphFactoryTraitsT::my_graph::propagate_config_t;
    using pruning_config_t   = typename GraphFactoryTraitsT::my_graph::pruning_config_t;
    using dist_func_t        = typename GraphFactoryTraitsT::dist_func_t;
    using vector_array_t     = typename GraphFactoryTraitsT::vector_array_t;
    using layer_config_t     = typename GraphFactoryTraitsT::layer_config_t;
    // ... other types from traits as needed ...

    // Parameterize edge generator types on index_t
    using propagate_engine_t = typename GraphFactoryTraitsT::template propagate_engine_t<index_t, false>;
    using triangle_updater_t = typename GraphFactoryTraitsT::template triangle_updater_t<index_t>;
    using reverse_updater_t  = typename GraphFactoryTraitsT::template reverse_updater_t<index_t>;
    using routing_updater_t  = typename GraphFactoryTraitsT::template routing_updater_t<index_t>;
    using truncate_updater_t = typename GraphFactoryTraitsT::template truncate_updater_t<index_t>;
    using random_eg_t        = typename GraphFactoryTraitsT::random_eg_t;

public:
    /** @brief Construct a graph from vector array. */
    static auto construct_graph(
        const vector_array_t& base_vecs,
        const layer_config_t layer_config,
        const pruning_config_t pruning_config,
        const propagate_config_t propagate_config
    ) -> index_t {
        index_t flat_graph(base_vecs, layer_config, pruning_config, propagate_config);
        dist_func_t dist_func(base_vecs.get_vec_dim());
        _build_loop(flat_graph, dist_func, pruning_config, propagate_config);
        return flat_graph;
    }

private:
    static auto _build_loop(
        index_t& flat_graph,
        const dist_func_t& dist_func,
        const pruning_config_t& pruning_config,
        const propagate_config_t& propagate_config,
        std::function<void(iter_t)> on_iter_end = nullptr
    ) -> void {
        // 1. Initialize random edges
        random_eg_t random_eg(dist_func);
        random_eg.generate(flat_graph, init_nbr_size);

        // 2. Create propagate engine and updaters
        propagate_engine_t propagate_engine(num_vertices, dist_func);
        propagate_engine.set_graph(flat_graph);

        auto triangle_updater = propagate_engine.template make_updater<triangle_updater_t>(...);
        auto reverse_updater  = propagate_engine.template make_updater<reverse_updater_t>();
        auto routing_updater  = propagate_engine.template make_updater<routing_updater_t>(...);
        auto truncate_updater = propagate_engine.template make_updater<truncate_updater_t>();

        // 3. Run your build schedule
        //    THIS IS WHERE DIFFERENT GRAPH TYPES DIVERGE.
        //    Design your own iteration pattern here.
        for (iter_t i = 0; i < propagate_config.num_build_loops(); ++i) {
            propagate_engine.run(propagate_config.num_triu_iters(), triangle_updater)
                            .next(reverse_updater).next(truncate_updater);
        }
    }
};

}   // namespace my_graph
}   // namespace cpu
}   // namespace artea
```

### Key points

- The factory is **the only place** where the graph construction algorithm lives.
- All edge generator types (`triangle_updater_t`, `routing_updater_t`, etc.) must be **parameterized on `index_t`**, not left as bare template aliases.
- The `_build_loop` method defines the construction schedule; this is where different graph types express their algorithmic differences. For example:
  - **conv_graph**: routing loop runs `routing -> truncate -> triangle -> truncate -> reverse -> truncate`
  - **knn_graph**: routing loop runs only `routing -> truncate`

---

## Step 4 : Register in the Type System

**Directory:** `include/artea/cpu/framework/type_traits/`

This is the most critical step. You must register your new graph type in **four** trait files, following the inheritance chain:

```
BaseTraits  ->  IndexTraits  ->  EdgeGeneratorTraits  ->  GraphFactoryTraits
(configs)       (index_t)        (inherited)              (factory_t)
```

### 4.1 `base_traits.hpp` -- Forward declarations + Config types

**Two locations** need to be updated:

**(a) Forward declaration area** (before `BaseTraits` struct):

```cpp
// After the artea_graph forward declarations
namespace my_graph {
    template <typename BaseTraitsT> using PropagateConfig = conv_graph::PropagateConfig<BaseTraitsT>;
    template <typename BaseTraitsT> using PruningConfig = conv_graph::PruningConfig<BaseTraitsT>;
}
```

> If you defined brand-new config classes (Option B in Step 1), use `struct` forward declarations instead of `using` aliases here.

**(b) Inside `BaseTraits` struct** (nested struct):

```cpp
struct my_graph {
    my_graph() = delete;
    using propagate_config_t = cpu::my_graph::PropagateConfig<base_traits_t>;
    using pruning_config_t = cpu::my_graph::PruningConfig<base_traits_t>;
};
```

> The `= delete` on the constructor prevents accidental instantiation of the namespace-like struct.

### 4.2 `index_traits.hpp` -- Register `index_t`

Add a nested struct that extends `BaseTraitsT::my_graph` and adds the `index_t` alias:

```cpp
struct my_graph : BaseTraitsT::my_graph {
    my_graph() = delete;
    using index_t = cpu::my_graph::IndexStructure<index_traits_t>;
};
```

If reusing `conv_graph::IndexStructure`:

```cpp
struct my_graph : BaseTraitsT::my_graph {
    my_graph() = delete;
    using index_t = cpu::conv_graph::IndexStructure<index_traits_t>;
};
```

### 4.3 `graph_factory_traits.hpp` -- Register `factory_t`

**(a) Forward declaration** (top of file):

```cpp
namespace my_graph {
    template <typename GraphFactoryTraitsT> class IndexFactory;
}
```

**(b) Inside `GraphFactoryTraits` struct:**

```cpp
struct my_graph : EdgeGeneratorTraitsT::my_graph {
    my_graph() = delete;
    using factory_t = cpu::my_graph::IndexFactory<graph_factory_traits_t>;
};
```

### Why the nested struct chain?

Each trait level inherits the previous one's nested struct:

```
BaseTraits::my_graph           -> { propagate_config_t, pruning_config_t }
IndexTraits::my_graph          -> { ..., index_t }
EdgeGeneratorTraits::my_graph  -> { ... } (inherited as-is)
GraphFactoryTraits::my_graph   -> { ..., factory_t }
```

By the time `GraphFactoryTraits::my_graph` is fully composed, it carries **all** the types needed to construct and use the graph.

---

## Step 5 : Export via Context and Include

### 5.1 `default_context.hpp`

**File:** `include/artea/cpu/framework/type_context/default_context.hpp`

Add a namespace block that extracts all types from the fully-composed traits:

```cpp
namespace my_graph {
    using index_t            = typename graph_factory_traits_t::my_graph::index_t;
    using factory_t          = typename graph_factory_traits_t::my_graph::factory_t;
    using propagate_config_t = typename graph_factory_traits_t::my_graph::propagate_config_t;
    using pruning_config_t   = typename graph_factory_traits_t::my_graph::pruning_config_t;
}
```

This makes `my_graph::index_t`, `my_graph::factory_t`, etc. available as convenient top-level aliases to end users.

### 5.2 `artea.hpp`

**File:** `include/artea/cpu/framework/artea.hpp`

Add `#include` directives for any **new** files you created:

```cpp
// In the index section (if you created a new IndexStructure)
#include <artea/cpu/index/my_graph_index.hpp>

// In the graph_factory section
#include <artea/cpu/graph_factory/my_graph_factory.hpp>
```

---

## Step 6 : Write Tests

**Directory:** `unit_tests/`

Create `test_my_graph_construct.cpp`. The CMake configuration (`unit_tests/CMakeLists.txt`) uses a `file(GLOB ...)` pattern, so any `.cpp` file in this directory is automatically picked up as a test target.

A typical test follows this pattern:

### 6.1 Build the graph

```cpp
#include <artea/cpu/framework/artea.hpp>
#include <artea/cpu/framework/type_context/default_context.hpp>

using namespace artea;
using namespace artea::cpu;

// Create configs
layer_config_t layer_config(max_nbr_size, reserved_nbr_size);
my_graph::pruning_config_t pruning_config(scale_coeffs, shifted_coeffs);
my_graph::propagate_config_t propagate_config(num_build_loops, num_triu_iters, prefill_ratio, num_routing_loops);

// Construct graph
auto flat_graph = my_graph::factory_t::construct_graph(
    base_vecs, layer_config, pruning_config, propagate_config
);
```

### 6.2 Convert to search graph

For search evaluation, convert to a CSR-format flat search graph:

```cpp
auto flat_search_graph = search_graph_converter_t::from_flat_graph(flat_graph, extracted_nbr_size);
```

### 6.3 Query with monolayer_graph_router and measure QPS vs Recall

```cpp
recall_estimator_t recall_estimator;

for (uint32_t queue_size = start; queue_size <= end; queue_size += step) {
    monolayer_graph_router_t<graph_mode_t::search_mode> router(
        base_vecs, dist_func, flat_search_graph, topk, queue_size
    );
    router.initialize();

    auto t0 = std::chrono::high_resolution_clock::now();
    knn_results_t results = router.batch_query(query_vecs);
    auto t1 = std::chrono::high_resolution_clock::now();

    double qps = query_vecs.get_num_vecs() * 1e6 /
        std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    float recall = recall_estimator.calculate_recall_at_k(
        results, groundtruth, topk, query_vecs.get_num_vecs());
}
```

### 6.4 Use argparse for configurable parameters

Expose all build and search parameters as command-line arguments so experiments are reproducible:

```cpp
argparse::ArgumentParser program("test_my_graph");
program.add_argument("--max-nbr-size").default_value(32u).scan<'u', uint32_t>();
program.add_argument("--num-build-loops").default_value(4u).scan<'u', uint32_t>();
// ... more arguments ...
```

---

## Quick Checklist

Before submitting your changes, verify every item:

- [ ] **Configs** declared in `include/artea/cpu/configs/` (new structs or namespace aliases)
- [ ] **IndexStructure** defined in `include/artea/cpu/index/` (or reused via type alias)
- [ ] **IndexFactory** implemented in `include/artea/cpu/graph_factory/`
- [ ] **base_traits.hpp** -- forward declarations **and** nested struct with config types
- [ ] **index_traits.hpp** -- nested struct with `index_t`
- [ ] **graph_factory_traits.hpp** -- forward declaration **and** nested struct with `factory_t`
- [ ] **default_context.hpp** -- namespace block exporting all types
- [ ] **artea.hpp** -- `#include` for all new headers
- [ ] **Test file** in `unit_tests/` -- builds graph, converts to search graph, measures QPS vs Recall
- [ ] **Compiles** and **tests pass**
