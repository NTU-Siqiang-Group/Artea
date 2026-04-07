# How to Implement a New Updater in Artea Framework

This guide uses `MyUpdater` as a placeholder name. The process consists of **4 steps**:

| Step | What | Where |
|------|------|-------|
| 1 | Implement the Updater | `include/artea/cpu/edge_generator/my_updater.hpp` |
| 2 | Register in the Type System | `include/artea/cpu/framework/type_traits/edge_generator_traits.hpp` |
| 3 | Add to PropagateEngine | `include/artea/cpu/edge_generator/propagate_engine.hpp` |
| 4 | Export via Include | `include/artea/cpu/framework/artea.hpp` |

---

## Background: How Updaters Work

Updaters are the building blocks of graph construction. The `PropagateEngine` drives the build schedule by calling updaters in sequence:

```
PropagateEngine::next(updater)
  1. propagate(updater)   -- calls updater(pivot_vid, origin_nbrs) for every vertex in parallel
  2. merge_logs()         -- applies all buffered log entries back to the graph
```

Each updater can do two things:
- **Modify `origin_nbrs` in-place** (e.g. pruning, truncation)
- **Write to `_log_table`** to buffer new edges that get merged after propagation

Existing updaters and their behaviors:

| Updater | Writes logs | Modifies origin_nbrs | Extra constructor args |
|---------|-------------|----------------------|----------------------|
| `TriangleUpdater` | Yes (reverse edges on RNG conflict) | Yes (prunes) | `scale_coeffs`, `shifted_coeffs` |
| `PruningUpdater` | No | Yes (prunes) | `scale_coeffs`, `shifted_coeffs` |
| `ReverseUpdater` | Yes (reverse edges) | No | None |
| `RandomUpdater` | Yes (random neighbors) | No | `rand_gen_size` |
| `RoutingUpdater` | Yes (KNN results) | No | `topk`, `candidate_queue_size` |
| `TruncateUpdater` | No | Yes (resize) | None |

---

## Step 1 : Implement the Updater

**File:** `include/artea/cpu/edge_generator/my_updater.hpp`

All updaters inherit from `NeighborUpdater` via CRTP:

```cpp
namespace artea::cpu {

template <typename EdgeGeneratorTraitsT, typename FlatGraphT>
class MyUpdater :
    public EdgeGeneratorTraitsT::template neighbor_updater_t<
        FlatGraphT, MyUpdater<EdgeGeneratorTraitsT, FlatGraphT>>
{
    // Extract types from traits
    using vertex_id_t    = typename EdgeGeneratorTraitsT::vertex_id_t;
    using vertex_num_t   = typename EdgeGeneratorTraitsT::vertex_num_t;
    using distance_t     = typename EdgeGeneratorTraitsT::distance_t;
    using dnbr_t          = typename EdgeGeneratorTraitsT::dnbr_t;
    using dnbr_arr_t      = typename EdgeGeneratorTraitsT::dnbr_arr_t;
    using log_table_t    = typename EdgeGeneratorTraitsT::log_table_t;
    using dist_func_t    = typename EdgeGeneratorTraitsT::dist_func_t;
    using vector_array_t = typename EdgeGeneratorTraitsT::vector_array_t;
    using base_class_t   = typename EdgeGeneratorTraitsT::template neighbor_updater_t<
        FlatGraphT, MyUpdater<EdgeGeneratorTraitsT, FlatGraphT>>;

public:
    // Required: unique name for debug/profiling output
    static constexpr const char* updater_name = "my_updater";

    // Constructor: always starts with the 4 base args, then updater-specific args
    MyUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const FlatGraphT& flat_graph
        // ... updater-specific args ...
    ) : base_class_t(dist_func, vecs_data, log_table, flat_graph)
        // ... initialize updater-specific members ...
    {}

    // Required: the core logic, called once per vertex during propagation
    auto update_impl(
        const vertex_id_t pivot_vid,
        dnbr_arr_t& origin_nbrs
    ) -> void {
        // Access base class members via this->:
        //   this->_dist_func    -- distance function
        //   this->_vecs_data    -- vector array (use .get(vid) to get raw pointer)
        //   this->_log_table    -- log table for buffering new edges
        //   this->_flat_graph   -- read-only graph reference

        // Option A: Write logs (deferred edge additions, merged after propagation)
        this->_log_table.write_log(executor_vid, nbr_id, distance);
        // or batch version:
        this->_log_table.write_logs(executor_vid, nbr_ids, distances);

        // Option B: Modify origin_nbrs in-place (immediate effect)
        origin_nbrs.resize(new_size);
        std::swap(origin_nbrs, new_nbrs);
    }

private:
    // Updater-specific members ...
};

} // namespace artea::cpu
```

### Rules

- **CRTP inheritance**: always inherit from `EdgeGeneratorTraitsT::template neighbor_updater_t<FlatGraphT, YourClass>`
- **`updater_name`**: required `static constexpr const char*`, used in profiling output
- **`update_impl`**: required method, called by base class `operator()` via CRTP
- **Constructor**: first 4 args are always `(dist_func, vecs_data, log_table, flat_graph)`, passed to base class; extra args are updater-specific
- **Extract types from `EdgeGeneratorTraitsT`**, never hard-code

---

## Step 2 : Register in the Type System

**File:** `include/artea/cpu/framework/type_traits/edge_generator_traits.hpp`

**(a) Forward declaration** (in the forward declaration area):

```cpp
template <typename EdgeGeneratorTraitsT, typename FlatGraphT> class MyUpdater;
```

**(b) Type alias** (inside `EdgeGeneratorTraits` struct):

```cpp
template <typename FlatGraphT>
using my_updater_t = MyUpdater<edge_generator_traits_t, FlatGraphT>;
```

---

## Step 3 : Add to PropagateEngine

**File:** `include/artea/cpu/edge_generator/propagate_engine.hpp`

Inside the `make_updater` factory method:

**(a) Add local type alias** (alongside the existing ones):

```cpp
using my_updater_t = typename EdgeGeneratorTraitsT::template my_updater_t<FlatGraphT>;
```

**(b) Add `if constexpr` branch** (before the `else { ARTEA_ERROR(...) }` fallback):

```cpp
else if constexpr (std::is_same_v<UpdaterT, my_updater_t>) {
    // MyUpdater(dist_func, vecs_arr, log_table, flat_graph, ...extra_args)
    return UpdaterT(_dist_func, vecs_arr, log_table, *_flat_graph, std::forward<Args>(args)...);
}
```

The `make_updater` factory auto-provides `dist_func`, `vecs_arr`, `log_table`, and `flat_graph`. Only updater-specific extra args need to be passed by the caller:

```cpp
// No extra args (like ReverseUpdater, TruncateUpdater):
auto updater = propagate_engine.template make_updater<my_updater_t>();

// With extra args (like TriangleUpdater):
auto updater = propagate_engine.template make_updater<my_updater_t>(arg1, arg2);
```

---

## Step 4 : Export via Include

**File:** `include/artea/cpu/framework/artea.hpp`

Add `#include` in the edge_generator section:

```cpp
#include <artea/cpu/edge_generator/my_updater.hpp>
```

---

## Using the Updater in a Build Schedule

In an `IndexFactory`'s `_build_loop`, add the type alias and use it:

```cpp
using my_updater_t = typename GraphFactoryTraitsT::template my_updater_t<this_index_t>;

auto my_updater = propagate_engine.template make_updater<my_updater_t>(...);

// Use in a build schedule chain:
propagate_engine.next(my_updater).next(truncate_updater);

// Or run multiple iterations:
propagate_engine.run(num_iters, my_updater);
```

---

## Checklist

- [ ] Updater class in `include/artea/cpu/edge_generator/my_updater.hpp`
- [ ] CRTP inheritance from `neighbor_updater_t`
- [ ] `static constexpr const char* updater_name` defined
- [ ] `update_impl(pivot_vid, origin_nbrs)` implemented
- [ ] Forward declaration in `edge_generator_traits.hpp`
- [ ] Type alias `my_updater_t` in `EdgeGeneratorTraits`
- [ ] `make_updater` branch in `propagate_engine.hpp`
- [ ] `#include` in `artea.hpp`
- [ ] Compiles and tests pass
