# How to Implement a New Updater in Artea Framework

This guide uses `MyUpdater` as a placeholder name. The process consists of **4 steps**:

| Step | What | Where |
|------|------|-------|
| 1 | Implement the Updater | `include/artea/cpu/refiner/my_updater.hpp` |
| 2 | Register in the Type System | `include/artea/cpu/framework/type_traits/refiner_traits.hpp` |
| 3 | Add to PropagateEngine | `include/artea/cpu/refiner/propagate_engine.hpp` |
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

**File:** `include/artea/cpu/refiner/my_updater.hpp`

All updaters inherit from `NeighborUpdater` via CRTP:

```cpp
namespace artea::cpu {

template <typename RefinerTraitsT, typename RefiningGraphT>
class MyUpdater :
    public RefinerTraitsT::template neighbor_updater_t<
        RefiningGraphT, MyUpdater<RefinerTraitsT, RefiningGraphT>>
{
    // Extract types from traits
    using vertex_id_t    = typename RefinerTraitsT::vertex_id_t;
    using vertex_num_t   = typename RefinerTraitsT::vertex_num_t;
    using distance_t     = typename RefinerTraitsT::distance_t;
    using nbr_t          = typename RefinerTraitsT::nbr_t;
    using nbr_arr_t      = typename RefinerTraitsT::nbr_arr_t;
    using log_table_t    = typename RefinerTraitsT::log_table_t;
    using dist_func_t    = typename RefinerTraitsT::dist_func_t;
    using vector_array_t = typename RefinerTraitsT::vector_array_t;
    using base_class_t   = typename RefinerTraitsT::template neighbor_updater_t<
        RefiningGraphT, MyUpdater<RefinerTraitsT, RefiningGraphT>>;

public:
    // Required: unique name for debug/profiling output
    static constexpr const char* updater_name = "my_updater";

    // Constructor: always starts with the 4 base args, then updater-specific args
    MyUpdater(
        const dist_func_t& dist_func,
        const vector_array_t& vecs_data,
        log_table_t& log_table,
        const RefiningGraphT& refining_graph
        // ... updater-specific args ...
    ) : base_class_t(dist_func, vecs_data, log_table, refining_graph)
        // ... initialize updater-specific members ...
    {}

    // Required: the core logic, called once per vertex during propagation
    auto update_impl(
        const vertex_id_t pivot_vid,
        nbr_arr_t& origin_nbrs
    ) -> void {
        // Access base class members via this->:
        //   this->_dist_func    -- distance function
        //   this->_vecs_data    -- vector array (use .get(vid) to get raw pointer)
        //   this->_log_table    -- log table for buffering new edges
        //   this->_refining_graph   -- read-only graph reference

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

- **CRTP inheritance**: always inherit from `RefinerTraitsT::template neighbor_updater_t<RefiningGraphT, YourClass>`
- **`updater_name`**: required `static constexpr const char*`, used in profiling output
- **`update_impl`**: required method, called by base class `operator()` via CRTP
- **Constructor**: first 4 args are always `(dist_func, vecs_data, log_table, refining_graph)`, passed to base class; extra args are updater-specific
- **Extract types from `RefinerTraitsT`**, never hard-code

---

## Step 2 : Register in the Type System

**File:** `include/artea/cpu/framework/type_traits/refiner_traits.hpp`

**(a) Forward declaration** (in the forward declaration area):

```cpp
template <typename RefinerTraitsT, typename RefiningGraphT> class MyUpdater;
```

**(b) Type alias** (inside `RefinerTraits` struct):

```cpp
template <typename RefiningGraphT>
using my_updater_t = MyUpdater<refiner_traits_t, RefiningGraphT>;
```

---

## Step 3 : Add to PropagateEngine

**File:** `include/artea/cpu/refiner/propagate_engine.hpp`

Inside the `make_updater` factory method:

**(a) Add local type alias** (alongside the existing ones):

```cpp
using my_updater_t = typename RefinerTraitsT::template my_updater_t<RefiningGraphT>;
```

**(b) Add `if constexpr` branch** (before the `else { ARTEA_ERROR(...) }` fallback):

```cpp
else if constexpr (std::is_same_v<UpdaterT, my_updater_t>) {
    // MyUpdater(dist_func, vecs_arr, log_table, refining_graph, ...extra_args)
    return UpdaterT(_dist_func, vecs_arr, log_table, *_refining_graph, std::forward<Args>(args)...);
}
```

The `make_updater` factory auto-provides `dist_func`, `vecs_arr`, `log_table`, and `refining_graph`. Only updater-specific extra args need to be passed by the caller:

```cpp
// No extra args (like ReverseUpdater, TruncateUpdater):
auto updater = propagate_engine.template make_updater<my_updater_t>();

// With extra args (like TriangleUpdater):
auto updater = propagate_engine.template make_updater<my_updater_t>(arg1, arg2);
```

---

## Step 4 : Export via Include

**File:** `include/artea/cpu/framework/artea.hpp`

Add `#include` in the refiner section:

```cpp
#include <artea/cpu/refiner/my_updater.hpp>
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

- [ ] Updater class in `include/artea/cpu/refiner/my_updater.hpp`
- [ ] CRTP inheritance from `neighbor_updater_t`
- [ ] `static constexpr const char* updater_name` defined
- [ ] `update_impl(pivot_vid, origin_nbrs)` implemented
- [ ] Forward declaration in `refiner_traits.hpp`
- [ ] Type alias `my_updater_t` in `RefinerTraits`
- [ ] `make_updater` branch in `propagate_engine.hpp`
- [ ] `#include` in `artea.hpp`
- [ ] Compiles and tests pass
