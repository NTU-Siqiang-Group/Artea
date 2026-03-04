# Claude Code Instructions

## Coding Style

### Naming Conventions

- **Private member variables**: Use leading underscore (e.g., `_member_variable`)
- **Public member variables**: No underscore prefix (e.g., `member_variable`)
- **Local variables**: No underscore prefix (e.g., `local_var`)
- **Function parameters**: No underscore prefix (e.g., `param_name`)

## Artea Logger

- Use `logger.error()` for error reporting instead of throwing exceptions
- Logger is available globally in the artea namespace

## Git Workflow

- Test code before committing to ensure it compiles and runs correctly
- **IMPORTANT**: Wait for user to complete testing before committing
- Commit and push changes after each modification whenever possible
- Keep commit messages concise: 1-2 sentences, maximum 3 sentences
- Do NOT include "Co-Authored-By" in commit messages

## Framework Type System

### Trait Hierarchy

Framework traits in `include/artea/cpu/framework/`:
- **BaseTraits** - Fundamental types (vertex_id_t, distance_t, nbr_t, vector_array_t, etc.)
- **ComputerTraits** - Distance computation (dist_func_t, simd_dist_t, fma_func_t)
- **BufferTraits** - Buffer management (log_buffer_t, log_table_t)
- **IndexTraits** - Graph structures (flat_graph_t, search_graph_t)
- **VertexGeneratorTraits** - Vertex generation (lsh_generator_t, greedy_vg_t)
- **EdgeGeneratorTraits** - Edge generation (triangle_updater_t, propagate_engine_t)
- **RouterTraits** - Query routing (candidate_entry_t, candidate_queue_t, router_t)
- **GraphFactoryTraits** - Graph construction (graph_factory_t)

### Type Reuse Rules

**MANDATORY**: Always extract types from traits using `typename TraitsT::type_name`:

```cpp
template <typename ComputerTraitsT>
class MyClass {
    // CORRECT: Extract from traits
    using distance_t = typename ComputerTraitsT::distance_t;
    using vertex_id_t = typename ComputerTraitsT::vertex_id_t;

    // WRONG: Never define your own
    // using distance_t = float;  // ❌
};
```

**Exceptions**: Only define local types for:
- Temporary variables local to a single function
- Types not available in any trait and truly implementation-specific
- Standard library types for internal implementation details

**Use virtual inheritance** when composing multiple traits to prevent type duplication.

