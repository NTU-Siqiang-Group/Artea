#pragma once

#include <artea/cpu/index_graph.hpp>

namespace artea {
namespace cpu {

class GraphInitializer {

public:
    GraphInitializer() = default;

    static auto operator()(IndexGraph* graph) -> void {
        _init_impl(graph);
    }

private:

    static virtual auto _init_impl(IndexGraph* graph) -> void = 0;

};  // class GraphInitializer

class RandomInitializer : public GraphInitializer {

public:
    RandomInitializer() = default;

    static virtual auto init(IndexGraph* graph) -> void override {

    }

private:

};  // class RandomInitializer


class LSHInitializer : public GraphInitializer {

public:
    LSHInitializer() = default;

    virtual auto init(IndexGraph* graph) -> void override {

    }

private:

};  // class LSHInitializer

}   // namespace cpu
}   // namespace artea