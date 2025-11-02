#pragma once

#include <cstdint>
#include <artea/types.hpp>
#include <artea/logger.hpp>

namespace artea {

constexpr uint32_t data_loader_threads = 16;

constexpr uint32_t graph_transpose_threads = 16;

constexpr uint32_t fast_1nn_threads = 16;

constexpr LogLevel system_log_level = LogLevel::INFO;

}   // namespace artea