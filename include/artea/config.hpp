#pragma once

#include <cstdint>
#include <artea/definitions.hpp>
#include <artea/logger.hpp>

namespace artea {

constexpr uint32_t data_loader_threads = 16;

constexpr uint32_t random_nn_threads = 16;

constexpr uint32_t merge_nbrs_threads = 16;

constexpr LogLevel system_log_level = LogLevel::INFO;

}   // namespace artea