/*
 * @FilePath: /Artea/include/artea/logger.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-11-15 16:18:30
 * @Date: 2025-10-23 13:26:15
 * @Description: A logger that supports colorful printing via termcolor.
 */

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>

#include <fmt/format.h>

#include <artea/definitions.hpp>
// Reference: https://github.com/ikaln/termcolor/blob/master/include/termcolor/termcolor.hpp
#include <termcolor/termcolor.hpp>

namespace artea {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class ArteaLogger {
public:
    static auto to_string(const LogLevel level) -> std::string {
        switch (level) {
            case LogLevel::DEBUG: return "DEBUG";
            case LogLevel::INFO:  return "INFO";
            case LogLevel::WARN:  return "WARN";
            case LogLevel::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }

    ArteaLogger(const std::string& logger_name, const LogLevel system_level = LogLevel::DEBUG) :
        _logger_name(logger_name), _system_level(system_level) {}

    ~ArteaLogger() = default;

    template<bool success_flag = false>
    auto log(const std::string& message, const LogLevel msg_level) -> void {
        if (static_cast<int>(msg_level) < static_cast<int>(_system_level)) {
            return; // Skip logging if message level is lower than current level
        }
        auto level_str = to_string(msg_level);

        switch (msg_level) {
            case LogLevel::DEBUG:
                std::cout << termcolor::grey;
                break;
            case LogLevel::INFO:
                if constexpr (success_flag) {
                    std::cout << termcolor::bold << termcolor::green;
                } else {
                    std::cout << termcolor::white;
                }
                break;
            case LogLevel::WARN:
                std::cout << termcolor::yellow;
                break;
            case LogLevel::ERROR:
                std::cout << termcolor::bold << termcolor::red;
                break;
        }

        std::cout << fmt::format("[{}] [{}] {}", level_str, _logger_name, message)
                  << termcolor::reset << std::endl;
    }

    __attribute__((always_inline))
    auto debug(const std::string& message) -> void {
        log(message, LogLevel::DEBUG);
    }

    __attribute__((always_inline))
    auto success(const std::string& message) -> void {
        log<true>(message, LogLevel::INFO);
    }

    __attribute__((always_inline))
    auto info(const std::string& message)  -> void {
        log(message, LogLevel::INFO);
    }

    __attribute__((always_inline))
    auto warn(const std::string& message)  -> void {
        log(message, LogLevel::WARN);
    }

    __attribute__((always_inline))
    auto error(const std::string& message) -> void {
        log(message, LogLevel::ERROR);
        // throw std::runtime_error(message);
    }

private:
    std::string _logger_name;
    LogLevel _system_level;

};  // class ArteaLogger

inline ArteaLogger logger("Artea");

}   // namespace artea