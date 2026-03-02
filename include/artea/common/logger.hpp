/*
 * @FilePath: /Artea/include/artea/common/logger.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2026-02-02 21:04:12
 * @Date: 2025-10-23 13:26:15
 * @Description: A logger that supports colorful printing via termcolor.
 */

#pragma once

#include <iostream>
#include <string>
#include <stdexcept>
#include <fmt/format.h>
// Reference: https://github.com/ikaln/termcolor/blob/master/include/termcolor/termcolor.hpp
#include <termcolor/termcolor.hpp>

namespace artea {

enum class LogLevelT {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class ArteaLogger {
public:
    static auto to_string(const LogLevelT level) -> std::string {
        switch (level) {
            case LogLevelT::DEBUG: return "DEBUG";
            case LogLevelT::INFO:  return "INFO";
            case LogLevelT::WARN:  return "WARN";
            case LogLevelT::ERROR: return "ERROR";
        }
        return "UNKNOWN";
    }

    ArteaLogger(const std::string& logger_name, const LogLevelT system_level = LogLevelT::DEBUG) :
        _logger_name(logger_name), _system_level(system_level) {}

    ~ArteaLogger() = default;

    template<bool success_flag = false>
    auto log(const std::string& message, const LogLevelT msg_level) -> void {
        if (static_cast<int>(msg_level) < static_cast<int>(_system_level)) {
            return; // Skip logging if message level is lower than current level
        }
        auto level_str = to_string(msg_level);

        switch (msg_level) {
            case LogLevelT::DEBUG:
                std::cout << termcolor::grey;
                break;
            case LogLevelT::INFO:
                if constexpr (success_flag) {
                    std::cout << termcolor::bold << termcolor::green;
                } else {
                    std::cout << termcolor::white;
                }
                break;
            case LogLevelT::WARN:
                std::cout << termcolor::yellow;
                break;
            case LogLevelT::ERROR:
                std::cout << termcolor::bold << termcolor::red;
                break;
        }

        std::cout << fmt::format("[{}] [{}] {}", level_str, _logger_name, message)
                  << termcolor::reset << std::endl;
    }

    __attribute__((always_inline))
    auto debug(const std::string& message) -> void {
        log(message, LogLevelT::DEBUG);
    }

    __attribute__((always_inline))
    auto success(const std::string& message) -> void {
        log<true>(message, LogLevelT::INFO);
    }

    __attribute__((always_inline))
    auto info(const std::string& message)  -> void {
        log(message, LogLevelT::INFO);
    }

    __attribute__((always_inline))
    auto warn(const std::string& message)  -> void {
        log(message, LogLevelT::WARN);
    }

    __attribute__((always_inline))
    auto error(const std::string& message) -> void {
        log(message, LogLevelT::ERROR);
        throw std::runtime_error(message);
    }

    /**
     * @brief Print a message without header prefix (for custom formatting)
     * @param message The message to print
     */
    auto print(const std::string& message) -> void {
        std::cout << message << termcolor::reset << std::endl;
    }

    /**
     * @brief Print a colored message without header prefix
     * @param message The message to print
     * @param color The termcolor manipulator (pass as function pointer or lambda)
     */
    auto print(const std::string& message, std::ostream& (*color)(std::ostream&)) -> void {
        std::cout << color << message << termcolor::reset << std::endl;
    }

    /**
     * @brief Print a message without header prefix and without newline
     * @param message The message to print
     */
    auto print_inline(const std::string& message) -> void {
        std::cout << message << termcolor::reset;
    }

    /**
     * @brief Print a colored message without header prefix and without newline
     * @param message The message to print
     * @param color The termcolor manipulator (pass as function pointer or lambda)
     */
    auto print_inline(const std::string& message, std::ostream& (*color)(std::ostream&)) -> void {
        std::cout << color << message << termcolor::reset;
    }

private:
    std::string _logger_name;
    LogLevelT _system_level;

};  // class ArteaLogger

inline ArteaLogger logger("Artea", LogLevelT::INFO);

}   // namespace artea