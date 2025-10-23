/*
 * @FilePath: /Artea/include/artea/logger.hpp
 * @Author: Chandler (Weitang Ye) <weitang.ye@ntu.edu.sg>
 * @LastEditTime: 2025-10-23 19:38:37
 * @Date: 2025-10-23 13:26:15
 * @Description: 
 */

#pragma once

#include <iostream>
#include <string>
#include <format>   // C++20

#include <artea/types.hpp>

namespace artea {

enum class LogLevel {
    DEBUG,
    INFO,
    WARN,
    ERROR
};

class ArteaLogger {
public:
    static constexpr auto to_string(const LogLevel level) -> std::string {
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

    auto log(const std::string& message, const LogLevel msg_level) -> void {
        if (static_cast<int>(msg_level) < static_cast<int>(_system_level)) {
            return; // Skip logging if message level is lower than current level
        }
        auto level_str = to_string(msg_level);
        // TODO: support colorful logging
        std::cout << std::format("[{}] [{}] {}", level_str, _logger_name, message) << std::endl;
    }

    __attribute__((always_inline))
    auto debug(const std::string& message) -> void { 
        log(message, LogLevel::DEBUG); 
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
    }

private:
    std::string _logger_name;
    LogLevel _system_level;

};  // class ArteaLogger


}   // namespace artea