#pragma once

#include <fstream>
#include <mutex>
#include <string>
#include <chrono>
#include <cstdio>

enum class LogLevel
{
    Info,
    Warning,
    Error
};

class Logger
{
public:
    explicit Logger(const std::string &logDirectory);
    ~Logger();

    void log(LogLevel level, const std::string &message);

    template <typename... Args>
    void info(const std::string &message, Args &&...args)
    {
        log(LogLevel::Info, format(message, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void warn(const std::string &message, Args &&...args)
    {
        log(LogLevel::Warning, format(message, std::forward<Args>(args)...));
    }

    template <typename... Args>
    void error(const std::string &message, Args &&...args)
    {
        log(LogLevel::Error, format(message, std::forward<Args>(args)...));
    }

    std::string logFilePath() const { return _logFilePath; }

private:
    std::string _logFilePath;
    std::ofstream _stream;
    mutable std::mutex _mutex;

    static std::string levelToString(LogLevel level);
    static std::string timestamp();

    template <typename... Args>
    static std::string format(const std::string &message, Args &&...args)
    {
        if constexpr (sizeof...(Args) == 0)
        {
            return message;
        }
        else
        {
            int size = std::snprintf(nullptr, 0, message.c_str(), args...) + 1;
            std::string buffer(size, '\0');
            std::snprintf(buffer.data(), size, message.c_str(), args...);
            buffer.pop_back();
            return buffer;
        }
    }
};
