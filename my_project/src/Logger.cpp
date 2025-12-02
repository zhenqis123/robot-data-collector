#include "Logger.h"

#include <filesystem>
#include <iomanip>
#include <sstream>

Logger::Logger(const std::string &logDirectory)
{
    std::filesystem::create_directories(logDirectory);
    _logFilePath = (std::filesystem::path(logDirectory) / "app.log").string();
    _stream.open(_logFilePath, std::ios::app);
    log(LogLevel::Info, "Logger initialized");
}

Logger::~Logger()
{
    log(LogLevel::Info, "Logger shutting down");
    _stream.close();
}

void Logger::log(LogLevel level, const std::string &message)
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (!_stream.is_open())
        return;
    _stream << "[" << timestamp() << "]"
            << "[" << levelToString(level) << "] " << message << std::endl;
}

std::string Logger::levelToString(LogLevel level)
{
    switch (level)
    {
    case LogLevel::Info:
        return "INFO";
    case LogLevel::Warning:
        return "WARN";
    case LogLevel::Error:
        return "ERROR";
    default:
        return "UNKNOWN";
    }
}

std::string Logger::timestamp()
{
    const auto now = std::chrono::system_clock::now();
    const auto time = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &time);
#else
    localtime_r(&time, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%d %H:%M:%S");
    return oss.str();
}
