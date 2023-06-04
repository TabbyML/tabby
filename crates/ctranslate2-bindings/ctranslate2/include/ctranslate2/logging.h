#pragma once

namespace ctranslate2 {

  enum class LogLevel {
    Off = -3,
    Critical = -2,
    Error = -1,
    Warning = 0,
    Info = 1,
    Debug = 2,
    Trace = 3,
  };

  void init_logger();
  void set_log_level(const LogLevel level);
  LogLevel get_log_level();

}
