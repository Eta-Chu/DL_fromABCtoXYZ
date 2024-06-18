#pragma once
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>

#define LOGI LOG(INFO)
#define LOGW LOG(WARNING)
#define LOGE LOG(ERROR)
#define LOGF LOG(FATAL)

inline void InitGlog() {
  google::InitGoogleLogging("FCNN");
  FLAGS_log_dir = "./logs";                 // 指定日志文件目录
  FLAGS_logtostderr = false;                // 禁止将日志输出到标准错误输出
  FLAGS_alsologtostderr = false;            // 不同时将日志输出到标准错误输出
  FLAGS_logbuflevel = -1;                   // 所有级别的日志都直接写入文件
  mkdir(FLAGS_log_dir.c_str(), 0777);       // 创建log文件夹
}

inline void ShutdownGlog() {
  // 终止Google日志库
  google::ShutdownGoogleLogging();
}