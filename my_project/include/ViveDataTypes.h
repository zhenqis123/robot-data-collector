#pragma once
#include <vector>
#include <Eigen/Dense>
#include <chrono>

// 对应 Python 的 3 个 Tracker
constexpr int NUM_TRACKERS = 3;

// 单个 Tracker 的处理后数据
struct ViveTrackerPose {
    bool valid = false;           // 如果全为0则认为无效
    Eigen::Matrix3f rotation;     // 旋转矩阵
    Eigen::Vector3f position;     // 位移向量
};

struct ViveFrameData {
    double python_timestamp;      // 发送端的时间戳
    std::chrono::system_clock::time_point host_timestamp; // 接收端时间
    std::vector<ViveTrackerPose> trackers; // 大小为 3
};

// 原始 UDP 包结构 (必须与 Python struct.pack 严格对应)
// 使用 #pragma pack 确保内存紧凑，无额外填充
#pragma pack(push, 1)
struct ViveRawPacket {
    double timestamp;             // 8 bytes
    float data[NUM_TRACKERS * 12]; // 3 * 12 * 4 = 144 bytes
};                                // Total = 152 bytes
#pragma pack(pop)

struct ViveConfig : public CameraConfig {
    int port = 6666;
};