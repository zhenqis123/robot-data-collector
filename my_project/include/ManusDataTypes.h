#pragma once
#include <vector>
#include <Eigen/Dense>
#include <chrono>

#include "CameraInterface.h"

constexpr int MANUS_NUM_HANDS = 2;
constexpr int MANUS_JOINTS_PER_HAND = 25;

struct ManusJointPose {
    Eigen::Vector3f position;
    Eigen::Quaternionf orientation;
};

struct ManusHandData {
    std::vector<ManusJointPose> joints;
};

struct ManusFrameData {
    double python_timestamp;      // Sender timestamp
    std::chrono::system_clock::time_point host_timestamp; // Receiver timestamp
    ManusHandData left_hand;
    ManusHandData right_hand;
};

// Raw packet for UDP (must match Python struct.pack)
// #pragma pack(push, 1)
struct ManusRawPacket {
    double timestamp;             // 8 bytes
    // Left hand
    float left_fingers[MANUS_JOINTS_PER_HAND][3];     // 25 * 3 * 4 = 300 bytes
    float left_orientations[MANUS_JOINTS_PER_HAND][4]; // 25 * 4 * 4 = 400 bytes
    // Right hand
    float right_fingers[MANUS_JOINTS_PER_HAND][3];    // 300 bytes
    float right_orientations[MANUS_JOINTS_PER_HAND][4];// 400 bytes
};                                // Total = 8 + 700 + 700 = 1408 bytes
// #pragma pack(pop)

struct ManusConfig : public CameraConfig {
    int port = 6667; // Default port for Manus
};
