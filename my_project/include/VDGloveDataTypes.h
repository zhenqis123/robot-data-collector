#pragma once

#include <array>
#include <chrono>
#include <vector>

#include <Eigen/Dense>

#include "ConfigManager.h"

// 定义常量 (来自 vdmocapsdk_nodelist.py)
constexpr int LENGTH_BODY = 23;
constexpr int LENGTH_HAND = 20;

// 为了防止编译器字节对齐差异，强制 1 字节对齐（视 SDK 编译选项而定，通常 ctypes 默认对齐，但在网络传输结构体中常紧凑）
// 这里假设 SDK 是标准对齐，若数据错乱可尝试 #pragma pack(1)
struct MocapData {
    bool isUpdate;
    unsigned int frameIndex;
    int frequency;
    int nsResult;
    
    // Body (23 nodes)
    unsigned int sensorState_body[LENGTH_BODY];
    float position_body[LENGTH_BODY * 3];       // c_float * 3 * LENGTH_BODY
    float quaternion_body[LENGTH_BODY * 4];     // c_float * 4 * LENGTH_BODY
    float gyr_body[LENGTH_BODY * 3];
    float acc_body[LENGTH_BODY * 3];
    float velocity_body[LENGTH_BODY * 3];

    // Right Hand (20 nodes)
    unsigned int sensorState_rHand[LENGTH_HAND];
    float position_rHand[LENGTH_HAND * 3];
    float quaternion_rHand[LENGTH_HAND * 4];
    float gyr_rHand[LENGTH_HAND * 3];
    float acc_rHand[LENGTH_HAND * 3];
    float velocity_rHand[LENGTH_HAND * 3];

    // Left Hand (20 nodes)
    unsigned int sensorState_lHand[LENGTH_HAND];
    float position_lHand[LENGTH_HAND * 3];
    float quaternion_lHand[LENGTH_HAND * 4];
    float gyr_lHand[LENGTH_HAND * 3];
    float acc_lHand[LENGTH_HAND * 3];
    float velocity_lHand[LENGTH_HAND * 3];

    // Face / Misc
    bool isUseFaceBlendShapesARKit;
    bool isUseFaceBlendShapesAudio;
    float faceBlendShapesARKit[52];
    float faceBlendShapesAudio[26];
    float localQuat_RightEyeball[4];
    float localQuat_LeftEyeball[4];
    
    int gestureResultL;
    int gestureResultR;
};

struct HandData {
    bool detected = false;
    std::vector<Eigen::Vector3f> keypoints; // 21 points (MANO)
    Eigen::Matrix3f wrist_rotation;
    Eigen::Vector4f wrist_quaternion; 
    int gesture_id = 0;
};

struct VDGloveFrameData {
    HandData left_hand;
    HandData right_hand;
    std::chrono::system_clock::time_point timestamp;
    int64_t deviceTimestampMs;
};

struct VDGloveConfig : public CameraConfig {
    std::string server_ip = "192.168.20.240"; // 对应 UDP 发送请求的目标
    int server_port = 9998;
    int local_port = 9999;                    // 对应 UdpOpen 的端口
    int device_index = 0;                     // SDK 实例索引
    bool process_mano = true;
};
