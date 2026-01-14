#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <thread>
#include <unistd.h>

#include <opencv2/highgui.hpp> 
#include <opencv2/imgproc.hpp>

#include "CameraInterface.h"
#include "Logger.h"
#include "VDGloveInterface.h"
#include "ViveInterface.h"

std::atomic<bool> g_running{true};

void signalHandler(int signum) {
    std::cout << "\nInterrupt signal (" << signum << ") received. Stopping..." << std::endl;
    g_running = false;
}

int test() {
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);

    Logger logger("logs"); 

    // // 1. 初始化相机
    // CameraConfig camConfig;
    // camConfig.type = "RealSense"; // 如果没有插 RealSense，改为 "RGB" 测试
    
    // auto camera = createCamera(camConfig, logger);
    // if (camera) {
    //     camera->initialize(camConfig);
    // } else {
    //     std::cerr << "Failed to create camera." << std::endl;
    //     return -1;
    // }

    // 2. 初始化手套
    VDGloveConfig VDGloveConfig;

    VDGloveConfig.local_port = 9999;
    VDGloveConfig.server_ip = "192.168.20.157";
    VDGloveConfig.server_port = 9998;
    VDGloveConfig.device_index = 0;
    VDGloveConfig.process_mano = true;


    auto glove = createGloveDevice("VDGlove", logger);

    if (!glove) {
        std::cerr << "Failed to create glove device. Check factory function." << std::endl;
        return -1;
    }

    if (!glove->initialize(VDGloveConfig)) {
        std::cerr << "Failed to initialize glove network." << std::endl;
        return -1;
    }


    ViveConfig viveConf;
    viveConf.port = 6666;
    auto vive = createViveDevice(logger);

    if (!vive->initialize(viveConf)) {
        return -1;
    }

    std::cout << "System initialized. Press ESC to exit." << std::endl;


    while (g_running) {
        // 获取图像
        // auto imgData = camera->captureFrame();

        // // 显示图像
        // if (!imgData.image.empty()) {
        //     cv::imshow("Camera", imgData.image);
        //     // waitKey(1) 返回按键的 ASCII 码，27 是 ESC
        //     if (cv::waitKey(1) == 27) break;
        // }

        auto handData = glove->captureGloveData();


        std::cout << "Left Hand: " << (handData.left_hand.detected ? "Detected" : "Not Detected") << std::endl;
        std::cout << "Right Hand: " << (handData.right_hand.detected ? "Detected" : "Not Detected") << std::endl;

        if (handData.right_hand.detected) {
             if (handData.right_hand.keypoints.size() > 8) {
                 auto tip = handData.right_hand.keypoints[8];
                 std::cout << "Right Hand Tip: " << tip.transpose() << std::endl;
             }
        }

        

        auto viveData = vive->captureViveData();

        // std::cout << "Python Timestamp: " << viveData.python_timestamp << std::endl;
        // std::cout << "Tracker Count: " << viveData.trackers.size() << std::endl;

        for (int i = 0; i < viveData.trackers.size(); ++i) {
            const auto& tracker = viveData.trackers[i];
            if (tracker.valid) {
                std::cout << "  Tracker " << i << ": Pos(" 
                          << tracker.position.x() << ", " 
                          << tracker.position.y() << ", " 
                          << tracker.position.z() << ")\n";
            } else {
                std::cout << "  Tracker " << i << ": Invalid\n";
            }
        }


        // if (!viveData.trackers.empty() && viveData.trackers[1].valid) {
        //     auto pos = viveData.trackers[1].position;
        //     // 计算延迟 (当前时间 - Python发送时间)
        //     // 注意: 这种计算需要两台机器时钟同步(PTP/NTP)，否则只能看相对抖动
        //     auto now_ts = std::chrono::duration<double>(viveData.host_timestamp.time_since_epoch()).count();
        //     double latency = now_ts - viveData.python_timestamp;

        //     printf("Tracker 1 Pos: [%.3f, %.3f, %.3f] | Latency: %.3f ms\n", 
        //            pos.x(), pos.y(), pos.z(), latency * 1000.0);
        // }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
    std::cout << "Exiting main loop..." << std::endl;

    if (vive){
        vive->close();
    }
    // camera->close();
    if (glove){
        glove->close();
    }
    
    std::cout << "Exiting main safely." << std::endl;
    _exit(0);
}
