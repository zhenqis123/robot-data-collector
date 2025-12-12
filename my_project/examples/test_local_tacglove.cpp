/**
 * @file test_local_tacglove.cpp
 * @brief LocalTacGlove IPC 通信测试程序
 *
 * 用法:
 *   1. 先运行 Python 发送端: python tac_glove_py/test_ipc.py --mode continuous
 *   2. 再运行此程序: ./build/test_local_tacglove
 */

#include <chrono>
#include <iostream>
#include <thread>

#include "TacGlove.h"
#include "Logger.h"

int main(int argc, char *argv[])
{
    std::cout << "======================================" << std::endl;
    std::cout << "LocalTacGlove IPC Test" << std::endl;
    std::cout << "======================================" << std::endl;
    std::cout << std::endl;
    std::cout << "Waiting for Python sender..." << std::endl;
    std::cout << "Run: python tac_glove_py/test_ipc.py --mode continuous" << std::endl;
    std::cout << std::endl;

    // 创建 Logger
    Logger logger("/tmp/tacglove_test");

    // 创建 LocalTacGlove
    auto glove = createTacGlove("Local", logger);
    if (!glove)
    {
        std::cerr << "[ERROR] Failed to create LocalTacGlove" << std::endl;
        return 1;
    }

    // 初始化
    if (!glove->initialize("LocalTacGlove#Test", TacGloveMode::Both))
    {
        std::cerr << "[ERROR] Failed to initialize LocalTacGlove" << std::endl;
        return 1;
    }

    std::cout << "[OK] LocalTacGlove initialized" << std::endl;

    auto *localGlove = dynamic_cast<LocalTacGlove *>(glove.get());

    // 主循环：读取并显示数据
    int frameCount = 0;
    int validFrames = 0;
    int missingFrames = 0;

    std::cout << "\nReading frames (Press Ctrl+C to stop)...\n" << std::endl;

    while (frameCount < 100)
    {
        auto now = std::chrono::system_clock::now();
        auto deviceTs = std::chrono::duration_cast<std::chrono::milliseconds>(
                            now.time_since_epoch())
                            .count();

        TacGloveDualFrameData frame = glove->captureFrame(now, deviceTs);

        bool leftValid = !frame.leftFrame.isMissing;
        bool rightValid = !frame.rightFrame.isMissing;

        if (leftValid || rightValid)
        {
            ++validFrames;

            std::cout << "Frame " << frameCount << ": ";
            if (leftValid)
            {
                std::cout << "L[" << frame.leftFrame.data[0] << ", "
                          << frame.leftFrame.data[1] << ", ...]";
            }
            else
            {
                std::cout << "L[missing]";
            }
            std::cout << " ";
            if (rightValid)
            {
                std::cout << "R[" << frame.rightFrame.data[0] << ", "
                          << frame.rightFrame.data[1] << ", ...]";
            }
            else
            {
                std::cout << "R[missing]";
            }
            std::cout << std::endl;
        }
        else
        {
            ++missingFrames;
            if (frameCount % 10 == 0)
            {
                std::cout << "Frame " << frameCount << ": [waiting for data...]" << std::endl;
            }
        }

        // 检查连接状态
        if (localGlove && frameCount % 20 == 0)
        {
            bool connected = localGlove->isConnected();
            size_t leftQueued = localGlove->queuedFrames(true);
            size_t rightQueued = localGlove->queuedFrames(false);
            std::cout << "  [Status] Connected: " << (connected ? "yes" : "no")
                      << ", Queued: L=" << leftQueued << " R=" << rightQueued << std::endl;
        }

        ++frameCount;
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "\n======================================" << std::endl;
    std::cout << "Test Complete" << std::endl;
    std::cout << "  Total frames: " << frameCount << std::endl;
    std::cout << "  Valid frames: " << validFrames << std::endl;
    std::cout << "  Missing frames: " << missingFrames << std::endl;
    std::cout << "======================================" << std::endl;

    glove->close();
    return 0;
}
