#include <gtest/gtest.h>

#include <filesystem>

#include <QApplication>
#include <QLabel>
#include <QVBoxLayout>
#include <QWidget>

#include <opencv2/core.hpp>

#include "Logger.h"
#include "Preview.h"

static QApplication &ensureQtApp()
{
    static int argc = 0;
    static char **argv = nullptr;
    static QApplication app(argc, argv);
    return app;
}

static std::string previewDir()
{
    auto dir = (std::filesystem::temp_directory_path() / "preview_tests").string();
    std::filesystem::create_directories(dir);
    return dir;
}

TEST(PreviewTest, UpdatesLabelPixmap)
{
    auto &app = ensureQtApp();
    (void)app;
    Logger logger(previewDir());
    Preview preview(logger);

    QWidget container;
    QVBoxLayout layout(&container);
    preview.setDevicesLayout(&layout);
    preview.registerCameraView("Preview", "RGB");

    auto labels = container.findChildren<QLabel *>();
    ASSERT_FALSE(labels.empty());
    QLabel *display = labels.front();
    display->resize(100, 100);

    FrameData frame;
    frame.cameraId = "Preview";
    frame.timestamp = std::chrono::system_clock::now();
    frame.image = std::make_shared<cv::Mat>(cv::Mat::ones(20, 20, CV_8UC3));
    frame.image = cv::Mat::ones(20, 20, CV_8UC3);

    preview.showFrame(frame);
    QApplication::processEvents();

    ASSERT_NE(display->pixmap(), nullptr);
    EXPECT_FALSE(display->pixmap()->isNull());
}
