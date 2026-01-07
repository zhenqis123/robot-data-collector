#pragma once

#include <mutex>
#include <string>
#include <unordered_map>
#include <optional>
#include <vector>

#include <QObject>
#include <QImage>

#include "CameraInterface.h"

class ArucoTracker;
#include "CameraStats.h"

class QLabel;
class QVBoxLayout;
class QWidget;
class Logger;

class Preview : public QObject
{
public:
    explicit Preview(Logger &logger);

    void setStatusLabel(QLabel *label);
    void setDevicesLayout(QVBoxLayout *layout);
    void setShowDepthPreview(bool enabled);
    void registerCameraView(const std::string &cameraId, const std::string &type);
    void clearViews();
    void showFrame(const FrameData &frame);
    void updateStatus(const std::string &status);
    void updateCameraStats(const std::string &cameraId, const CameraFps &fps);
    void setArucoTracker(ArucoTracker *tracker);
    std::optional<FrameData> latestFrame(const std::string &cameraId);
    std::vector<std::string> cameraIds() const;

private:
    struct View
    {
        QLabel *colorLabel{nullptr};
        QLabel *depthLabel{nullptr};
        QLabel *dataLabel{nullptr}; // For text-only devices
        bool showDepth{false};
        bool isTextOnly{false};
        QWidget *container{nullptr};
        QLabel *infoLabel{nullptr};
        std::string infoBaseText;
        std::string lastArucoText;
    };

    std::unordered_map<std::string, View> _views;
    std::unordered_map<std::string, FrameData> _latestFrames;
    QLabel *_statusLabel{nullptr};
    QVBoxLayout *_devicesLayout{nullptr};
    Logger &_logger;
    mutable std::mutex _mutex;
    ArucoTracker *_arucoTracker{nullptr};
    bool _showDepthPreview{true};

    static QImage matToQImage(const cv::Mat &mat);
    static QImage depthToQImage(const cv::Mat &mat);
    void dispatchToLabel(QLabel *label, const QImage &image);
    void renderInfo(View &view);
};
