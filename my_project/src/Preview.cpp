#include "Preview.h"

#include <iomanip>
#include <memory>
#include <sstream>

#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLayoutItem>
#include <QMetaObject>
#include <QPixmap>
#include <QVBoxLayout>
#include <opencv2/aruco.hpp>
#include <opencv2/imgproc.hpp>

#include "ArucoTracker.h"
#include "Logger.h"

namespace
{
std::unique_ptr<QLabel> makeDisplayLabel()
{
    auto label = std::make_unique<QLabel>();
    label->setMinimumSize(640, 360);
    label->setSizePolicy(QSizePolicy::Ignored, QSizePolicy::Ignored);
    label->setStyleSheet(
        "background-color: #222; color: #fff; border: 1px solid #444;"
    );
    label->setAlignment(Qt::AlignCenter);
    label->setText("Waiting...");
    return label;
}

std::unique_ptr<QLabel> makeTextDataLabel()
{
    auto label = std::make_unique<QLabel>();
    label->setMinimumSize(320, 40);
    label->setStyleSheet(
        "background-color: #111; color: #0f0; border: 1px solid #333; font-family: Monospace;"
    );
    label->setAlignment(Qt::AlignLeft | Qt::AlignVCenter);
    label->setText("Waiting for data...");
    return label;
}
} // namespace

Preview::Preview(Logger &logger)
    : QObject(nullptr), _logger(logger)
{
}

void Preview::setStatusLabel(QLabel *label)
{
    _statusLabel = label;
}

void Preview::setDevicesLayout(QVBoxLayout *layout)
{
    _devicesLayout = layout;
}

void Preview::setShowDepthPreview(bool enabled)
{
    _showDepthPreview = enabled;
}

void Preview::registerCameraView(const std::string &cameraId, const std::string &type)
{
    if (!_devicesLayout || _views.count(cameraId))
        return;

    auto group = new QGroupBox(QString::fromStdString(cameraId));
    auto boxLayout = new QVBoxLayout(group);

    View view;
    view.container = group;
    // Update: Treat TacGlove and Manus as text-only views for compact display
    view.isTextOnly = (type == "VDGlove" || type == "Vive" || type == "ViveTracker" || 
                       type == "TacGlove" || type == "Manus" || type == "ManusGloves");

    if (view.isTextOnly)
    {
        auto dataLabel = makeTextDataLabel();
        view.dataLabel = dataLabel.get();
        boxLayout->addWidget(dataLabel.release());
    }
    else
    {
        auto streamLayout = new QHBoxLayout();

        auto colorLabel = makeDisplayLabel();
        auto colorContainer = new QWidget();
        colorContainer->setLayout(new QVBoxLayout());
        colorContainer->layout()->setContentsMargins(0, 0, 0, 0);
        colorContainer->layout()->addWidget(colorLabel.get());
        streamLayout->addWidget(colorContainer, 1);

        QLabel *depthLabelRaw = nullptr;
        bool showDepth = (type == "RealSense") && _showDepthPreview;
        if (showDepth)
        {
            auto depthLabel = makeDisplayLabel();
            depthLabelRaw = depthLabel.get();
            depthLabel->setText("Depth");
            auto depthContainer = new QWidget();
            depthContainer->setLayout(new QVBoxLayout());
            depthContainer->layout()->setContentsMargins(0, 0, 0, 0);
            depthContainer->layout()->addWidget(depthLabel.release());
            streamLayout->addWidget(depthContainer, 1);
        }

        boxLayout->addLayout(streamLayout);
        view.colorLabel = colorLabel.release();
        view.depthLabel = depthLabelRaw;
        view.showDepth = showDepth;
    }

    const bool showDepthFps = view.showDepth && !view.isTextOnly;
    const QString baseText = showDepthFps
                                 ? "Cap R/D: 0.0/0.0 | Disp R/D: 0.0/0.0 | Write R/D: 0.0/0.0"
                                 : "Capture: 0.0 fps | Display: 0.0 fps | Write: 0.0 fps";
    auto infoLabel = new QLabel(baseText);
    infoLabel->setStyleSheet(
        "color: #f0f0f0; background-color: rgba(0,0,0,0.5); font-size: 12px; padding: 2px 6px;"
    );
    infoLabel->setAlignment(Qt::AlignLeft);
    boxLayout->addWidget(infoLabel, 0, Qt::AlignBottom | Qt::AlignLeft);
    _devicesLayout->addWidget(group);

    view.infoLabel = infoLabel;
    view.infoBaseText = infoLabel->text().toStdString();
    _views.emplace(cameraId, view);
}

void Preview::setArucoTracker(ArucoTracker *tracker)
{
    _arucoTracker = tracker;
}

std::optional<FrameData> Preview::latestFrame(const std::string &cameraId)
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _latestFrames.find(cameraId);
    if (it != _latestFrames.end())
        return it->second;
    return std::nullopt;
}

std::vector<std::string> Preview::cameraIds() const
{
    std::lock_guard<std::mutex> lock(_mutex);
    std::vector<std::string> ids;
    ids.reserve(_views.size());
    for (const auto &p : _views)
        ids.push_back(p.first);
    return ids;
}

void Preview::showFrame(const FrameData &frame)
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _views.find(frame.cameraId);
    if (it == _views.end())
        return;

    auto &view = it->second;
    std::string arucoInfo;
    const std::string detectorLabel = _arucoTracker ? _arucoTracker->detectorName() : "Markers";

    if (view.isTextOnly && view.dataLabel)
    {
        if (frame.gloveData)
        {
            std::ostringstream ss;
            ss << "Left: " << (frame.gloveData->left_hand.detected ? "Detected" : "None")
               << " | Right: " << (frame.gloveData->right_hand.detected ? "Detected" : "None")
               << " | TS: " << frame.deviceTimestampMs;
            QMetaObject::invokeMethod(view.dataLabel, [lbl = view.dataLabel, text = ss.str()]() {
                lbl->setText(QString::fromStdString(text));
            });
        }
        else if (frame.viveData)
        {
            std::ostringstream ss;
            ss << "Trackers: ";
            for (size_t i = 0; i < frame.viveData->trackers.size(); ++i)
            {
                ss << "#" << i << (frame.viveData->trackers[i].valid ? "[OK]" : "[NO]") << " ";
            }
            QMetaObject::invokeMethod(view.dataLabel, [lbl = view.dataLabel, text = ss.str()]() {
                lbl->setText(QString::fromStdString(text));
            });
        }
        else if (frame.tacGloveData)
        {
            std::ostringstream ss;
            ss << "Left: " << (frame.tacGloveData->leftFrame.isMissing ? "None" : "OK")
               << " | Right: " << (frame.tacGloveData->rightFrame.isMissing ? "None" : "OK")
               << " | TS: " << frame.deviceTimestampMs;
            QMetaObject::invokeMethod(view.dataLabel, [lbl = view.dataLabel, text = ss.str()]() {
                lbl->setText(QString::fromStdString(text));
            });
        }
        else if (frame.manusData)
        {
            std::ostringstream ss;
            bool leftOk = !frame.manusData->left_hand.joints.empty();
            bool rightOk = !frame.manusData->right_hand.joints.empty();
            ss << "Left: " << (leftOk ? "OK" : "None")
               << " | Right: " << (rightOk ? "OK" : "None");
            QMetaObject::invokeMethod(view.dataLabel, [lbl = view.dataLabel, text = ss.str()]() {
                lbl->setText(QString::fromStdString(text));
            });
        }
        _latestFrames[frame.cameraId] = frame;
        return;
    }

    if (view.colorLabel && frame.hasImage())
    {
        cv::Mat converted;
        const cv::Mat *source = &frame.imageRef();
        if (frame.colorFormat == "YUYV")
        {
            cv::cvtColor(frame.imageRef(), converted, cv::COLOR_YUV2BGR_YUY2);
            source = &converted;
        }
        cv::Mat overlayMat;
        if (_arucoTracker)
        {
            auto detections = _arucoTracker->getLatestDetections(frame.cameraId);
            if (!detections.empty())
            {
                overlayMat = source->clone();
                std::vector<int> ids;
                std::vector<std::vector<cv::Point2f>> corners;
                for (const auto &det : detections)
                {
                    ids.push_back(det.markerId);
                    corners.push_back(det.corners);
                }

                for (size_t i = 0; i < corners.size(); ++i)
                {
                    const auto &c = corners[i];
                    if (c.size() == 4)
                    {
                        for (int k = 0; k < 4; ++k)
                        {
                            cv::line(overlayMat, c[k], c[(k + 1) % 4], cv::Scalar(0, 255, 0), 3);
                        }
                        cv::putText(overlayMat, std::to_string(ids[i]), c[0] + cv::Point2f(0, -6),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3);
                        cv::putText(overlayMat, std::to_string(ids[i]), c[0] + cv::Point2f(0, -6),
                                    cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 255, 0), 2);
                    }
                }
                cv::aruco::drawDetectedMarkers(overlayMat, corners, ids, cv::Scalar(0, 255, 0));
                source = &overlayMat;

                std::ostringstream ss;
                ss << detections.size() << " ids: ";
                for (size_t i = 0; i < detections.size(); ++i)
                {
                    ss << detections[i].markerId;
                    if (i + 1 < detections.size())
                        ss << ",";
                }
                arucoInfo = ss.str();
            }
        }
        auto image = matToQImage(*source);
        dispatchToLabel(view.colorLabel, image);
    }

    if (view.showDepth && view.depthLabel && frame.hasDepth())
    {
        auto image = depthToQImage(frame.depthRef());
        dispatchToLabel(view.depthLabel, image);
    }

    _latestFrames[frame.cameraId] = frame;

    if (!arucoInfo.empty() && view.infoLabel)
    {
        view.lastArucoText = detectorLabel + " " + arucoInfo;
        renderInfo(view);
    }
    else
    {
        updateStatus("Displaying " + frame.cameraId);
    }
}

void Preview::clearViews()
{
    std::lock_guard<std::mutex> lock(_mutex);
    if (_devicesLayout)
    {
        QLayoutItem *child;
        while ((child = _devicesLayout->takeAt(0)) != nullptr)
        {
            if (auto widget = child->widget())
                widget->deleteLater();
            delete child;
        }
    }
    _views.clear();
    _latestFrames.clear();
}

void Preview::updateStatus(const std::string &status)
{
    if (!_statusLabel)
        return;
    auto label = _statusLabel;
    QMetaObject::invokeMethod(label, [label, status]() {
        label->setText(QString::fromStdString(status));
    });
}

void Preview::updateCameraStats(const std::string &cameraId, const CameraFps &fps)
{
    std::lock_guard<std::mutex> lock(_mutex);
    auto it = _views.find(cameraId);
    if (it == _views.end() || !it->second.infoLabel)
        return;
    auto &view = it->second;
    if (!view.isTextOnly && view.showDepth)
    {
        view.infoBaseText =
            (QString("Cap R/D: %1/%2 | Disp R/D: %3/%4 | Write R/D: %5/%6")
                 .arg(fps.captureColor, 0, 'f', 1)
                 .arg(fps.captureDepth, 0, 'f', 1)
                 .arg(fps.displayColor, 0, 'f', 1)
                 .arg(fps.displayDepth, 0, 'f', 1)
                 .arg(fps.writeColor, 0, 'f', 1)
                 .arg(fps.writeDepth, 0, 'f', 1))
                .toStdString();
    }
    else
    {
        view.infoBaseText =
            (QString("Capture: %1 fps | Display: %2 fps | Write: %3 fps")
                 .arg(fps.captureColor, 0, 'f', 1)
                 .arg(fps.displayColor, 0, 'f', 1)
                 .arg(fps.writeColor, 0, 'f', 1))
                .toStdString();
    }
    renderInfo(view);
}

QImage Preview::matToQImage(const cv::Mat &mat)
{
    cv::Mat rgb;
    if (mat.channels() == 3)
        cv::cvtColor(mat, rgb, cv::COLOR_BGR2RGB);
    else
        cv::cvtColor(mat, rgb, cv::COLOR_GRAY2RGB);
    return QImage(rgb.data, rgb.cols, rgb.rows, static_cast<int>(rgb.step), QImage::Format_RGB888).copy();
}

QImage Preview::depthToQImage(const cv::Mat &mat)
{
    if (mat.empty())
        return {};
    cv::Mat depth8u;
    const double scale = 255.0 / 8000.0;
    mat.convertTo(depth8u, CV_8U, scale);
    cv::Mat colored;
    cv::applyColorMap(depth8u, colored, cv::COLORMAP_JET);
    return matToQImage(colored);
}

void Preview::renderInfo(View &view)
{
    if (!view.infoLabel)
        return;
    std::string combined = view.infoBaseText;
    if (!view.lastArucoText.empty())
        combined += " | " + view.lastArucoText;
    auto label = view.infoLabel;
    QMetaObject::invokeMethod(label, [label, combined]() {
        if (label->text().toStdString() == combined)
            return;
        label->setText(QString::fromStdString(combined));
    }, Qt::QueuedConnection);
}

void Preview::dispatchToLabel(QLabel *label, const QImage &image)
{
    if (!label)
        return;
    QMetaObject::invokeMethod(label, [label, image]() {
        const QSize target = label->size().isEmpty() ? label->minimumSize() : label->size();
        label->setPixmap(QPixmap::fromImage(image).scaled(target, Qt::KeepAspectRatio, Qt::SmoothTransformation));
    });
}
