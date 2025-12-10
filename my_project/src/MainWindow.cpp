#include "MainWindow.h"

#include <QComboBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGridLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QLineEdit>
#include <QPushButton>
#include <QScrollArea>
#include <QVBoxLayout>
#include <QWidget>
#include <QSize>
#include <QKeyEvent>
#include <QMessageBox>
#include <QTextEdit>
#include <QThread>
#include <QApplication>
#include <QNetworkAccessManager>
#include <QNetworkReply>
#include <QNetworkRequest>
#include <QEventLoop>
#include <QUrl>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonArray>
#include <QMediaPlayer>
#include <QVideoWidget>
#include <QUrl>
#include <QFileInfo>
#include <QFile>

#include <algorithm>
#include <set>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <librealsense2/rs.hpp>
#include <nlohmann/json.hpp>
#include <cstdlib>
#include <opencv2/imgcodecs.hpp>

using json = nlohmann::json;

namespace
{
std::unique_ptr<QLabel> makeDisplayLabel()
{
    auto label = std::make_unique<QLabel>();
    label->setMinimumSize(320, 180);
    label->setStyleSheet("background-color: #222; color: #fff; border: 1px solid #444;");
    label->setAlignment(Qt::AlignCenter);
    label->setText("Waiting...");
    return label;
}

} // namespace

MainWindow::MainWindow()
    : _logger(APP_LOG_DIR),
      _configManager(_logger),
      _preview(_logger),
      _defaultCaptureRoot(std::string(APP_LOG_DIR) + "/captures"),
      _storage(_defaultCaptureRoot, _logger),
      _taskLoader(_logger),
      _audioPlayer(_logger),
      _audioConfig(_configManager.getAudioPromptsConfig())
{
    std::filesystem::create_directories(APP_LOG_DIR);
    if (!_configManager.load(APP_CONFIG_PATH))
    {
        _logger.error("Failed to load configuration file");
    }
    _audioConfig = _configManager.getAudioPromptsConfig();
    _arucoTracker = std::make_unique<ArucoTracker>(_configManager.getArucoTargets());

    setupUi();
    _captureNameEdit->setPlaceholderText("例如：Session01");
    _subjectEdit->setPlaceholderText("受试者信息");
    _savePathEdit->setText(QString::fromStdString(_defaultCaptureRoot));
    _preview.setStatusLabel(_statusLabel);
    _preview.setDevicesLayout(_devicesLayout);
    _preview.setArucoTracker(_arucoTracker.get());

    loadTaskTemplates();
    applyVlmConfigUi();
    onModeChanged(_modeSelect->currentIndex());
    // Configure audio prompts
    applyAudioConfigForLanguage();
    updateVlmPromptMetadata();
    refreshCameraSettingsList();
    connectSignals();
    onOpenCameras();
    updateControls();
}

void MainWindow::setupUi()
{
    auto *central = new QWidget(this);
    auto *rootLayout = new QHBoxLayout(central);
    rootLayout->setSpacing(16);

    _controlPanel = new QWidget(central);
    _controlPanel->setMinimumWidth(400);
    auto *controlsLayout = new QVBoxLayout(_controlPanel);
    controlsLayout->setSpacing(12);
    controlsLayout->addWidget(createMetadataGroup());
    controlsLayout->addWidget(createTaskSelectionGroup());
    controlsLayout->addWidget(createVlmGroup());
    controlsLayout->addWidget(createCameraControlGroup());
    controlsLayout->addWidget(createCaptureControlGroup());
    controlsLayout->addWidget(createPromptControlGroup());
    controlsLayout->addWidget(createCameraSettingsGroup());
    controlsLayout->addWidget(createStatusGroup());
    controlsLayout->addStretch();

    auto *controlsScroll = new QScrollArea(central);
    controlsScroll->setWidgetResizable(true);
    controlsScroll->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
    controlsScroll->setWidget(_controlPanel);
    controlsScroll->setMinimumWidth(_controlPanel->minimumWidth() + 24);
    rootLayout->addWidget(controlsScroll, 0);

    _scrollArea = new QScrollArea(this);
    _scrollArea->setWidgetResizable(true);
    auto *devicesContainer = new QWidget(_scrollArea);
    auto *containerLayout = new QVBoxLayout(devicesContainer);
    containerLayout->setSpacing(10);

    _devicesWidget = new QWidget(devicesContainer);
    _devicesLayout = new QVBoxLayout(_devicesWidget);
    _devicesLayout->setSpacing(16);
    containerLayout->addWidget(_devicesWidget);

    _scrollArea->setWidget(devicesContainer);
    rootLayout->addWidget(_scrollArea, 1);
    _recordingLabel = new QLabel("Not recording", _scrollArea->viewport());
    _recordingLabel->setAlignment(Qt::AlignCenter);
    _recordingLabel->setStyleSheet("background-color: #333; color: #fff; padding: 4px; font-weight: bold;");
    _recordingLabel->setAttribute(Qt::WA_TransparentForMouseEvents, true);
    _recordingLabel->adjustSize();
    _recordingLabel->show();

    setCentralWidget(central);
    setWindowTitle("Modular Data Acquisition");
    setMinimumSize(1200, 900);
    resize(1600, 1000);
}

void MainWindow::updateControls()
{
    const bool hasCapture = _capture && _capture->isRunning();
    const bool recording = hasCapture && _capture->isRecording();
    const bool paused = recording && _capture->isPaused();

    _openButton->setEnabled(!hasCapture);
    _closeButton->setEnabled(hasCapture);
    _startCaptureButton->setEnabled(hasCapture && !recording);
    _stopCaptureButton->setEnabled(hasCapture && recording);
    _pauseButton->setEnabled(hasCapture && recording && !paused);
    _resumeButton->setEnabled(hasCapture && recording && paused);
    const bool taskActive = recording;
    _advanceButton->setEnabled(hasCapture); // allow step to start recording path
    _retryButton->setEnabled(taskActive);
    _skipButton->setEnabled(taskActive);
    _errorButton->setEnabled(taskActive);
    _abortButton->setEnabled(taskActive);
    if (_vlmCameraSelect)
    {
        _vlmCameraSelect->clear();
        for (const auto &cam : _preview.cameraIds())
        {
            _vlmCameraSelect->addItem(QString::fromStdString(cam));
        }
    }
}

MainWindow::CaptureInfo MainWindow::gatherCaptureInfo() const
{
    CaptureInfo info;
    info.name = _captureNameEdit->text().trimmed().toStdString();
    info.subject = _subjectEdit->text().trimmed().toStdString();
    auto sanitize = [](std::string value) {
        for (auto &ch : value)
        {
            if (!std::isalnum(static_cast<unsigned char>(ch)) && ch != '-' && ch != '_')
                ch = '_';
        }
        return value;
    };
    const auto now = std::chrono::system_clock::now();
    const auto timeT = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    localtime_s(&tm, &timeT);
#else
    localtime_r(&timeT, &tm);
#endif
    char dateBuf[16];
    char timeBuf[16];
    std::strftime(dateBuf, sizeof(dateBuf), "%Y-%m-%d", &tm);
    std::strftime(timeBuf, sizeof(timeBuf), "%Y%m%d_%H%M%S", &tm);
    const std::string sessionId = std::string("sess_") + timeBuf;

    std::filesystem::path base(_defaultCaptureRoot);
    base /= dateBuf;
    base /= sessionId;
    info.path = base.string();
    info.name = sessionId; // record session id as capture name
    return info;
}

void MainWindow::refreshCameraSettingsList()
{
    _cameraSelect->clear();
    const auto &configs = _configManager.getCameraConfigs();
    for (size_t i = 0; i < configs.size(); ++i)
    {
        const auto &cfg = configs[i];
        QString label = QString("%1 (ID %2)").arg(QString::fromStdString(cfg.type)).arg(cfg.id);
        _cameraSelect->addItem(label, static_cast<int>(i));
    }
    populateAvailableProfiles();
    onCameraSelectionChanged(_cameraSelect->currentIndex());
}

void MainWindow::populateAvailableProfiles()
{
    _colorResCombo->clear();
    _colorFpsCombo->clear();
    _depthResCombo->clear();
    _depthFpsCombo->clear();

    std::vector<std::pair<int, int>> colorRes;
    std::vector<std::pair<int, int>> depthRes;
    std::vector<int> colorFps;
    std::vector<int> depthFps;

    auto addProfile = [&](const CameraConfig::StreamConfig &cfg) {
        if (cfg.width <= 0 || cfg.height <= 0 || cfg.frameRate <= 0)
            return;
        if (cfg.streamType == CameraConfig::StreamConfig::StreamType::Color)
        {
            colorRes.push_back({cfg.width, cfg.height});
            colorFps.push_back(cfg.frameRate);
        }
        else
        {
            depthRes.push_back({cfg.width, cfg.height});
            depthFps.push_back(cfg.frameRate);
        }
    };

    if (_capture)
    {
        // Device-side profile enumeration is skipped; we will fall back to config entries below.
    }
    if (colorRes.empty() && depthRes.empty())
    {
        for (const auto &cfg : _configManager.getCameraConfigs())
        {
            CameraConfig::StreamConfig color = cfg.color;
            color.streamType = CameraConfig::StreamConfig::StreamType::Color;
            addProfile(color);
            CameraConfig::StreamConfig depth = cfg.depth;
            depth.streamType = CameraConfig::StreamConfig::StreamType::Depth;
            addProfile(depth);
        }
    }

    auto populateCombo = [](QComboBox *combo, std::vector<std::pair<int, int>> resVec) {
        std::sort(resVec.begin(), resVec.end());
        resVec.erase(std::unique(resVec.begin(), resVec.end()), resVec.end());
        for (const auto &res : resVec)
            combo->addItem(QString("%1 x %2").arg(res.first).arg(res.second),
                           QVariant::fromValue(QSize(res.first, res.second)));
    };
    auto populateFpsCombo = [](QComboBox *combo, std::vector<int> fpsVec) {
        std::sort(fpsVec.begin(), fpsVec.end());
        fpsVec.erase(std::unique(fpsVec.begin(), fpsVec.end()), fpsVec.end());
        for (int fps : fpsVec)
            combo->addItem(QString::number(fps), fps);
    };
    populateCombo(_colorResCombo, colorRes);
    populateCombo(_depthResCombo, depthRes);
    populateFpsCombo(_colorFpsCombo, colorFps);
    populateFpsCombo(_depthFpsCombo, depthFps);

    if (_colorResCombo->count() == 0)
        _colorResCombo->addItem("640 x 480", QSize(640, 480));
    if (_depthResCombo->count() == 0)
        _depthResCombo->addItem("640 x 480", QSize(640, 480));
    if (_colorFpsCombo->count() == 0)
        _colorFpsCombo->addItem("30", 30);
    if (_depthFpsCombo->count() == 0)
        _depthFpsCombo->addItem("30", 30);
}

void MainWindow::connectSignals()
{
    connect(_openButton, &QPushButton::clicked, this, &MainWindow::onOpenCameras);
    connect(_closeButton, &QPushButton::clicked, this, &MainWindow::onCloseCameras);
    connect(_startCaptureButton, &QPushButton::clicked, this, &MainWindow::onStartRecording);
    connect(_stopCaptureButton, &QPushButton::clicked, this, &MainWindow::onStopRecording);
    connect(_pauseButton, &QPushButton::clicked, this, &MainWindow::onPauseRecording);
    connect(_resumeButton, &QPushButton::clicked, this, &MainWindow::onResumeRecording);
    connect(_advanceButton, &QPushButton::clicked, this, &MainWindow::onAdvanceStep);
    connect(_retryButton, &QPushButton::clicked, this, &MainWindow::onRetryStep);
    connect(_skipButton, &QPushButton::clicked, this, &MainWindow::onSkipStep);
    connect(_errorButton, &QPushButton::clicked, this, &MainWindow::onErrorStep);
    connect(_abortButton, &QPushButton::clicked, this, &MainWindow::onAbortTask);
    connect(_cameraSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onCameraSelectionChanged);
    connect(_applySettingsButton, &QPushButton::clicked, this, &MainWindow::onApplyCameraSettings);
    connect(_sceneSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onSceneSelectionChanged);
    connect(_taskSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onTaskSelectionChanged);
    connect(_modeSelect, QOverload<int>::of(&QComboBox::currentIndexChanged),
            this, &MainWindow::onModeChanged);
    connect(_vlmGenerateButton, &QPushButton::clicked,
            this, &MainWindow::onGenerateVlmTask);
    connect(_viewTaskButton, &QPushButton::clicked, this, &MainWindow::showCurrentTaskDialog);
}

QGroupBox *MainWindow::createMetadataGroup()
{
    auto *box = new QGroupBox(tr("Capture Metadata"), _controlPanel);
    auto *formLayout = new QFormLayout(box);
    _captureNameEdit = new QLineEdit();
    _subjectEdit = new QLineEdit();
    _savePathEdit = new QLineEdit();
    auto *browseButton = new QPushButton(tr("Browse"));
    auto *pathLayout = new QHBoxLayout();
    pathLayout->addWidget(_savePathEdit, 1);
    pathLayout->addWidget(browseButton);
    formLayout->addRow(tr("Capture Name"), _captureNameEdit);
    formLayout->addRow(tr("Subject"), _subjectEdit);
    formLayout->addRow(tr("Save Path"), pathLayout);
    connect(browseButton, &QPushButton::clicked, this, &MainWindow::onBrowseSavePath);
    return box;
}

QGroupBox *MainWindow::createTaskSelectionGroup()
{
    auto *box = new QGroupBox(tr("Task Selection"), _controlPanel);
    auto *layout = new QFormLayout(box);

    _modeSelect = new QComboBox();
    _modeSelect->addItem(tr("Script (Local)"), "script");
    _modeSelect->addItem(tr("VLM (Generate)"), "vml");

    _sceneSelect = new QComboBox();
    _taskSelect = new QComboBox();
    _sceneLabel = new QLabel(tr("Scene"));
    _taskLabel = new QLabel(tr("Task"));
    layout->addRow(tr("Mode"), _modeSelect);
    layout->addRow(_sceneLabel, _sceneSelect);
    layout->addRow(_taskLabel, _taskSelect);
    _taskSelectionGroup = box;
    return box;
}

QGroupBox *MainWindow::createVlmGroup()
{
    auto *box = new QGroupBox(tr("VLM Generation"), _controlPanel);
    auto *layout = new QFormLayout(box);
    layout->setRowWrapPolicy(QFormLayout::DontWrapRows);
    layout->setFieldGrowthPolicy(QFormLayout::AllNonFixedFieldsGrow);
    layout->setVerticalSpacing(12);
    layout->setContentsMargins(10, 12, 10, 12);

    _vlmEndpointEdit = new QLineEdit();
    _vlmEndpointEdit->setReadOnly(true);
    _vlmPromptPathEdit = new QLineEdit();
    _vlmPromptPathEdit->setReadOnly(true);
    _vlmApiKeyEdit = new QLineEdit();
    _vlmApiKeyEdit->setEchoMode(QLineEdit::Password);
    _vlmApiKeyEdit->setReadOnly(true);
    _vlmCameraSelect = new QComboBox();
    _vlmGenerateButton = new QPushButton(tr("Generate Task (VLM)"));

    layout->addRow(tr("Endpoint"), _vlmEndpointEdit);
    layout->addRow(tr("Prompt Path"), _vlmPromptPathEdit);
    layout->addRow(tr("API Key"), _vlmApiKeyEdit);
    layout->addRow(tr("Input Camera"), _vlmCameraSelect);
    layout->addRow(_vlmGenerateButton);
    box->setSizePolicy(QSizePolicy::Preferred, QSizePolicy::Preferred);
    _vlmGroup = box;
    return box;
}

QGroupBox *MainWindow::createCameraControlGroup()
{
    auto *box = new QGroupBox(tr("Cameras"), _controlPanel);
    auto *layout = new QHBoxLayout(box);
    _openButton = new QPushButton("Open");
    _closeButton = new QPushButton("Close");
    layout->addWidget(_openButton);
    layout->addWidget(_closeButton);
    return box;
}

QGroupBox *MainWindow::createCaptureControlGroup()
{
    auto *box = new QGroupBox(tr("Capture Control"), _controlPanel);
    auto *grid = new QGridLayout(box);
    _startCaptureButton = new QPushButton("Start");
    _stopCaptureButton = new QPushButton("Stop");
    _pauseButton = new QPushButton("Pause");
    _resumeButton = new QPushButton("Resume");
    _advanceButton = new QPushButton("Step");
    _errorButton = new QPushButton("Error");
    _skipButton = new QPushButton("Skip");
    _retryButton = new QPushButton("Retry");
    _abortButton = new QPushButton("Abort");
    grid->addWidget(_startCaptureButton, 0, 0);
    grid->addWidget(_stopCaptureButton, 0, 1);
    grid->addWidget(_pauseButton, 1, 0);
    grid->addWidget(_resumeButton, 1, 1);
    grid->addWidget(_advanceButton, 2, 0);
    grid->addWidget(_retryButton, 2, 1);
    grid->addWidget(_skipButton, 3, 0);
    grid->addWidget(_errorButton, 3, 1);
    grid->addWidget(_abortButton, 4, 0, 1, 2);
    return box;
}

QGroupBox *MainWindow::createPromptControlGroup()
{
    auto *box = new QGroupBox(tr("Participant Prompt"), _controlPanel);
    auto *layout = new QVBoxLayout(box);
    auto *button = new QPushButton(tr("Show Prompt Window"));
    layout->addWidget(button);
    auto *engineLabel = new QLabel(tr("TTS Engine"));
    _audioEngineSelect = new QComboBox(box);
    _audioEngineSelect->addItem("IndexTTS", "index_tts");
    _audioEngineSelect->setCurrentIndex(0);
    layout->addWidget(engineLabel);
    layout->addWidget(_audioEngineSelect);
    auto *langLabel = new QLabel(tr("Prompt Language"));
    _promptLanguageSelect = new QComboBox(box);
    _promptLanguageSelect->addItem("English", static_cast<int>(PromptLanguage::English));
    _promptLanguageSelect->addItem("中文", static_cast<int>(PromptLanguage::Chinese));
    _promptLanguageSelect->setCurrentIndex(static_cast<int>(_promptLanguage));
    layout->addWidget(langLabel);
    layout->addWidget(_promptLanguageSelect);
    connect(_promptLanguageSelect, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        _promptLanguage = static_cast<PromptLanguage>(_promptLanguageSelect->itemData(idx).toInt());
        applyAudioConfigForLanguage();
    });
    connect(_audioEngineSelect, QOverload<int>::of(&QComboBox::currentIndexChanged), this, [this](int idx) {
        _audioConfig.mode = _audioEngineSelect->itemData(idx).toString().toStdString();
        applyAudioConfigForLanguage();
    });
    connect(button, &QPushButton::clicked, this, &MainWindow::onShowParticipantPrompt);
    return box;
}

QGroupBox *MainWindow::createCameraSettingsGroup()
{
    auto *box = new QGroupBox(tr("Camera Settings"), _controlPanel);
    auto *layout = new QGridLayout(box);
    int row = 0;
    layout->addWidget(new QLabel(tr("Camera")), row, 0);
    _cameraSelect = new QComboBox();
    layout->addWidget(_cameraSelect, row++, 1, 1, 2);

    _colorResCombo = new QComboBox();
    _colorFpsCombo = new QComboBox();
    _depthResCombo = new QComboBox();
    _depthFpsCombo = new QComboBox();

    layout->addWidget(new QLabel(tr("Color Resolution / FPS")), row, 0);
    auto *colorLayout = new QHBoxLayout();
    colorLayout->addWidget(_colorResCombo);
    colorLayout->addWidget(_colorFpsCombo);
    layout->addLayout(colorLayout, row++, 1, 1, 2);

    layout->addWidget(new QLabel(tr("Depth Resolution / FPS")), row, 0);
    auto *depthLayout = new QHBoxLayout();
    depthLayout->addWidget(_depthResCombo);
    depthLayout->addWidget(_depthFpsCombo);
    layout->addLayout(depthLayout, row++, 1, 1, 2);

    _applySettingsButton = new QPushButton(tr("Apply Settings"));
    layout->addWidget(_applySettingsButton, row, 0, 1, 3);
    return box;
}

QGroupBox *MainWindow::createStatusGroup()
{
    auto *box = new QGroupBox(tr("Status"), _controlPanel);
    auto *vlayout = new QVBoxLayout(box);
    _statusLabel = new QLabel("Idle");
    _statusLabel->setAlignment(Qt::AlignCenter);
    _taskStateLabel = new QLabel(tr("Task: -"));
    _taskStepLabel = new QLabel(tr("Step: -"));
    _taskNameLabel = new QLabel(tr("Current Task: -"));
    vlayout->addWidget(_statusLabel);
    vlayout->addWidget(_taskNameLabel);
    vlayout->addWidget(_taskStateLabel);
    vlayout->addWidget(_taskStepLabel);
    _viewTaskButton = new QPushButton(tr("View Current Task"), box);
    vlayout->addWidget(_viewTaskButton);
    return box;
}

void MainWindow::loadTaskTemplates()
{
    _taskTemplates.clear();
    const std::string root = _configManager.getTasksRootPath();
    _taskTemplates = _taskLoader.loadAllTasks(root);
    populateSceneList();
}

void MainWindow::populateSceneList()
{
    _sceneSelect->blockSignals(true);
    _taskSelect->blockSignals(true);
    _sceneSelect->clear();
    _taskSelect->clear();
    _sceneSelect->addItem(tr("Select Scene"), QString());
    _taskSelect->addItem(tr("Select Task"), QString());

    std::set<std::string> sceneIds;
    for (const auto &t : _taskTemplates)
        sceneIds.insert(t.sceneId);

    for (const auto &scene : sceneIds)
    {
        _sceneSelect->addItem(QString::fromStdString(scene), QString::fromStdString(scene));
    }
    _sceneSelect->blockSignals(false);
    _taskSelect->blockSignals(false);
}

void MainWindow::populateTaskList(const std::string &sceneId)
{
    _taskSelect->blockSignals(true);
    _taskSelect->clear();
    _taskSelect->addItem(tr("Select Task"), QString());
    for (const auto &t : _taskTemplates)
    {
        if (t.sceneId == sceneId)
        {
            QString label = QString::fromStdString(t.task.id);
            _taskSelect->addItem(label, QString::fromStdString(t.task.id));
        }
    }
    _taskSelect->blockSignals(false);
}

void MainWindow::onSceneSelectionChanged(int index)
{
    Q_UNUSED(index);
    const auto sceneId = _sceneSelect->currentData().toString().toStdString();
    populateTaskList(sceneId);
    _currentTask.reset();
    _taskMachine.reset();
    updateTaskStatusUi();
}

void MainWindow::onTaskSelectionChanged(int index)
{
    Q_UNUSED(index);
    const auto sceneId = _sceneSelect->currentData().toString().toStdString();
    const auto taskId = _taskSelect->currentData().toString().toStdString();
    _currentTask.reset();
    for (const auto &t : _taskTemplates)
    {
        if (t.sceneId == sceneId && t.task.id == taskId)
        {
            setCurrentTask(t, "script");
            break;
        }
    }
    updateTaskStatusUi();
}

void MainWindow::onModeChanged(int index)
{
    Q_UNUSED(index);
    const auto mode = _modeSelect->currentData().toString().toStdString();
    setMode(mode == "script" ? TaskMode::Script : TaskMode::Vlm);
    updateTaskStatusUi();
}

void MainWindow::onGenerateVlmTask()
{
    _storage.logEvent("vlm_generate_clicked");
    std::string cameraId;
    if (_vlmCameraSelect && _vlmCameraSelect->currentIndex() >= 0)
    {
        cameraId = _vlmCameraSelect->currentText().toStdString();
    }
    if (cameraId.empty())
    {
        QMessageBox::warning(this, tr("VLM"), tr("Please select an input camera."));
        return;
    }

    // Grab latest frame from preview if available
    std::optional<FrameData> latestFrame = _preview.latestFrame(cameraId);
    if (!latestFrame.has_value() || latestFrame->image.empty())
    {
        QMessageBox::warning(this, tr("VLM"), tr("No frame available for selected camera."));
        return;
    }
    // Save a temp image to pass to generator
    std::filesystem::path tempImg = std::filesystem::path(APP_LOG_DIR) / "vlm_input.png";
    cv::imwrite(tempImg.string(), latestFrame->image);

    QApplication::setOverrideCursor(Qt::WaitCursor);
    auto generated = generateTaskFromVlm(tempImg.string());
    QApplication::restoreOverrideCursor();
    if (!generated.has_value())
    {
        _audioPlayer.play("vlm_fail");
        return;
    }

    QString error;
    if (!validateTaskTemplate(*generated, error))
    {
        QMessageBox::warning(this, tr("VLM"), tr("VLM output invalid: %1").arg(error));
        _audioPlayer.play("vlm_fail");
        return;
    }

    setCurrentTask(*generated, "vml");
    updateTaskStatusUi();
    QMessageBox::information(this, tr("VLM"), tr("Task generated and loaded successfully."));
    _audioPlayer.play("vlm_success");
}

void MainWindow::applyVlmConfigUi()
{
    const auto &vlm = _configManager.getVlmConfig();
    if (_vlmEndpointEdit)
        _vlmEndpointEdit->setText(QString::fromStdString(vlm.endpoint));
    if (_vlmPromptPathEdit)
        _vlmPromptPathEdit->setText(QString::fromStdString(vlm.promptPath));
    if (_vlmApiKeyEdit)
        _vlmApiKeyEdit->setText(QString::fromStdString(vlm.apiKey));
    if (_modeSelect)
        _modeSelect->setCurrentIndex(0);
}

void MainWindow::setMode(TaskMode mode)
{
    _mode = mode;
    const bool isScript = (_mode == TaskMode::Script);
    if (_sceneLabel)
        _sceneLabel->setVisible(isScript);
    if (_taskLabel)
        _taskLabel->setVisible(isScript);
    if (_sceneSelect)
        _sceneSelect->setVisible(isScript);
    if (_taskSelect)
        _taskSelect->setVisible(isScript);
    if (_vlmGroup)
        _vlmGroup->setVisible(!isScript);
    clearCurrentTask();
}

void MainWindow::clearCurrentTask()
{
    _currentTask.reset();
    _taskMachine.reset();
}

std::string MainWindow::makeStartPromptText() const
{
    if (_currentTask)
    {
        if (useChinesePrompts())
        {
            const auto cn = getTaskSpokenPromptCn();
            if (!cn.empty())
                return cn;
        }
        return "Recording started. Task: " + (_currentTask->task.spokenPrompt.empty() ? _currentTask->task.description : _currentTask->task.spokenPrompt);
    }
    return "Recording started.";
}

std::string MainWindow::makeNextStepPromptText(const std::string &stepId, const std::string &subtaskId) const
{
    if (useChinesePrompts())
    {
        const auto cn = getStepSpokenPromptCnById(stepId, subtaskId);
        if (!cn.empty())
            return cn;
    }
    const auto spoken = getStepSpokenPromptById(stepId, subtaskId);
    if (!spoken.empty())
        return spoken;
    return "Next step: " + getStepDescriptionById(stepId, subtaskId);
}

std::string MainWindow::makeCompletePromptText() const
{
    if (useChinesePrompts())
        return "本轮任务已完成，将从第一个子任务重新开始。";
    return "Task completed; restarting from the first subtask.";
}

void MainWindow::setCurrentTask(const TaskTemplate &task, const std::string &source)
{
    _currentTask = task;
    _taskMachine.setTask(task);
    _storage.setTaskSelection(task.sceneId, task.task.id, task.sourcePath,
                              task.schemaVersion, source);
    // Preload TTS prompts for this task
    std::vector<std::string> texts;
    std::vector<std::tuple<std::string, std::string, std::string>> allSteps;
    if (!task.task.subtasks.empty())
    {
        for (const auto &st : task.task.subtasks)
        {
            for (const auto &s : st.steps)
                allSteps.emplace_back(s.id, s.description, st.id);
        }
    }
    else
    {
        for (const auto &s : task.task.steps)
            allSteps.emplace_back(s.id, s.description, "");
    }
    if (!allSteps.empty())
    {
        texts.push_back(makeStartPromptText());
        for (const auto &p : allSteps)
            texts.push_back(makeNextStepPromptText(std::get<0>(p), std::get<2>(p)));
        texts.push_back(makeCompletePromptText());
    }
    for (const auto &st : task.task.subtasks)
    {
        if (useChinesePrompts() && !st.spokenPromptCn.empty())
            texts.push_back(st.spokenPromptCn);
        else
            texts.push_back(st.spokenPrompt.empty() ? st.description : st.spokenPrompt);
    }
    // static prompts
    texts.push_back(_audioPlayer.renderText("abort", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("retry", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("skip", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("error", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("stop_completed", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("stop_running", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("vlm_success", task.task.id, "", "", ""));
    texts.push_back(_audioPlayer.renderText("vlm_fail", task.task.id, "", "", ""));
    _audioPlayer.preloadTexts(texts);
}

bool MainWindow::validateTaskTemplate(const TaskTemplate &task, QString &errorMessage) const
{
    if (task.sceneId.empty())
    {
        errorMessage = tr("scene_id is empty");
        return false;
    }
    if (task.task.id.empty())
    {
        errorMessage = tr("task.id is empty");
        return false;
    }
    // At minimum require steps present either directly or within subtasks.
    bool hasSteps = !task.task.steps.empty();
    for (const auto &sub : task.task.subtasks)
        hasSteps = hasSteps || !sub.steps.empty();
    if (!hasSteps)
    {
        errorMessage = tr("no steps defined");
        return false;
    }
    // Basic integrity: ids/descriptions for any provided steps/subtasks.
    for (const auto &sub : task.task.subtasks)
    {
        if (sub.id.empty() || sub.description.empty())
        {
            errorMessage = tr("subtask missing id or description");
            return false;
        }
        for (const auto &step : sub.steps)
        {
            if (step.id.empty() || step.description.empty())
            {
                errorMessage = tr("step missing id or description");
                return false;
            }
        }
    }
    for (const auto &step : task.task.steps)
    {
        if (step.id.empty() || step.description.empty())
        {
            errorMessage = tr("step missing id or description");
            return false;
        }
    }
    errorMessage.clear();
    return true;
}

void MainWindow::showCurrentTaskDialog()
{
    if (!_currentTask)
    {
        QMessageBox::information(this, tr("Task"), tr("No task selected."));
        return;
    }
    json j;
    j["schema_version"] = _currentTask->schemaVersion;
    j["scene"] = {
        {"scene_id", _currentTask->sceneId},
        {"description", _currentTask->sceneDescription},
    };
    json objs = json::array();
    for (const auto &o : _currentTask->sceneObjects)
    {
        objs.push_back({
            {"object_id", o.objectId},
            {"name", o.name},
            {"category", o.category},
            {"state", {{"placement", o.placement}, {"relative", o.relative}}}
        });
    }
    j["scene"]["objects"] = objs;

    json involved = json::array();
    for (const auto &io : _currentTask->task.involvedObjects)
    {
        involved.push_back({
            {"object_id", io.objectId},
            {"name", io.name},
            {"role", io.role},
            {"optional", io.optional}
        });
    }
    json steps = json::array();
    for (const auto &s : _currentTask->task.steps)
    {
        json stepObj = {{"id", s.id}, {"description", s.description}};
        json sInvolved = json::array();
        for (const auto &io : s.involvedObjects)
        {
            sInvolved.push_back({
                {"object_id", io.objectId},
                {"name", io.name},
                {"role", io.role},
                {"optional", io.optional}
            });
        }
        stepObj["involved_objects"] = sInvolved;
        steps.push_back(stepObj);
    }

    json subtasks = json::array();
    for (const auto &st : _currentTask->task.subtasks)
    {
        json stObj = {{"id", st.id}, {"description", st.description}};
        json stInvolved = json::array();
        for (const auto &io : st.involvedObjects)
        {
            stInvolved.push_back({
                {"object_id", io.objectId},
                {"name", io.name},
                {"role", io.role},
                {"optional", io.optional}
            });
        }
        stObj["involved_objects"] = stInvolved;
        json stSteps = json::array();
        for (const auto &s : st.steps)
        {
            json stepObj = {{"id", s.id}, {"description", s.description}};
            json sInvolved = json::array();
            for (const auto &io : s.involvedObjects)
            {
                sInvolved.push_back({
                    {"object_id", io.objectId},
                    {"name", io.name},
                    {"role", io.role},
                    {"optional", io.optional}
                });
            }
            stepObj["involved_objects"] = sInvolved;
            stSteps.push_back(stepObj);
        }
        stObj["steps"] = stSteps;
        subtasks.push_back(stObj);
    }
    j["task"] = {
        {"id", _currentTask->task.id},
        {"description", _currentTask->task.description},
        {"involved_objects", involved},
        {"steps", steps},
        {"subtasks", subtasks}
    };
    j["constraints"] = {{"notes", _currentTask->constraintsNotes}};
    j["source_path"] = _currentTask->sourcePath;
    std::string content = j.dump(2);
    auto *dialog = new QDialog(this);
    dialog->setWindowTitle(tr("Task Definition"));
    auto *layout = new QVBoxLayout(dialog);
    auto *textEdit = new QTextEdit(dialog);
    textEdit->setReadOnly(true);
    textEdit->setText(QString::fromStdString(content));
    auto *scroll = new QScrollArea(dialog);
    scroll->setWidgetResizable(true);
    scroll->setWidget(textEdit);
    layout->addWidget(scroll);
    auto *btn = new QPushButton(tr("Close"), dialog);
    connect(btn, &QPushButton::clicked, dialog, &QDialog::accept);
    layout->addWidget(btn);
    dialog->resize(700, 600);
    dialog->exec();
}

std::optional<TaskTemplate> MainWindow::loadTaskFromJson(const std::string &path, const std::string &source)
{
    auto loaded = _taskLoader.loadTaskFile(path);
    if (loaded.has_value())
    {
        loaded->sourcePath = path;
        return loaded;
    }
    return std::nullopt;
}

std::optional<TaskTemplate> MainWindow::generateTaskFromVlm(const std::string &imagePath)
{
    const auto &vlmCfg = _configManager.getVlmConfig();
    std::string apiKey = vlmCfg.apiKey;
    if (apiKey.empty())
    {
        const char *envKey = std::getenv("GPT_API_KEY");
        if (envKey)
            apiKey = envKey;
    }
    if (apiKey.empty())
    {
        QMessageBox::warning(this, tr("VLM"), tr("API Key is empty. Please set it in config.json or GPT_API_KEY."));
        return std::nullopt;
    }

    QFile imgFile(QString::fromStdString(imagePath));
    if (!imgFile.open(QIODevice::ReadOnly))
    {
        QMessageBox::warning(this, tr("VLM"), tr("Failed to read image: %1").arg(QString::fromStdString(imagePath)));
        return std::nullopt;
    }
    QByteArray imgData = imgFile.readAll();
    QString imgBase64 = imgData.toBase64();

    std::filesystem::path promptFullPath(vlmCfg.promptPath);
    if (promptFullPath.is_relative())
    {
        std::filesystem::path cfg(_configManager.configPath());
        promptFullPath = cfg.parent_path().parent_path() / promptFullPath;
    }

    QFile promptFile(QString::fromStdString(promptFullPath.string()));
    QString promptText;
    if (promptFile.open(QIODevice::ReadOnly))
    {
        promptText = QString::fromUtf8(promptFile.readAll());
    }
    if (promptText.isEmpty())
    {
        QMessageBox::warning(this, tr("VLM"), tr("Prompt is empty or not found."));
        return std::nullopt;
    }

    QJsonObject root;
    root["model"] = QString::fromStdString(vlmCfg.model);
    QJsonArray messages;
    QJsonObject sys;
    sys["role"] = "system";
    sys["content"] = promptText;
    messages.append(sys);
    QJsonObject user;
    user["role"] = "user";
    QJsonArray contentArr;
    QJsonObject textPart;
    textPart["type"] = "text";
    textPart["text"] = "Generate a task template for this image.";
    QJsonObject imagePart;
    imagePart["type"] = "image_url";
    QJsonObject urlObj;
    urlObj["url"] = "data:image/jpeg;base64," + imgBase64;
    imagePart["image_url"] = urlObj;
    contentArr.append(textPart);
    contentArr.append(imagePart);
    user["content"] = contentArr;
    messages.append(user);
    root["messages"] = messages;
    QJsonObject respFmt;
    respFmt["type"] = "json_object";
    root["response_format"] = respFmt;

    QNetworkAccessManager manager;
    QNetworkRequest request(QUrl(QString::fromStdString(vlmCfg.endpoint)));
    request.setHeader(QNetworkRequest::ContentTypeHeader, "application/json");
    request.setRawHeader("Authorization", QByteArray("Bearer ").append(QString::fromStdString(apiKey).toUtf8()));
    QByteArray body = QJsonDocument(root).toJson(QJsonDocument::Compact);

    QEventLoop loop;
    QNetworkReply *reply = manager.post(request, body);
    QObject::connect(reply, &QNetworkReply::finished, &loop, &QEventLoop::quit);
    loop.exec();

    if (reply->error() != QNetworkReply::NoError)
    {
        QMessageBox::warning(this, tr("VLM"), tr("Request failed: %1").arg(reply->errorString()));
        reply->deleteLater();
        return std::nullopt;
    }

    QByteArray respData = reply->readAll();
    reply->deleteLater();
    QJsonParseError parseError;
    auto respDoc = QJsonDocument::fromJson(respData, &parseError);
    if (respDoc.isNull())
    {
        QMessageBox::warning(this, tr("VLM"), tr("Invalid JSON response: %1").arg(parseError.errorString()));
        return std::nullopt;
    }

    QJsonObject respObj = respDoc.object();
    auto choices = respObj.value("choices").toArray();
    if (choices.isEmpty())
    {
        QMessageBox::warning(this, tr("VLM"), tr("No choices in response."));
        return std::nullopt;
    }
    auto msgObj = choices.first().toObject().value("message").toObject();
    auto contentVal = msgObj.value("content");
    json parsed;
    try
    {
        if (contentVal.isArray())
        {
            QString combined;
            for (const auto &part : contentVal.toArray())
            {
                if (part.isObject() && part.toObject().value("type") == "text")
                    combined += part.toObject().value("text").toString();
            }
            parsed = json::parse(combined.toStdString());
        }
        else if (contentVal.isString())
        {
            parsed = json::parse(contentVal.toString().toStdString());
        }
        else if (contentVal.isObject())
        {
            QJsonDocument tmp(contentVal.toObject());
            parsed = json::parse(tmp.toJson(QJsonDocument::Compact).toStdString());
        }
    }
    catch (const std::exception &e)
    {
        QMessageBox::warning(this, tr("VLM"), tr("Failed to parse VLM content: %1").arg(e.what()));
        return std::nullopt;
    }

    if (parsed.is_null())
    {
        QMessageBox::warning(this, tr("VLM"), tr("Parsed content is null."));
        return std::nullopt;
    }

    // Save output
    std::filesystem::path outPath = std::filesystem::path(APP_LOG_DIR) / "vlm_output.json";
    try
    {
        std::ofstream out(outPath);
        out << parsed.dump(2);
    }
    catch (...)
    {
    }

    // Convert to TaskTemplate using loader on saved file
    auto generated = loadTaskFromJson(outPath.string(), "vml");
    if (!generated.has_value())
    {
        QMessageBox::warning(this, tr("VLM"), tr("Failed to load generated task."));
        return std::nullopt;
    }
    return generated;
}

std::string MainWindow::getStepDescriptionById(const std::string &stepId, const std::string &subtaskId) const
{
    if (!_currentTask || stepId.empty())
        return {};
    if (!subtaskId.empty())
    {
        for (const auto &st : _currentTask->task.subtasks)
        {
            if (st.id != subtaskId)
                continue;
            for (const auto &s : st.steps)
            {
                if (s.id == stepId)
                    return s.description;
            }
        }
    }
    for (const auto &s : _currentTask->task.steps)
    {
        if (s.id == stepId)
            return s.description;
    }
    return {};
}

std::string MainWindow::getStepVideoPathById(const std::string &stepId, const std::string &subtaskId) const
{
    if (!_currentTask || stepId.empty())
        return {};
    if (!subtaskId.empty())
    {
        for (const auto &st : _currentTask->task.subtasks)
        {
            if (st.id != subtaskId)
                continue;
            for (const auto &s : st.steps)
            {
                if (s.id == stepId && !s.videoPath.empty())
                    return s.videoPath;
            }
        }
    }
    for (const auto &s : _currentTask->task.steps)
    {
        if (s.id == stepId && !s.videoPath.empty())
            return s.videoPath;
    }
    return {};
}

std::string MainWindow::getSubtaskDescriptionById(const std::string &subtaskId) const
{
    if (!_currentTask || subtaskId.empty())
        return {};
    for (const auto &st : _currentTask->task.subtasks)
    {
        if (st.id == subtaskId)
            return st.description;
    }
    return {};
}

std::string MainWindow::getStepSpokenPromptById(const std::string &stepId, const std::string &subtaskId) const
{
    if (!_currentTask || stepId.empty())
        return {};
    if (!subtaskId.empty())
    {
        for (const auto &st : _currentTask->task.subtasks)
        {
            if (st.id != subtaskId)
                continue;
            for (const auto &s : st.steps)
            {
                if (s.id == stepId && !s.spokenPrompt.empty())
                    return s.spokenPrompt;
            }
        }
    }
    for (const auto &s : _currentTask->task.steps)
    {
        if (s.id == stepId && !s.spokenPrompt.empty())
            return s.spokenPrompt;
    }
    return {};
}

std::string MainWindow::getSubtaskSpokenPromptById(const std::string &subtaskId) const
{
    if (!_currentTask || subtaskId.empty())
        return {};
    for (const auto &st : _currentTask->task.subtasks)
    {
        if (st.id == subtaskId && !st.spokenPrompt.empty())
            return st.spokenPrompt;
    }
    return {};
}

std::string MainWindow::getStepSpokenPromptCnById(const std::string &stepId, const std::string &subtaskId) const
{
    if (!_currentTask || stepId.empty())
        return {};
    if (!subtaskId.empty())
    {
        for (const auto &st : _currentTask->task.subtasks)
        {
            if (st.id != subtaskId)
                continue;
            for (const auto &s : st.steps)
            {
                if (s.id == stepId && !s.spokenPromptCn.empty())
                    return s.spokenPromptCn;
            }
        }
    }
    for (const auto &s : _currentTask->task.steps)
    {
        if (s.id == stepId && !s.spokenPromptCn.empty())
            return s.spokenPromptCn;
    }
    return {};
}

std::string MainWindow::getSubtaskSpokenPromptCnById(const std::string &subtaskId) const
{
    if (!_currentTask || subtaskId.empty())
        return {};
    for (const auto &st : _currentTask->task.subtasks)
    {
        if (st.id == subtaskId && !st.spokenPromptCn.empty())
            return st.spokenPromptCn;
    }
    return {};
}

std::string MainWindow::getTaskSpokenPromptCn() const
{
    if (!_currentTask)
        return {};
    return _currentTask->task.spokenPromptCn;
}

bool MainWindow::useChinesePrompts() const
{
    return _promptLanguage == PromptLanguage::Chinese;
}

void MainWindow::applyAudioConfigForLanguage()
{
    AudioPromptConfig apc;
    const auto &cfg = _audioConfig;
    apc.enabled = cfg.enabled;
    apc.volume = cfg.volume;
    apc.mode = "index_tts";
    apc.indexTts.endpoint = cfg.indexTts.endpoint;
    apc.indexTts.audioPaths = cfg.indexTts.audioPaths;
    apc.texts = cfg.texts;
    _audioPlayer.configure(apc);
}

void MainWindow::updateVlmPromptMetadata()
{
    const auto &vlm = _configManager.getVlmConfig();
    std::string content;
    if (!vlm.promptPath.empty())
    {
        std::ifstream in(vlm.promptPath);
        if (in)
        {
            std::ostringstream ss;
            ss << in.rdbuf();
            content = ss.str();
        }
    }
    _storage.setVlmPrompt(vlm.promptPath, content);
}

int MainWindow::keyFromString(const std::string &keyStr) const
{
    if (keyStr.empty())
        return 0;
    if (keyStr.size() == 1)
    {
        char c = keyStr[0];
        return Qt::Key(std::toupper(static_cast<unsigned char>(c)));
    }
    // Simple mappings; extend as needed
    if (keyStr == "Space" || keyStr == "SPACE")
        return Qt::Key_Space;
    return 0;
}

void MainWindow::updateKeyBindings()
{
    // No-op for now; keyPressEvent reads directly from config when needed
}

void MainWindow::keyPressEvent(QKeyEvent *event)
{
    const auto &keys = _configManager.getAudioPromptsConfig().keybindings;
    auto matchKey = [&](const std::string &action) -> bool {
        auto it = keys.find(action);
        if (it == keys.end())
            return false;
        int k = keyFromString(it->second);
        return k != 0 && event->key() == k;
    };

    if (matchKey("advance"))
        onAdvanceStep();
    else if (matchKey("error"))
        onErrorStep();
    else if (matchKey("skip"))
        onSkipStep();
    else if (matchKey("retry"))
        onRetryStep();
    else if (matchKey("abort"))
        onAbortTask();
    QMainWindow::keyPressEvent(event);
}

void MainWindow::onOpenCameras()
{
    if (_capture && _capture->isRunning())
        return;
    _preview.clearViews();

    auto configs = _configManager.getCameraConfigs();
    std::vector<std::string> rsSerials;
    try
    {
        rs2::context ctx;
        for (auto &&dev : ctx.query_devices())
        {
            if (dev.supports(RS2_CAMERA_INFO_SERIAL_NUMBER))
                rsSerials.push_back(dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER));
        }
    }
    catch (const rs2::error &e)
    {
        _logger.warn("Failed to enumerate RealSense devices: %s", e.what());
    }
    size_t rsIndex = 0;

    std::vector<DeviceSpec> devices;
    for (auto &cfg : configs)
    {
        if (cfg.type == "RealSense" && cfg.serial.empty())
        {
            if (rsIndex < rsSerials.size())
            {
                cfg.serial = rsSerials[rsIndex++];
                _logger.info("Auto-assigned RealSense serial %s to config id=%d", cfg.serial.c_str(), cfg.id);
            }
            else
            {
                _logger.warn("No available RealSense device for config id=%d; initialization may fail", cfg.id);
            }
        }

        auto dev = createCamera(cfg, _logger);
        if (dev && dev->initialize(cfg))
        {
            DeviceSpec spec;
            spec.device = std::move(dev);
            spec.type = cfg.type;
            devices.push_back(std::move(spec));
        }
        else
        {
            _logger.warn("Failed to initialize device id=%d type=%s", cfg.id, cfg.type.c_str());
        }
    }

    _capture = std::make_unique<DataCapture>(std::move(devices),
                                             _storage, _preview, _logger,
                                             _arucoTracker.get());
    if (_capture->start())
    {
        _statusLabel->setText("Cameras opened");
        if (_arucoTracker && _arucoTracker->isAvailable())
            _arucoTracker->startSession(_storage.basePath());
    }
    else
    {
        _statusLabel->setText("Failed to open cameras");
        _capture.reset();
    }
    populateAvailableProfiles();
    updateControls();
}

void MainWindow::onCloseCameras()
{
    if (_capture)
    {
        _capture->stop();
        _capture.reset();
        if (_arucoTracker && _arucoTracker->isAvailable())
            _arucoTracker->endSession();
        _preview.clearViews();
        _statusLabel->setText("Cameras closed");
    }
    populateAvailableProfiles();
    updateControls();
}

void MainWindow::onStartRecording()
{
    if (_capture)
    {
        if (!_currentTask)
        {
            QMessageBox::warning(this, tr("Start"), tr("Please select a task first."));
            return;
        }
        if (_subjectEdit->text().trimmed().isEmpty())
        {
            QMessageBox::warning(this, tr("Start"), tr("Please enter subject information before recording."));
            return;
        }
        if (_taskMachine.state() == TaskStateMachine::State::Idle)
            _taskMachine.setTask(*_currentTask);

        auto t = _taskMachine.beginSession();
        if (t.state != TaskStateMachine::State::Ready && t.state != TaskStateMachine::State::SubtaskReady && t.state != TaskStateMachine::State::Running)
        {
            QMessageBox::warning(this, tr("Start"), tr("Cannot start from current state."));
            return;
        }
        _storage.setTaskSelection(_currentTask->sceneId, _currentTask->task.id,
                                  _currentTask->sourcePath, _currentTask->schemaVersion, "script");
        auto info = gatherCaptureInfo();
        _capture->startRecording(info.name, info.subject, info.path);
        logAnnotation("start_recording");
        _statusLabel->setText("Recording...");
        _audioPlayer.play("start", _currentTask->task.id, "", "", makeStartPromptText());
        if (t.current.has_value())
            updatePromptWindowMedia(t.current->subtaskId, t.current->stepId);
        updateTaskStatusUi();
    }
    updateControls();
}

void MainWindow::onStopRecording()
{
    if (_capture)
    {
        bool wasCompleted = (_taskMachine.state() == TaskStateMachine::State::Completed);
        if (_taskMachine.state() == TaskStateMachine::State::Running || _taskMachine.state() == TaskStateMachine::State::SubtaskReady)
            _taskMachine.abort();
        _taskMachine.stop();
        logAnnotation("stop_recording");
        _capture->stopRecording();
        _statusLabel->setText("Recording stopped");
        stopPromptVideo();
        if (wasCompleted)
            _audioPlayer.play("stop_completed");
        else
            _audioPlayer.play("stop_running");
        updateTaskStatusUi();
    }
    updateControls();
}

void MainWindow::onPauseRecording()
{
    if (_capture)
    {
        _capture->pauseRecording();
        _statusLabel->setText("Recording paused");
        updateTaskStatusUi();
    }
    updateControls();
}

void MainWindow::onResumeRecording()
{
    if (_capture)
    {
        _capture->resumeRecording();
        _statusLabel->setText("Recording resumed");
        updateTaskStatusUi();
    }
    updateControls();
}

void MainWindow::onAdvanceStep()
{
    if (!_capture || !_currentTask)
    {
        QMessageBox::warning(this, tr("Advance"), tr("Please start recording and select a task first."));
        return;
    }
    if (!_capture->isRecording())
    {
        onStartRecording();
        return; // avoid double-advancing immediately after starting
    }
    _storage.logEvent("step_trigger");
    auto t = _taskMachine.advance();
    if (t.taskCompleted)
    {
        _audioPlayer.play("complete", _currentTask->task.id, "", "", makeCompletePromptText());
    }
    else if (t.state == TaskStateMachine::State::SubtaskReady && !t.subtaskStarted)
    {
        const auto subId = t.current ? t.current->subtaskId : "";
        const auto subDescCn = getSubtaskSpokenPromptCnById(subId);
        const auto subDesc = getSubtaskSpokenPromptById(subId);
        const auto fallback = getSubtaskDescriptionById(subId);
        if (useChinesePrompts() && !subDescCn.empty())
            _audioPlayer.play("next_step", _currentTask->task.id, "", "", subDescCn);
        else
            _audioPlayer.play("next_step", _currentTask->task.id, "", "", subDesc.empty() ? fallback : subDesc);
        if (t.current.has_value())
            updatePromptWindowMedia(t.current->subtaskId, t.current->stepId);
    }
    else if (t.subtaskStarted && t.current.has_value())
    {
        _audioPlayer.play("next_step", _currentTask->task.id, t.current->stepId, "", makeNextStepPromptText(t.current->stepId, t.current->subtaskId));
        updatePromptWindowMedia(t.current->subtaskId, t.current->stepId);
    }
    else if (t.current.has_value())
    {
        _audioPlayer.play("next_step", _currentTask->task.id, t.current->stepId, "", makeNextStepPromptText(t.current->stepId, t.current->subtaskId));
        updatePromptWindowMedia(t.current->subtaskId, t.current->stepId);
    }
    if (t.taskCompleted)
        stopPromptVideo();
    logAnnotation("button_step");
    updateTaskStatusUi();
}

void MainWindow::onRetryStep()
{
    _storage.logEvent("step_retry");
    _audioPlayer.play("retry");
    logAnnotation("button_retry");
}

void MainWindow::onSkipStep()
{
    _storage.logEvent("step_skip");
    _audioPlayer.play("skip");
    logAnnotation("button_skip");
}

void MainWindow::onErrorStep()
{
    _storage.logEvent("step_error");
    _audioPlayer.play("error");
    logAnnotation("button_error");
}

void MainWindow::onAbortTask()
{
    _storage.logEvent("task_abort");
    _taskMachine.abort();
    _taskMachine.stop();
    _audioPlayer.play("abort");
    logAnnotation("button_abort");
    updateTaskStatusUi();
}

int64_t MainWindow::nowMs() const
{
    const auto now = std::chrono::system_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
}

std::string MainWindow::stateToString(TaskStateMachine::State state) const
{
    switch (state)
    {
    case TaskStateMachine::State::Idle:
        return "idle";
    case TaskStateMachine::State::Ready:
        return "ready";
    case TaskStateMachine::State::SubtaskReady:
        return "subtask_ready";
    case TaskStateMachine::State::Running:
        return "running";
    case TaskStateMachine::State::Completed:
        return "completed";
    case TaskStateMachine::State::Aborted:
        return "aborted";
    default:
        return "unknown";
    }
}

void MainWindow::logAnnotation(const std::string &trigger)
{
    if (!_currentTask)
        return;
    auto snap = _taskMachine.snapshot();
    DataStorage::AnnotationEntry entry;
    entry.source = _currentTask ? "script" : "unknown";
    entry.sceneId = _currentTask->sceneId;
    entry.taskId = _currentTask->task.id;
    entry.templatePath = "task_used.json";
    entry.templateVersion = _currentTask->schemaVersion;
    entry.state = stateToString(snap.state);
    entry.currentStepId = snap.currentStepId;
    // For subtasks, prepend subtask id to step if needed
    if (!snap.currentSubtaskId.empty())
        entry.currentStepId = snap.currentSubtaskId + ":" + snap.currentStepId;
    entry.timestampMs = nowMs();
    entry.triggerType = trigger;

    for (const auto &st : snap.steps)
    {
        if (st.done || st.attempts > 0)
        {
            DataStorage::StepOverride ov;
            ov.stepId = st.id;
            ov.done = st.done;
            ov.attempts = st.attempts;
            entry.stepOverrides.push_back(ov);
        }
    }
    _storage.logAnnotation(entry);
}

void MainWindow::updateRecordingBanner()
{
    if (!_recordingLabel)
        return;
    bool recording = _capture && _capture->isRecording() && !_capture->isPaused();
    bool paused = _capture && _capture->isRecording() && _capture->isPaused();
    QString text;
    QString style;
    if (recording)
    {
        text = "Recording...";
        style = "background-color: #b71c1c; color: #fff; padding: 6px; font-weight: bold;";
    }
    else if (paused)
    {
        text = "Paused";
        style = "background-color: #f9a825; color: #000; padding: 6px; font-weight: bold;";
    }
    else
    {
        text = "Not recording";
        style = "background-color: #333; color: #fff; padding: 6px; font-weight: bold;";
    }
    auto label = _recordingLabel;
    if (QThread::currentThread() == label->thread())
    {
        label->setText(text);
        label->setStyleSheet(style);
    }
    else
    {
        QMetaObject::invokeMethod(label, [label, text, style]() {
            label->setText(text);
            label->setStyleSheet(style);
        }, Qt::QueuedConnection);
    }
    positionRecordingLabel();
}

void MainWindow::positionRecordingLabel()
{
    if (!_recordingLabel || !_scrollArea)
        return;
    auto *vp = _scrollArea->viewport();
    if (!vp)
        return;
    const int margin = 8;
    _recordingLabel->adjustSize();
    int x = vp->width() - _recordingLabel->width() - margin;
    int y = margin;
    _recordingLabel->move(x, y);
}

void MainWindow::resizeEvent(QResizeEvent *event)
{
    QMainWindow::resizeEvent(event);
    positionRecordingLabel();
}

void MainWindow::updateTaskStatusUi()
{
    const auto stateStr = stateToString(_taskMachine.state());
    _taskStateLabel->setText(QString("Task State: %1").arg(QString::fromStdString(stateStr)));
    if (_currentTask)
    {
        _taskNameLabel->setText(QString("Current Task: %1").arg(QString::fromStdString(_currentTask->task.id)));
    }
    else
    {
        _taskNameLabel->setText("Current Task: -");
    }

    QString stepText = "Step: -";
    if (_taskMachine.state() == TaskStateMachine::State::Running ||
        _taskMachine.state() == TaskStateMachine::State::Completed ||
        _taskMachine.state() == TaskStateMachine::State::Aborted)
    {
        if (auto id = _taskMachine.currentStepId(); id.has_value())
        {
            std::string desc;
            if (_currentTask)
            {
                for (const auto &s : _currentTask->task.steps)
                {
                    if (s.id == *id)
                    {
                        desc = s.description;
                        break;
                    }
                }
            }
            if (!desc.empty())
                stepText = QString("Step: %1 - %2").arg(QString::fromStdString(*id),
                                                        QString::fromStdString(desc));
            else
                stepText = QString("Step: %1").arg(QString::fromStdString(*id));
        }
    }
    _taskStepLabel->setText(stepText);
    updateRecordingBanner();
}

void MainWindow::onBrowseSavePath()
{
    const auto dir = QFileDialog::getExistingDirectory(this, tr("选择保存目录"),
                                                       _savePathEdit->text());
    if (!dir.isEmpty())
        _savePathEdit->setText(dir);
}

void MainWindow::onCameraSelectionChanged(int index)
{
    const auto &configs = _configManager.getCameraConfigs();
    if (index < 0 || index >= static_cast<int>(configs.size()))
        return;
    const auto &cfg = configs[index];
    auto setResolution = [](QComboBox *combo, int width, int height) {
        QSize target(width, height);
        int idx = -1;
        for (int i = 0; i < combo->count(); ++i)
        {
            if (combo->itemData(i).toSize() == target)
            {
                idx = i;
                break;
            }
        }
        if (idx == -1)
        {
            combo->addItem(QString("%1 x %2").arg(width).arg(height), target);
            idx = combo->count() - 1;
        }
        combo->setCurrentIndex(idx);
    };
    auto setFps = [](QComboBox *combo, int fps) {
        int idx = combo->findData(fps);
        if (idx == -1)
        {
            combo->addItem(QString::number(fps), fps);
            idx = combo->count() - 1;
        }
        combo->setCurrentIndex(idx);
    };
    setResolution(_colorResCombo, cfg.color.width > 0 ? cfg.color.width : cfg.width,
                  cfg.color.height > 0 ? cfg.color.height : cfg.height);
    setResolution(_depthResCombo, cfg.depth.width > 0 ? cfg.depth.width : cfg.width,
                  cfg.depth.height > 0 ? cfg.depth.height : cfg.height);
    setFps(_colorFpsCombo, cfg.color.frameRate > 0 ? cfg.color.frameRate : cfg.frameRate);
    setFps(_depthFpsCombo, cfg.depth.frameRate > 0 ? cfg.depth.frameRate : cfg.frameRate);
}

void MainWindow::onApplyCameraSettings()
{
    const auto &configs = _configManager.getCameraConfigs();
    int index = _cameraSelect->currentIndex();
    if (index < 0 || index >= static_cast<int>(configs.size()))
        return;
    CameraConfig updated = configs[index];
    auto colorSize = _colorResCombo->currentData().toSize();
    auto depthSize = _depthResCombo->currentData().toSize();
    if (!colorSize.isValid())
        colorSize = QSize(updated.color.width, updated.color.height);
    if (!depthSize.isValid())
        depthSize = QSize(updated.depth.width, updated.depth.height);
    updated.color.width = colorSize.width();
    updated.color.height = colorSize.height();
    updated.color.frameRate = _colorFpsCombo->currentData().toInt();
    updated.depth.width = depthSize.width();
    updated.depth.height = depthSize.height();
    updated.depth.frameRate = _depthFpsCombo->currentData().toInt();
    updated.width = updated.color.width;
    updated.height = updated.color.height;
    updated.frameRate = updated.color.frameRate;
    _configManager.updateCameraConfig(updated.id, updated);
    if (_capture && _capture->isRunning())
    {
        onCloseCameras();
        _statusLabel->setText("Settings applied. Cameras closed, please reopen.");
    }
    else
    {
        _statusLabel->setText("Settings applied.");
    }
}

void MainWindow::onShowParticipantPrompt()
{
    ensurePromptWindow();
    if (_promptWindow && !_promptWindow->isVisible())
    {
        if (auto id = _taskMachine.currentStepId(); id.has_value())
            updatePromptWindowMedia(_taskMachine.currentSubtaskId().value_or(""), *id);
    }
    _promptWindow->show();
    _promptWindow->raise();
    _promptWindow->activateWindow();
}

void MainWindow::ensurePromptWindow()
{
    if (_promptWindow)
        return;
    _promptWindow = new QDialog(this);
    _promptWindow->setWindowTitle(tr("Participant Prompt"));
    _promptWindow->setMinimumSize(520, 360);
    auto *layout = new QVBoxLayout(_promptWindow);
    _promptStepLabel = new QLabel(tr("No task selected."), _promptWindow);
    _promptStepLabel->setWordWrap(true);
    layout->addWidget(_promptStepLabel);
    _promptVideoWidget = new QVideoWidget(_promptWindow);
    _promptVideoWidget->setMinimumSize(480, 270);
    layout->addWidget(_promptVideoWidget);
    _promptPlayer = new QMediaPlayer(_promptWindow);
    _promptPlayer->setVideoOutput(_promptVideoWidget);
}

void MainWindow::stopPromptVideo()
{
    if (_promptPlayer)
        _promptPlayer->stop();
    if (_promptVideoWidget)
        _promptVideoWidget->hide();
}

void MainWindow::updatePromptWindowMedia(const std::string &subtaskId, const std::string &stepId)
{
    if (stepId.empty() || !_currentTask)
        return;
    ensurePromptWindow();
    const auto desc = getStepDescriptionById(stepId, subtaskId);
    if (_promptStepLabel)
        _promptStepLabel->setText(QString::fromStdString(desc.empty() ? stepId : desc));
    const auto path = getStepVideoPathById(stepId, subtaskId);
    if (path.empty() || !QFile::exists(QString::fromStdString(path)))
    {
        stopPromptVideo();
        return;
    }
    if (_promptVideoWidget)
        _promptVideoWidget->show();
    if (_promptPlayer)
    {
#if QT_VERSION >= QT_VERSION_CHECK(6, 0, 0)
        _promptPlayer->setSource(QUrl::fromLocalFile(QString::fromStdString(path)));
#else
        _promptPlayer->setMedia(QUrl::fromLocalFile(QString::fromStdString(path)));
#endif
        _promptPlayer->play();
    }
}

#include "MainWindow.moc"
#include <utility>
