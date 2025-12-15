#pragma once

#include <QMainWindow>
#include <memory>
#include <string>
#include <optional>
#include <cstdint>
#include <unordered_map>
#include <chrono>

#include "Logger.h"
#include "ConfigManager.h"
#include "ArucoTracker.h"
#include "DataCapture.h"
#include "DataStorage.h"
#include "Preview.h"
#include "TaskLoader.h"
#include "TaskStateMachine.h"
#include "AudioPromptPlayer.h"

#ifndef APP_CONFIG_PATH
#define APP_CONFIG_PATH "./resources/config.json"
#endif

#ifndef APP_LOG_DIR
#define APP_LOG_DIR "./resources/logs"
#endif

enum class PromptLanguage
{
    English,
    Chinese
};

class QLineEdit;
class QPushButton;
class QComboBox;
class QLabel;
class QVBoxLayout;
class QWidget;
class QScrollArea;
class QGroupBox;
class QDialog;
class QMediaPlayer;
class QVideoWidget;

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    MainWindow();

private slots:
    void onOpenCameras();
    void onCloseCameras();
    void onStartRecording();
    void onStopRecording();
    void onPauseRecording();
    void onResumeRecording();
    void onStartStep();
    void onEndStep();
    void onSkipStep();
    void onRetryStep();
    void onMarkKeyframe();
    void onBrowseSavePath();
    void onCameraSelectionChanged(int index);
    void onApplyCameraSettings();
    void onShowParticipantPrompt();
    void onSceneSelectionChanged(int index);
    void onTaskSelectionChanged(int index);
    void onModeChanged(int index);
    void onGenerateVlmTask();

private:
    Logger _logger;
    ConfigManager _configManager;
    Preview _preview;
    const std::string _defaultCaptureRoot;
    DataStorage _storage;
    std::unique_ptr<ArucoTracker> _arucoTracker;
    std::unique_ptr<DataCapture> _capture;
    TaskLoader _taskLoader;
    TaskStateMachine _taskMachine;
    AudioPromptPlayer _audioPlayer;

    QPushButton *_openButton{nullptr};
    QPushButton *_closeButton{nullptr};
    QPushButton *_startCaptureButton{nullptr};
    QPushButton *_stopCaptureButton{nullptr};
    QPushButton *_pauseButton{nullptr};
    QPushButton *_resumeButton{nullptr};
    QPushButton *_startStepButton{nullptr};
    QPushButton *_endStepButton{nullptr};
    QPushButton *_skipButton{nullptr};
    QPushButton *_retryButton{nullptr};
    QPushButton *_keyframeButton{nullptr};
    QComboBox *_modeSelect{nullptr};
    QPushButton *_applySettingsButton{nullptr};
    QLineEdit *_captureNameEdit{nullptr};
    QLineEdit *_subjectEdit{nullptr};
    QLineEdit *_savePathEdit{nullptr};
    QComboBox *_cameraSelect{nullptr};
    QComboBox *_colorResCombo{nullptr};
    QComboBox *_colorFpsCombo{nullptr};
    QComboBox *_depthResCombo{nullptr};
    QComboBox *_depthFpsCombo{nullptr};
    QLabel *_statusLabel{nullptr};
    QLabel *_taskStateLabel{nullptr};
    QLabel *_taskStepLabel{nullptr};
    QLabel *_taskNameLabel{nullptr};
    QLabel *_sceneLabel{nullptr};
    QLabel *_taskLabel{nullptr};
    QLabel *_recordingLabel{nullptr};
    QLabel *_promptStepLabel{nullptr};
    QVBoxLayout *_devicesLayout{nullptr};
    QWidget *_devicesWidget{nullptr};
    QScrollArea *_scrollArea{nullptr};
    QWidget *_controlPanel{nullptr};
    QDialog *_promptWindow{nullptr};
    QMediaPlayer *_promptPlayer{nullptr};
    QVideoWidget *_promptVideoWidget{nullptr};
    QComboBox *_sceneSelect{nullptr};
    QComboBox *_taskSelect{nullptr};
    QWidget *_taskSelectionGroup{nullptr};
    QWidget *_vlmGroup{nullptr};
    QLineEdit *_vlmEndpointEdit{nullptr};
    QLineEdit *_vlmPromptPathEdit{nullptr};
    QLineEdit *_vlmApiKeyEdit{nullptr};
    QPushButton *_vlmGenerateButton{nullptr};
    QComboBox *_vlmCameraSelect{nullptr};
    QPushButton *_viewTaskButton{nullptr};
    AudioPromptsConfig _audioConfig;

    std::vector<TaskTemplate> _taskTemplates;
    std::optional<TaskTemplate> _currentTask;
    enum class TaskMode { Script, Vlm };
    TaskMode _mode{TaskMode::Script};
    std::vector<std::string> _previewCameraList;

    void setupUi();
    void updateControls();
    struct CaptureInfo
    {
        std::string name;
        std::string subject;
        std::string path;
    };
    CaptureInfo gatherCaptureInfo() const;
    void refreshCameraSettingsList();
    void populateAvailableProfiles();
    void connectSignals();
    QGroupBox *createMetadataGroup();
    QGroupBox *createTaskSelectionGroup();
    QGroupBox *createVlmGroup();
    QGroupBox *createCameraControlGroup();
    QGroupBox *createCaptureControlGroup();
    QGroupBox *createPromptControlGroup();
    QGroupBox *createCameraSettingsGroup();
    QGroupBox *createStatusGroup();
    void loadTaskTemplates();
    void populateSceneList();
    void populateTaskList(const std::string &sceneId);
    void keyPressEvent(QKeyEvent *event) override;
    void logAnnotation(const std::string &trigger);
    int64_t nowMs() const;
    std::string stateToString(TaskStateMachine::State state) const;
    void updateTaskStatusUi();
    void setMode(TaskMode mode);
    void clearCurrentTask();
    void setCurrentTask(const TaskTemplate &task, const std::string &source);
    void applyVlmConfigUi();
    std::string makeStartPromptText() const;
    std::string makeNextStepPromptText(const std::string &stepId, const std::string &subtaskId = "") const;
    std::string makeCompletePromptText() const;
    bool validateTaskTemplate(const TaskTemplate &task, QString &errorMessage) const;
    void showCurrentTaskDialog();
    std::optional<TaskTemplate> loadTaskFromJson(const std::string &path, const std::string &source);
    std::optional<TaskTemplate> generateTaskFromVlm(const std::string &imagePath);
    std::string getStepDescriptionById(const std::string &stepId, const std::string &subtaskId = "") const;
    std::string getSubtaskDescriptionById(const std::string &subtaskId) const;
    std::string getStepSpokenPromptById(const std::string &stepId, const std::string &subtaskId = "") const;
    std::string getSubtaskSpokenPromptById(const std::string &subtaskId) const;
    std::string getStepSpokenPromptCnById(const std::string &stepId, const std::string &subtaskId = "") const;
    std::string getSubtaskSpokenPromptCnById(const std::string &subtaskId) const;
    std::string getTaskSpokenPromptCn() const;
    std::string getStepVideoPathById(const std::string &stepId, const std::string &subtaskId = "") const;
    void updatePromptWindowMedia(const std::string &subtaskId, const std::string &stepId);
    void stopPromptVideo();
    void ensurePromptWindow();
    void updateKeyBindings();
    int keyFromString(const std::string &keyStr) const;
    void updateRecordingBanner();
    void positionRecordingLabel();
    void resizeEvent(QResizeEvent *event) override;
    ::PromptLanguage promptLanguageFromConfig() const;
    void applyAudioConfigForLanguage();
    void updateVlmPromptMetadata();

    ::PromptLanguage _promptLanguage{::PromptLanguage::Chinese};
    struct StepTiming
    {
        std::string subtaskId;
        std::string stepId;
        int64_t startMs{0};
    };
    std::optional<StepTiming> _activeStepTiming;
    std::unordered_map<std::string, std::chrono::steady_clock::time_point> _lastTrigger;
    bool throttleTrigger(const std::string &action, int ms);
    bool useChinesePrompts() const;
};
#include "ArucoTracker.h"
