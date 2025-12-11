#pragma once

#include <optional>
#include <string>
#include <vector>
#include <unordered_map>

#include <QJsonObject>

class Logger;

struct SceneMeta
{
    std::string id;
    std::string name;
    std::string nameCn;
    std::string description;
};

struct SceneObject
{
    std::string objectId;
    std::string name;
    std::string category;
    std::string placement;
    std::string relative;
};

struct TaskStepObject
{
    std::string objectId;
    std::string name;
    std::string role;
    bool optional = false;
};

struct TaskStep
{
    std::string id;
    std::string description;
    std::string spokenPrompt;
    std::string spokenPromptCn;
    std::string videoPath;
    std::vector<TaskStepObject> involvedObjects;
};

struct TaskDefinition
{
    std::string id;
    std::string name;
    std::string nameCn;
    std::string description;
    std::string spokenPrompt;
    std::string spokenPromptCn;
    std::vector<TaskStepObject> involvedObjects;
    std::vector<TaskStep> steps;
    std::vector<struct Subtask> subtasks; // forward decl placeholder
};

struct Subtask
{
    std::string id;
    std::string description;
    std::string spokenPrompt;
    std::string spokenPromptCn;
    std::vector<TaskStepObject> involvedObjects;
    std::vector<TaskStep> steps;
};

struct TaskTemplate
{
    std::string schemaVersion;
    std::string sourcePath;
    std::string sceneFolder;
    std::string sceneId;
    std::string sceneName;
    std::string sceneNameCn;
    std::string sceneDescription;
    std::vector<SceneObject> sceneObjects;
    TaskDefinition task;
    std::string constraintsNotes;
};

class TaskLoader
{
public:
    explicit TaskLoader(Logger &logger);

    // 加载单个任务文件
    std::optional<TaskTemplate> loadTaskFile(const std::string &path);

    // 加载指定场景目录下的所有任务
    std::vector<TaskTemplate> loadSceneTasks(const std::string &sceneDir);

    // 加载根目录下所有场景的任务
    std::vector<TaskTemplate> loadAllTasks(const std::string &rootDir);
    const std::unordered_map<std::string, SceneMeta> &sceneMetadata() const { return _sceneMeta; }

private:
    Logger &_logger;
    std::unordered_map<std::string, SceneMeta> _sceneMeta;

    TaskStepObject parseStepObject(const QJsonObject &obj);
    SceneObject parseSceneObject(const QJsonObject &obj);
    TaskStep parseTaskStep(const QJsonObject &obj);
    void loadSceneMetadata(const std::string &rootDir);
};
