#pragma once

#include <optional>
#include <string>
#include <vector>

#include <QJsonObject>

class Logger;

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
    std::vector<TaskStepObject> involvedObjects;
};

struct TaskDefinition
{
    std::string id;
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
    std::string sceneId;
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

private:
    Logger &_logger;

    TaskStepObject parseStepObject(const QJsonObject &obj);
    SceneObject parseSceneObject(const QJsonObject &obj);
    TaskStep parseTaskStep(const QJsonObject &obj);
};
