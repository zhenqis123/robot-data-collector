#include "TaskLoader.h"

#include <QDir>
#include <QFile>
#include <QFileInfo>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

#include "Logger.h"

namespace
{
bool hasRequiredTaskFields(const TaskTemplate &task)
{
    if (task.sceneId.empty() || task.task.id.empty())
        return false;
    return true;
}
}

TaskLoader::TaskLoader(Logger &logger)
    : _logger(logger)
{
}

void TaskLoader::loadSceneMetadata(const std::string &rootDir)
{
    _sceneMeta.clear();
    const QString metaPath = QDir(QString::fromStdString(rootDir)).filePath("scenes.json");
    QFile f(metaPath);
    if (!f.exists())
        return;
    if (!f.open(QIODevice::ReadOnly))
    {
        _logger.warn("Failed to open scenes metadata: %s", metaPath.toStdString().c_str());
        return;
    }
    const auto doc = QJsonDocument::fromJson(f.readAll());
    if (!doc.isArray())
    {
        _logger.warn("Scenes metadata malformed (expected array): %s", metaPath.toStdString().c_str());
        return;
    }
    for (const auto &entry : doc.array())
    {
        if (!entry.isObject())
            continue;
        const auto obj = entry.toObject();
        SceneMeta meta;
        meta.id = obj.value("id").toString().toStdString();
        if (meta.id.empty())
            continue;
        meta.name = obj.value("name").toString().toStdString();
        meta.nameCn = obj.value("name_cn").toString().toStdString();
        meta.description = obj.value("description").toString().toStdString();
        _sceneMeta[meta.id] = meta;
    }
}

TaskStepObject TaskLoader::parseStepObject(const QJsonObject &obj)
{
    TaskStepObject result;
    result.objectId = obj.value("object_id").toString().toStdString();
    result.name = obj.value("name").toString().toStdString();
    result.role = obj.value("role").toString().toStdString();
    result.optional = obj.value("optional").toBool(false);
    return result;
}

SceneObject TaskLoader::parseSceneObject(const QJsonObject &obj)
{
    SceneObject result;
    result.objectId = obj.value("object_id").toString().toStdString();
    result.name = obj.value("name").toString().toStdString();
    result.category = obj.value("category").toString().toStdString();
    const auto stateObj = obj.value("state").toObject();
    result.placement = stateObj.value("placement").toString().toStdString();
    result.relative = stateObj.value("relative").toString().toStdString();
    return result;
}

TaskStep TaskLoader::parseTaskStep(const QJsonObject &obj)
{
    TaskStep step;
    step.id = obj.value("id").toString().toStdString();
    step.description = obj.value("description").toString().toStdString();
    step.spokenPrompt = obj.value("spoken_prompt").toString().toStdString();
    step.spokenPromptCn = obj.value("spoken_prompt_cn").toString().toStdString();
    step.videoPath = obj.value("video_path").toString().toStdString();
    const auto involved = obj.value("involved_objects").toArray();
    for (const auto &item : involved)
    {
        if (item.isObject())
            step.involvedObjects.push_back(parseStepObject(item.toObject()));
    }
    return step;
}

std::optional<TaskTemplate> TaskLoader::loadTaskFile(const std::string &path)
{
    QFile file(QString::fromStdString(path));
    if (!file.open(QIODevice::ReadOnly))
    {
        _logger.error("Failed to open task file: %s", path.c_str());
        return std::nullopt;
    }
    const auto data = file.readAll();
    const auto doc = QJsonDocument::fromJson(data);
    if (doc.isNull() || !doc.isObject())
    {
        _logger.error("Invalid JSON in task file: %s", path.c_str());
        return std::nullopt;
    }

    TaskTemplate task;
    task.sourcePath = path;

    const auto root = doc.object();
    task.schemaVersion = root.value("schema_version").toString().toStdString();

    const auto sceneObj = root.value("scene").toObject();
    task.sceneId = sceneObj.value("scene_id").toString().toStdString();
    task.sceneName = sceneObj.value("name").toString().toStdString();
    task.sceneNameCn = sceneObj.value("name_cn").toString().toStdString();
    task.sceneDescription = sceneObj.value("description").toString().toStdString();
    const auto sceneObjects = sceneObj.value("objects").toArray();
    for (const auto &entry : sceneObjects)
    {
        if (entry.isObject())
            task.sceneObjects.push_back(parseSceneObject(entry.toObject()));
    }

    const auto taskObj = root.value("task").toObject();
    task.task.id = taskObj.value("id").toString().toStdString();
    task.task.name = taskObj.value("name").toString().toStdString();
    task.task.nameCn = taskObj.value("name_cn").toString().toStdString();
    task.task.description = taskObj.value("description").toString().toStdString();
    task.task.spokenPrompt = taskObj.value("spoken_prompt").toString().toStdString();
    task.task.spokenPromptCn = taskObj.value("spoken_prompt_cn").toString().toStdString();
    const auto involved = taskObj.value("involved_objects").toArray();
    for (const auto &entry : involved)
    {
        if (entry.isObject())
            task.task.involvedObjects.push_back(parseStepObject(entry.toObject()));
    }

    const auto subtasks = taskObj.value("subtasks").toArray();
    const QString baseDir = QFileInfo(QString::fromStdString(path)).absolutePath();
    if (!subtasks.isEmpty())
    {
        for (const auto &entry : subtasks)
        {
            if (!entry.isObject())
                continue;
            Subtask st;
            const auto stObj = entry.toObject();
            st.id = stObj.value("id").toString().toStdString();
            st.description = stObj.value("description").toString().toStdString();
            st.spokenPrompt = stObj.value("spoken_prompt").toString().toStdString();
            st.spokenPromptCn = stObj.value("spoken_prompt_cn").toString().toStdString();
            const auto stInvolved = stObj.value("involved_objects").toArray();
            for (const auto &io : stInvolved)
                if (io.isObject())
                    st.involvedObjects.push_back(parseStepObject(io.toObject()));
            const auto stSteps = stObj.value("steps").toArray();
            for (const auto &s : stSteps)
            {
                if (s.isObject())
                {
                    auto parsed = parseTaskStep(s.toObject());
                    if (!parsed.videoPath.empty())
                    {
                        QFileInfo fi(parsed.videoPath.c_str());
                        if (fi.isRelative())
                            fi.setFile(baseDir + "/" + fi.filePath());
                        parsed.videoPath = fi.absoluteFilePath().toStdString();
                    }
                    st.steps.push_back(std::move(parsed));
                }
            }
            task.task.subtasks.push_back(std::move(st));
        }
    }
    else
    {
        const auto steps = taskObj.value("steps").toArray();
        for (const auto &entry : steps)
        {
            if (entry.isObject())
            {
                auto parsed = parseTaskStep(entry.toObject());
                if (!parsed.videoPath.empty())
                {
                    QFileInfo fi(parsed.videoPath.c_str());
                    if (fi.isRelative())
                        fi.setFile(baseDir + "/" + fi.filePath());
                    parsed.videoPath = fi.absoluteFilePath().toStdString();
                }
                task.task.steps.push_back(std::move(parsed));
            }
        }
    }

    const auto constraintsObj = root.value("constraints").toObject();
    task.constraintsNotes = constraintsObj.value("notes").toString().toStdString();

    auto it = _sceneMeta.find(task.sceneId);
    if (it != _sceneMeta.end())
    {
        if (task.sceneName.empty())
            task.sceneName = it->second.name;
        if (task.sceneNameCn.empty())
            task.sceneNameCn = it->second.nameCn;
        if (task.sceneDescription.empty())
            task.sceneDescription = it->second.description;
    }

    if (!hasRequiredTaskFields(task))
    {
        _logger.error("Task file missing required fields: %s", path.c_str());
        return std::nullopt;
    }

    return task;
}

std::vector<TaskTemplate> TaskLoader::loadSceneTasks(const std::string &sceneDir)
{
    std::vector<TaskTemplate> tasks;
    QDir dir(QString::fromStdString(sceneDir));
    if (!dir.exists())
    {
        _logger.warn("Scene directory not found: %s", sceneDir.c_str());
        return tasks;
    }
    const auto files = dir.entryList(QStringList() << "*.json", QDir::Files);
    for (const auto &fileName : files)
    {
        const auto fullPath = dir.absoluteFilePath(fileName).toStdString();
        auto task = loadTaskFile(fullPath);
        if (task.has_value())
        {
            task->sceneFolder = dir.dirName().toStdString();
            tasks.push_back(*task);
        }
    }
    return tasks;
}

std::vector<TaskTemplate> TaskLoader::loadAllTasks(const std::string &rootDir)
{
    std::vector<TaskTemplate> tasks;
    QDir root(QString::fromStdString(rootDir));
    if (!root.exists())
    {
        _logger.warn("Tasks root directory not found: %s", rootDir.c_str());
        return tasks;
    }

    loadSceneMetadata(rootDir);
    if (!_sceneMeta.empty())
    {
        for (const auto &pair : _sceneMeta)
        {
            const auto scenePath = root.absoluteFilePath(QString::fromStdString(pair.first)).toStdString();
            const auto sceneTasks = loadSceneTasks(scenePath);
            tasks.insert(tasks.end(), sceneTasks.begin(), sceneTasks.end());
        }
    }
    else
    {
        const auto sceneDirs = root.entryList(QDir::Dirs | QDir::NoDotAndDotDot);
        for (const auto &scene : sceneDirs)
        {
            const auto scenePath = root.absoluteFilePath(scene).toStdString();
            const auto sceneTasks = loadSceneTasks(scenePath);
            tasks.insert(tasks.end(), sceneTasks.begin(), sceneTasks.end());
        }
    }
    return tasks;
}
