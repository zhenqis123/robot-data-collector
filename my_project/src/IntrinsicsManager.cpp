#include "IntrinsicsManager.h"

#include <filesystem>

#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>

IntrinsicsManager::IntrinsicsManager()
{
    load();
}

IntrinsicsManager &IntrinsicsManager::instance()
{
    static IntrinsicsManager manager;
    return manager;
}

void IntrinsicsManager::load()
{
    QString path = QString::fromUtf8(APP_INTRINSICS_PATH);
    QFile file(path);
    if (!file.exists())
    {
        QFile create(path);
        if (create.open(QIODevice::WriteOnly))
        {
            QJsonObject root;
            root["cameras"] = QJsonArray();
            create.write(QJsonDocument(root).toJson());
            create.close();
        }
    }
    if (!file.open(QIODevice::ReadOnly))
        return;

    auto doc = QJsonDocument::fromJson(file.readAll());
    file.close();
    if (!doc.isObject())
        return;
    auto cameras = doc.object().value("cameras").toArray();
    for (const auto &entry : cameras)
    {
        auto obj = entry.toObject();
        Entry e;
        e.identifier = obj.value("identifier").toString().toStdString();
        e.intr.stream = obj.value("stream").toString().toStdString();
        e.intr.width = obj.value("width").toInt();
        e.intr.height = obj.value("height").toInt();
        e.intr.fx = obj.value("fx").toDouble();
        e.intr.fy = obj.value("fy").toDouble();
        e.intr.cx = obj.value("cx").toDouble();
        e.intr.cy = obj.value("cy").toDouble();
        auto coeffArray = obj.value("coeffs").toArray();
        for (const auto &coeff : coeffArray)
            e.intr.coeffs.push_back(coeff.toDouble());
        _entries.push_back(std::move(e));
    }
}

void IntrinsicsManager::persist() const
{
    QJsonArray cameras;
    for (const auto &entry : _entries)
    {
        QJsonObject obj;
        obj["identifier"] = QString::fromStdString(entry.identifier);
        obj["stream"] = QString::fromStdString(entry.intr.stream);
        obj["width"] = entry.intr.width;
        obj["height"] = entry.intr.height;
        obj["fx"] = entry.intr.fx;
        obj["fy"] = entry.intr.fy;
        obj["cx"] = entry.intr.cx;
        obj["cy"] = entry.intr.cy;
        QJsonArray coeffs;
        for (float c : entry.intr.coeffs)
            coeffs.append(c);
        obj["coeffs"] = coeffs;
        cameras.append(obj);
    }
    QJsonObject root;
    root["cameras"] = cameras;
    QFile file(QString::fromUtf8(APP_INTRINSICS_PATH));
    if (file.open(QIODevice::WriteOnly))
    {
        file.write(QJsonDocument(root).toJson());
        file.close();
    }
}

std::optional<StreamIntrinsics> IntrinsicsManager::find(const std::string &identifier,
                                                        const std::string &stream,
                                                        int width,
                                                        int height) const
{
    for (const auto &entry : _entries)
    {
        if (entry.identifier == identifier && entry.intr.stream == stream &&
            entry.intr.width == width && entry.intr.height == height)
        {
            return entry.intr;
        }
    }
    return std::nullopt;
}

void IntrinsicsManager::save(const std::string &identifier,
                             const StreamIntrinsics &intrinsics)
{
    for (auto &entry : _entries)
    {
        if (entry.identifier == identifier &&
            entry.intr.stream == intrinsics.stream &&
            entry.intr.width == intrinsics.width &&
            entry.intr.height == intrinsics.height)
        {
            entry.intr = intrinsics;
            persist();
            return;
        }
    }
    Entry e;
    e.identifier = identifier;
    e.intr = intrinsics;
    _entries.push_back(e);
    persist();
}
