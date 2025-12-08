#include <gtest/gtest.h>

#include <filesystem>

#include <opencv2/core.hpp>

#include "DataStorage.h"
#include "Logger.h"

static std::string storageDir()
{
    auto dir = (std::filesystem::temp_directory_path() / "data_storage_tests").string();
    std::filesystem::create_directories(dir);
    return dir;
}

TEST(DataStorageTest, WritesMetadataAndEventsOnly)
{
    Logger logger(storageDir());
    DataStorage storage(storageDir(), logger);
    auto dir = storageDir();
    storage.beginRecording("UnitTest", "Subject", dir, {});
    storage.logEvent("test_event");
    DataStorage::AnnotationEntry entry;
    entry.source = "test";
    entry.sceneId = "scene";
    entry.taskId = "task";
    entry.templatePath = "tpl";
    entry.templateVersion = "v1";
    entry.state = "state";
    entry.timestampMs = 123;
    storage.logAnnotation(entry);
    storage.endRecording();

    auto base = std::filesystem::path(dir);
    EXPECT_TRUE(std::filesystem::exists(base / "meta.json"));
    EXPECT_TRUE(std::filesystem::exists(base / "events.jsonl"));
    EXPECT_TRUE(std::filesystem::exists(base / "annotations.jsonl"));
}
