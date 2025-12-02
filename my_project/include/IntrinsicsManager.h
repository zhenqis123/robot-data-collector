#pragma once

#include <array>
#include <optional>
#include <string>
#include <vector>

struct StreamIntrinsics
{
    int width = 0;
    int height = 0;
    float fx = 0.f;
    float fy = 0.f;
    float cx = 0.f;
    float cy = 0.f;
    std::vector<float> coeffs;
    std::string stream;
};

class IntrinsicsManager
{
public:
    static IntrinsicsManager &instance();

    std::optional<StreamIntrinsics> find(const std::string &identifier,
                                         const std::string &stream,
                                         int width,
                                         int height) const;
    void save(const std::string &identifier,
              const StreamIntrinsics &intrinsics);

private:
    IntrinsicsManager();
    void load();
    void persist() const;

    struct Entry
    {
        std::string identifier;
        StreamIntrinsics intr;
    };

    std::vector<Entry> _entries;
};
