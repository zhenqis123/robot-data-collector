#include <iostream>
#include <string>
#include <cstdlib>      // getenv
#include <fstream>
#include <iterator>
#include <vector>
#include <curl/curl.h>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

// 用于接收 HTTP 响应的回调
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, void* userp) {
    size_t totalSize = size * nmemb;
    std::string* s = static_cast<std::string*>(userp);
    s->append(static_cast<char*>(contents), totalSize);
    return totalSize;
}

// Minimal Base64 encoder for binary buffers
static std::string base64Encode(const std::vector<unsigned char>& data) {
    static const char table[] =
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    std::string encoded;
    size_t i = 0;
    while (i + 2 < data.size()) {
        unsigned triple = (data[i] << 16) | (data[i + 1] << 8) | data[i + 2];
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        encoded.push_back(table[(triple >> 12) & 0x3F]);
        encoded.push_back(table[(triple >> 6) & 0x3F]);
        encoded.push_back(table[triple & 0x3F]);
        i += 3;
    }
    if (i < data.size()) {
        unsigned triple = data[i] << 16;
        encoded.push_back(table[(triple >> 18) & 0x3F]);
        if (i + 1 < data.size()) {
            triple |= data[i + 1] << 8;
            encoded.push_back(table[(triple >> 12) & 0x3F]);
            encoded.push_back(table[(triple >> 6) & 0x3F]);
            encoded.push_back('=');
        } else {
            encoded.push_back(table[(triple >> 12) & 0x3F]);
            encoded.push_back('=');
            encoded.push_back('=');
        }
    }
    return encoded;
}

static std::vector<unsigned char> readBinaryFile(const std::string& path) {
    std::ifstream file(path, std::ios::binary);
    if (!file) {
        return {};
    }
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(file)),
                                      std::istreambuf_iterator<char>());
}

static std::string readFileToString(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return {};
    }
    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

int main() {
    // 从环境变量读取 API Key，未设置则使用提供的 Key
    const char* envKey = std::getenv("GPT_API_KEY");
    std::string apiKey = envKey ? std::string(envKey) : "sk-aPCXdOLYxqjFbNKwvh4JeqKbTLpTs30mY2MvjvKcoK78uDzm";
    if (apiKey.empty()) {
        std::cerr << "API Key 未设置，请先设置环境变量 GPT_API_KEY" << std::endl;
        return 1;
    }

    // 目标 API：OpenAI 兼容多模态接口
    std::string url = "https://api.gptoai.top/v1/chat/completions";

    // 读取并编码本地图片（desk.jpg）
    // Default to current directory or relative path
    const std::string imagePath = "desk.jpg";
    auto imageData = readBinaryFile(imagePath);
    if (imageData.empty()) {
        std::cerr << "无法读取图片文件: " << imagePath << " (Please ensure desk.jpg is in the current directory)" << std::endl;
        return 1;
    }
    std::string imgBase64 = base64Encode(imageData);

    // 从 prompt 文件加载 system prompt（对齐任务模板结构）
    // Assuming run from build directory, resources are in ../resources
    const std::string promptPath = "../resources/prompts/vlm_task_prompt.txt";
    std::string promptText = readFileToString(promptPath);
    if (promptText.empty()) {
        promptText = "You are a vision-language planner. Output JSON only.";
    }

    // 组装 JSON 请求体（多模态文本 + 图片）
    json j;
    j["model"] = "gemini-2.5-flash";
    j["messages"] = json::array({
        json{
            {"role", "system"},
            {"content", promptText}
        },
        json{
            {"role", "user"},
            {"content", json::array({
                json{{"type", "text"}, {"text", "Generate a task template for this image."}},
                json{{"type", "image_url"}, {"image_url", json{{"url", "data:image/jpeg;base64," + imgBase64}}}}
            })}
        }
    });
    j["response_format"] = json{{"type", "json_object"}};
    std::string requestBody = j.dump();  // 序列化为字符串

    CURL* curl = curl_easy_init();
    if (!curl) {
        std::cerr << "初始化 CURL 失败" << std::endl;
        return 1;
    }

    std::string responseString;  // 用来存返回结果
    struct curl_slist* headers = nullptr;

    // 设置 Header：Authorization + Content-Type
    std::string authHeader = std::string("Authorization: Bearer ") + apiKey;
    headers = curl_slist_append(headers, authHeader.c_str());
    headers = curl_slist_append(headers, "Content-Type: application/json");
    headers = curl_slist_append(headers, "Accept: application/json");

    curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, requestBody.c_str());
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, requestBody.size());

    // 设置回调，接收 Body
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, WriteCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &responseString);

    // 发送请求
    CURLcode res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
        std::cerr << "请求失败，CURL error: " << curl_easy_strerror(res) << std::endl;
    } else {
        long httpCode = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &httpCode);
        if (httpCode == 200) {
            std::cout << "响应：" << std::endl;
            std::cout << responseString << std::endl;

            // 如果你想解析出模型回复：
            try {
                auto respJson = json::parse(responseString);
                auto content = respJson["choices"][0]["message"]["content"];

                const std::string outPath = "../resources/logs/vlm_output.json";
                json parsedOutput;

                if (content.is_array()) {
                    std::string textCombined;
                    for (const auto& part : content) {
                        if (part.contains("type") && part["type"] == "text" && part.contains("text")) {
                            textCombined += part["text"].get<std::string>();
                        }
                    }
                    std::cout << "模型回复: " << textCombined << std::endl;
                    parsedOutput = json::parse(textCombined);
                } else if (content.is_string()) {
                    std::cout << "模型回复: " << content.get<std::string>() << std::endl;
                    parsedOutput = json::parse(content.get<std::string>());
                } else if (content.is_object()) {
                    std::cout << "模型回复: " << content.dump() << std::endl;
                    parsedOutput = content;
                } else {
                    std::cerr << "未知的 content 类型，无法解析" << std::endl;
                }

                if (!parsedOutput.is_null()) {
                    std::ofstream out(outPath);
                    if (out.is_open()) {
                        out << parsedOutput.dump(2);
                        out.close();
                        std::cout << "已写入文件: my_project/resources/logs/vlm_output.json" << std::endl;
                    } else {
                        std::cerr << "写文件失败: my_project/resources/logs/vlm_output.json" << std::endl;
                    }
                }
            } catch (std::exception& e) {
                std::cerr << "解析 JSON 失败: " << e.what() << std::endl;
            }
        } else {
            std::cerr << "HTTP 状态码: " << httpCode << std::endl;
            std::cerr << "响应内容: " << responseString << std::endl;
        }
    }

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    return 0;
}
