#pragma once
#include "VDGloveDataTypes.h"
#include <string>
#include <iostream>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#endif

// 定义函数指针类型
typedef bool (*FuncUdpOpen)(int index, int port);
typedef void (*FuncUdpClose)(int index);
typedef bool (*FuncUdpSendRequestConnect)(int index, char* ip, unsigned short port);
typedef bool (*FuncUdpRecvMocapData)(int index, char* ip, unsigned short port, MocapData* data);

class VDSDKLoader {
public:
    FuncUdpOpen UdpOpen = nullptr;
    FuncUdpClose UdpClose = nullptr;
    FuncUdpSendRequestConnect UdpSendRequestConnect = nullptr;
    FuncUdpRecvMocapData UdpRecvMocapData = nullptr;

    bool loadLibrary(const std::string& path) {
        if (_handle) return true;
#ifdef _WIN32
        _handle = LoadLibraryA(path.c_str());
#else
        _handle = dlopen(path.c_str(), RTLD_LAZY | RTLD_NODELETE);
        if (!_handle) _handle = dlopen(path.c_str(), RTLD_LAZY);
        // _handle = dlopen(path.c_str(), RTLD_LAZY);
#endif
        if (!_handle) {
            std::cerr << "Failed to load library: " << path << std::endl;
            return false;
        }

        UdpOpen = (FuncUdpOpen)getSymbol("UdpOpen");
        UdpClose = (FuncUdpClose)getSymbol("UdpClose");
        UdpSendRequestConnect = (FuncUdpSendRequestConnect)getSymbol("UdpSendRequestConnect");
        UdpRecvMocapData = (FuncUdpRecvMocapData)getSymbol("UdpRecvMocapData");

        return UdpOpen && UdpClose && UdpSendRequestConnect && UdpRecvMocapData;
    }

    void unload() {
        if (_handle) {
            _handle = nullptr;
        }
    }

    ~VDSDKLoader() { 
        // unload(); 
    }

private:
    void* _handle = nullptr;

    void* getSymbol(const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress((HMODULE)_handle, name);
#else
        return dlsym(_handle, name);
#endif
    }
};