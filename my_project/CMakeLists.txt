cmake_minimum_required(VERSION 3.16)
project(DataCollectorApp VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTORCC ON)

find_package(OpenCV REQUIRED)

find_package(Qt5 COMPONENTS Widgets Core Network Multimedia MultimediaWidgets Concurrent REQUIRED)
set(QT_LIBS Qt5::Widgets Qt5::Core Qt5::Network Qt5::Multimedia Qt5::MultimediaWidgets Qt5::Concurrent)

find_path(REALSENSE2_INCLUDE_DIR librealsense2/rs.hpp
    PATHS /usr/include /usr/local/include
)
find_library(REALSENSE2_LIBRARY realsense2
    PATHS /usr/lib /usr/local/lib /usr/lib/x86_64-linux-gnu
)
if(NOT REALSENSE2_INCLUDE_DIR OR NOT REALSENSE2_LIBRARY)
    message(FATAL_ERROR "librealsense2 headers or library not found.")
endif()

find_package(Threads REQUIRED)

set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${PROJECT_INCLUDE_DIR})

# AprilTag (optional fiducial detector)
set(APRILTAG_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/apriltag)
if(NOT EXISTS ${APRILTAG_SOURCE_DIR}/CMakeLists.txt)
    message(FATAL_ERROR "AprilTag source not found. Expected at ../third_party/apriltag")
endif()
set(BUILD_EXAMPLES OFF CACHE BOOL "Disable AprilTag examples" FORCE)
add_subdirectory(${APRILTAG_SOURCE_DIR} ${CMAKE_CURRENT_BINARY_DIR}/apriltag)

file(GLOB PROJECT_HEADERS CONFIGURE_DEPENDS
    ${PROJECT_INCLUDE_DIR}/*.h)

set(PROJECT_SOURCES
    src/main.cpp
    src/MainWindow.cpp
    src/CameraInterface.cpp
    src/DataCapture.cpp
    src/DataStorage.cpp
    src/Preview.cpp
    src/ConfigManager.cpp
    src/Logger.cpp
    src/CameraStats.cpp
    src/IntrinsicsManager.cpp
    src/ArucoTracker.cpp
    src/TaskLoader.cpp
    src/TaskStateMachine.cpp
    src/AudioPromptPlayer.cpp
    src/NetworkDevice.cpp
)

add_executable(${PROJECT_NAME}
    ${PROJECT_SOURCES}
    ${PROJECT_HEADERS}
)

target_link_libraries(${PROJECT_NAME}
    PRIVATE
        ${QT_LIBS}
        ${OpenCV_LIBS}
        ${REALSENSE2_LIBRARY}
        apriltag
        Threads::Threads
)
target_include_directories(${PROJECT_NAME}
    PRIVATE
        ${REALSENSE2_INCLUDE_DIR}
        ${APRILTAG_SOURCE_DIR}
)
target_compile_definitions(${PROJECT_NAME} PRIVATE USE_PIPER_AUDIO=0)

add_executable(rs_test src/test_realsense.cpp)
target_link_libraries(rs_test
    PRIVATE
        ${REALSENSE2_LIBRARY}
        ${OpenCV_LIBS}
        Threads::Threads
)
target_include_directories(rs_test PRIVATE ${REALSENSE2_INCLUDE_DIR})
target_compile_definitions(rs_test
    PRIVATE APP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/config.json"
            APP_LOG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources/logs"
            APP_INTRINSICS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/intrinsics.json"
)
target_compile_definitions(${PROJECT_NAME}
    PRIVATE APP_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/config.json"
            APP_LOG_DIR="${CMAKE_CURRENT_SOURCE_DIR}/resources/logs"
            APP_INTRINSICS_PATH="${CMAKE_CURRENT_SOURCE_DIR}/resources/intrinsics.json"
)

add_subdirectory(examples)

enable_testing()

add_subdirectory(tests)
