find_package(GTest REQUIRED)
find_package(CURL REQUIRED)

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)

add_executable(unit_tests ${TEST_SOURCES}
    ${PROJECT_SOURCE_DIR}/src/CameraInterface.cpp
    ${PROJECT_SOURCE_DIR}/src/DataCapture.cpp
    ${PROJECT_SOURCE_DIR}/src/DataStorage.cpp
    ${PROJECT_SOURCE_DIR}/src/NetworkDevice.cpp
    ${PROJECT_SOURCE_DIR}/src/Preview.cpp
    ${PROJECT_SOURCE_DIR}/src/ConfigManager.cpp
    ${PROJECT_SOURCE_DIR}/src/Logger.cpp
    ${PROJECT_SOURCE_DIR}/src/CameraStats.cpp
    ${PROJECT_SOURCE_DIR}/src/IntrinsicsManager.cpp
    ${PROJECT_SOURCE_DIR}/src/ArucoTracker.cpp
    ${PROJECT_SOURCE_DIR}/src/TacGlove.cpp
)

target_include_directories(unit_tests PRIVATE
    ${PROJECT_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${REALSENSE2_INCLUDE_DIR}
)

target_link_libraries(unit_tests
    PRIVATE
        GTest::GTest
        GTest::Main
        ${QT_LIBS}
        ${OpenCV_LIBS}
        ${REALSENSE2_LIBRARY}
        Threads::Threads
        CURL::libcurl
        rt  # POSIX shared memory and semaphores
)
target_compile_definitions(unit_tests
    PRIVATE APP_CONFIG_PATH="${PROJECT_SOURCE_DIR}/resources/config.json"
            APP_LOG_DIR="${PROJECT_SOURCE_DIR}/resources/logs"
            APP_INTRINSICS_PATH="${PROJECT_SOURCE_DIR}/resources/intrinsics.json"
)

add_test(NAME unit_tests COMMAND unit_tests)
