find_package(CURL REQUIRED)

add_executable(gemini_demo gemini_demo.cpp)
target_link_libraries(gemini_demo PRIVATE CURL::libcurl)

add_executable(make_sound make_sound.cpp)
target_link_libraries(make_sound PRIVATE Qt5::Core Qt5::Multimedia)

# LocalTacGlove IPC 测试程序
add_executable(test_local_tacglove
    test_local_tacglove.cpp
    ${PROJECT_SOURCE_DIR}/src/TacGlove.cpp
    ${PROJECT_SOURCE_DIR}/src/Logger.cpp
)
target_include_directories(test_local_tacglove PRIVATE ${PROJECT_INCLUDE_DIR})
target_link_libraries(test_local_tacglove PRIVATE Threads::Threads rt)
