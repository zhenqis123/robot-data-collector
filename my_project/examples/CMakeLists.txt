find_package(CURL REQUIRED)

add_executable(gemini_demo gemini_demo.cpp)
target_link_libraries(gemini_demo PRIVATE CURL::libcurl)

add_executable(make_sound make_sound.cpp)
target_link_libraries(make_sound PRIVATE Qt5::Core Qt5::Multimedia)
