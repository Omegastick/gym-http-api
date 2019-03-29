cmake_minimum_required(VERSION 3.10.0)
project(gym-http-api
    LANGUAGES CXX
    VERSION 1.0.0
    DESCRIPTION "Gym HTTP API"
)

# Project-wide properties
set(CMAKE_CXX_STANDARD 17)

# Dependencies
set(JSON_BuildTests OFF CACHE INTERNAL "Build the unit tests when BUILD_TESTING is enabled.")
add_subdirectory(lib/json)

# Define targets
add_library(gymhttpapi STATIC "gym_binding.cpp")

# Enable all warnings
if(MSVC)
    target_compile_options(gymhttpapi PRIVATE /W4 /WX)
else(MSVC)
    target_compile_options(gymhttpapi PRIVATE -Wall -Wextra -pedantic)
endif(MSVC)  

# Includes
target_include_directories(gymhttpapi PUBLIC include)

# Linking
target_link_libraries(gymhttpapi PRIVATE nlohmann_json::nlohmann_json)