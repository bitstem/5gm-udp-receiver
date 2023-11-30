cmake_minimum_required(VERSION 3.16)
project(5gm-udp-receier)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_FLAGS_DEBUG_INIT "-Wall -Wextra -Werror -g3")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-Wall -O3")

find_package(Boost REQUIRED)

add_executable(5gm-udp-receiver main.cpp)

target_link_libraries( 5gm-udp-receiver
    LINK_PUBLIC
    pthread
    boost_system
)
