cmake_minimum_required(VERSION 3.5)
project(Task1)

find_package(Ceres REQUIRED)

add_executable(task1 task1.cpp)

target_link_libraries(task1
    ${CERES_LIBRARIES}
)
