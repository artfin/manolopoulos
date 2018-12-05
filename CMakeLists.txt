cmake_minimum_required(VERSION 3.5)
project(log_derivative)

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

set(CMAKE_CXX_FLAGS "-std=c++11 -Wall")

add_executable(log_derivative log_derivative.cpp log.cpp)
target_link_libraries(log_derivative Eigen3::Eigen)
