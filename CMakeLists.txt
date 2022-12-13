# cmake needs this line
cmake_minimum_required(VERSION 3.1)

# Define project name
project(restruction)

# Find OpenCV, you may need to set OpenCV_DIR variable
# to the absolute path to the directory containing OpenCVConfig.cmake file
# via the command line or GUI
find_package(OpenCV REQUIRED)

# C++11设置
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# 头文件目录搜索路径，需要加上${OpenCV_INCLUDE_DIRS}，即所需的库文件
include_directories(include ${OpenCV_INCLUDE_DIRS})
# 生成库文件(.a)，需要正确的路径名称
# add_library(denoise include/denoise.h src/denoise.cpp)
# add_library(noise include/noise.h src/noise.cpp)

# 生成可执行文件
add_executable(main src/main.cpp)

# 连接库文件
# target_link_libraries(main PRIVATE ${OpenCV_LIBS} denoise noise)
target_link_libraries(main PRIVATE ${OpenCV_LIBS})
# 设置可执行文件生成路径
set(EXECUTABLE_OUTPUT_PATH "../out")