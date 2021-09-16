cmake_minimum_required(VERSION 3.0)
project(facedet)

find_package(OpenCV 3.2.0 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#这里链接的时编译后的ncnn的include和lib文件夹，根据自己的路径去更改
include_directories(/meidia/ExtA/envs/MNN/build/install/usr/local/include)
link_directories(/meidia/ExtA/envs/MNN/build/install/usr/local/lib)
link_directories(/meidia/ExtA/envs/MNN/build/source/backend/cuda)

include_directories(./inc)
aux_source_directory(./src DIR_SRCS)

set(CMAKE_CXX_STANDARD 11)
#set(CMAKE_BUILD_TYPE Debug)

add_executable(facedet main.cpp ${DIR_SRCS})
target_link_libraries(facedet ${OpenCV_LIBS} libMNN.so libMNN_Cuda_Main.so)