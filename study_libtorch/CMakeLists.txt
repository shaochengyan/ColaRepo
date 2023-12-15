cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(example-app)

# find torch lib
# set(CMAKE_PREFIX_PATH "/home/cola/anaconda3/envs/cppcuda/lib/python3.8/site-packages/torch/share/cmake")
find_package(Torch REQUIRED)
# 设置全局内置变量 for compile
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

add_executable(example-app example-app.cpp)
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 17)

add_executable(end_to_end end_to_end.cpp)
target_link_libraries(end_to_end "${TORCH_LIBRARIES}")


# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET example-app
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:example-app>)
endif (MSVC)