cmake_minimum_required(VERSION 2.8)
project(corr_tf)
find_package(CUDA REQUIRED)
# get tensorflow include dirs, see https://www.tensorflow.org/how_tos/adding_an_op/
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_include())" OUTPUT_VARIABLE TF_INC)
execute_process(COMMAND python -c "import tensorflow; print(tensorflow.sysconfig.get_lib())" OUTPUT_VARIABLE TF_LIB)

set(CMAKE_BUILD_TYPE RelWithDebInfo)
# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-fPIC -std=c++11 ${CMAKE_CXX_FLAGS} -march=native -DGOOGLE_CUDA=1")
set(CUDA_NVCC_FLAGS "-L${TF_LIB} -ltensorflow_framework -I ${TF_INC} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC")

# build the actual operation which can be used directory
include_directories(${TF_INC})
link_libraries("/usr/lib/python2.7/site-packages/tensorflow/libtensorflow_framework.so")
cuda_add_library(correlation SHARED correlation.cc correlation_gpu.cc.cu correlation_param.h)
target_link_libraries(correlation cudart)


# build the gradient operation which is used in correlation_grad.py
# to register it
include_directories(${TF_INC})
link_libraries("/usr/lib/python2.7/site-packages/tensorflow/libtensorflow_framework.so")
#add_library(correlation_grad SHARED correlation_grad.cc)
cuda_add_library(correlation_grad SHARED correlation_grad.cc correlation_grad_gpu.cc.cu)
target_link_libraries(correlation_grad cudart correlation)
