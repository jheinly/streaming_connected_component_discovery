cmake_minimum_required(VERSION 3.0)

find_package(OpenGL)
find_package(CUDA)

# If using Windows, use the freeglut library provided in the CUDA SDK.
if(NOT WIN32)
  find_package(GLUT)
endif()

# Detect if this is a 64-bit build.
cmh_test_if_64_bit(64BIT)

if(WIN32)
  if(64BIT)
    set(GLEW_LIBRARY_NAME "glew64" CACHE STRING "The name of the GLEW library on your system.")
  else()
    set(GLEW_LIBRARY_NAME "glew32" CACHE STRING "The name of the GLEW library on your system.")
  endif()
elseif(APPLE)
  set(GLEW_LIBRARY_NAME "GLEW" CACHE STRING "The name of the GLEW library on your system.")
else()
  set(GLEW_LIBRARY_NAME "glew" CACHE STRING "The name of the GLEW library on your system.")
endif()
set_property(CACHE GLEW_LIBRARY_NAME PROPERTY STRINGS glew glew32 glew64 GLEW)

add_definitions(
  -DCUDA_SIFTGPU_ENABLED
  -DSERVER_SIFTGPU_ENABLED
  -DSIFT_GPU_BIN_RELEASE_FOLDER="${CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE}/"
)

include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${CUDA_SDK_ROOT_DIR}/common/inc
)

if(NOT WIN32)
  include_directories(${GLUT_INCLUDE_DIR})
endif()

cmh_add_cuda_library_module(
  src/sift_gpu/CLTexImage.h
  src/sift_gpu/CLTexImage.cpp
  src/sift_gpu/CuTexImage.h
  src/sift_gpu/CuTexImage.cpp
  src/sift_gpu/FrameBufferObject.h
  src/sift_gpu/FrameBufferObject.cpp
  src/sift_gpu/GlobalUtil.h
  src/sift_gpu/GlobalUtil.cpp
  src/sift_gpu/GLTexImage.h
  src/sift_gpu/GLTexImage.cpp
  src/sift_gpu/LiteWindow.h
  src/sift_gpu/ProgramCG.h
  src/sift_gpu/ProgramCG.cpp
  src/sift_gpu/ProgramCL.h
  src/sift_gpu/ProgramCL.cpp
  src/sift_gpu/ProgramCU.h
  src/sift_gpu/ProgramCU.cu
  src/sift_gpu/ProgramGLSL.h
  src/sift_gpu/ProgramGLSL.cpp
  src/sift_gpu/ProgramGPU.h
  src/sift_gpu/ProgramGPU.cpp
  src/sift_gpu/PyramidCL.h
  src/sift_gpu/PyramidCL.cpp
  src/sift_gpu/PyramidCU.h
  src/sift_gpu/PyramidCU.cpp
  src/sift_gpu/PyramidGL.h
  src/sift_gpu/PyramidGL.cpp
  src/sift_gpu/ServerSiftGPU.h
  src/sift_gpu/ServerSiftGPU.cpp
  src/sift_gpu/ShaderMan.h
  src/sift_gpu/ShaderMan.cpp
  src/sift_gpu/SiftGPU.h
  src/sift_gpu/SiftGPU.cpp
  src/sift_gpu/SiftMatch.h
  src/sift_gpu/SiftMatch.cpp
  src/sift_gpu/SiftMatchCU.h
  src/sift_gpu/SiftMatchCU.cpp
  src/sift_gpu/SiftPyramid.h
  src/sift_gpu/SiftPyramid.cpp
)

set(LINK_DIRECTORY_PATHS "")

if(WIN32)
  list(APPEND LINK_DIRECTORY_PATHS ${CUDA_SDK_ROOT_DIR}/common/lib)
  if(64BIT)
    list(APPEND LINK_DIRECTORY_PATHS ${CUDA_SDK_ROOT_DIR}/common/lib/x64)
  else()
    list(APPEND LINK_DIRECTORY_PATHS ${CUDA_SDK_ROOT_DIR}/common/lib/Win32)
  endif()
else()
  list(APPEND LINK_DIRECTORY_PATHS /usr/local/lib)
  if(APPLE)
    list(APPEND LINK_DIRECTORY_PATHS ${CUDA_SDK_ROOT_DIR}/common/lib/darwin)
  else()
    list(APPEND LINK_DIRECTORY_PATHS ${CUDA_SDK_ROOT_DIR}/common/lib/linux)
  endif()
endif()

find_library(GLEW_LIBRARY_PATH
  NAMES ${GLEW_LIBRARY_NAME}
  PATHS ${LINK_DIRECTORY_PATHS}
)

cmh_target_link_libraries(${GLEW_LIBRARY_PATH} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY})
if(WIN32)
  find_library(GLUT_LIBRARY_PATH
    NAMES freeglut
    PATHS ${LINK_DIRECTORY_PATHS}
  )
  cmh_target_link_libraries(${GLUT_LIBRARY_PATH})
else()
  cmh_target_link_libraries(${GLUT_glut_LIBRARY})
endif()
