cmake_minimum_required(VERSION 3.0)

find_package(Boost COMPONENTS system thread timer date_time chrono filesystem)
find_package(OpenCV COMPONENTS core)

set(CORE_FILES
  src/core/alignment.h
  src/core/checkpoint.h
  src/core/colorspace.h
  src/core/colorspace_opencv.h
  src/core/compiler_macros.h
  src/core/file_helper.h
  src/core/file_helper.cpp
  src/core/image_dimensions.h
  src/core/indexed_storage.h
  src/core/manager_worker_synchronization.h
  src/core/math.h
  src/core/shared_batch_buffer.h
  src/core/shared_bounded_queue.h
  src/core/shared_counter.h
  src/core/shared_speed_stats.h
  src/core/shared_value.h
  src/core/signal.h
  src/core/speed_stats.h
  src/core/streaming_module_interface.h
  src/core/thread_helper.h
  src/core/timer.h
  src/core/two_pass_checkpoint_with_thread_assignment.h
)

set(CORE_SRC_FILES
  src/core/src/shared_buffer_base.h
)

source_group(${CMH_MODULE_NAME} FILES ${CORE_FILES})
source_group(${CMH_MODULE_NAME}\\src FILES ${CORE_SRC_FILES})

add_library(${CMH_MODULE_NAME} ${CORE_FILES} ${CORE_SRC_FILES})

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${Boost_INCLUDE_DIR}
  ${OpenCV_INCLUDE_DIRS}
)

cmh_target_link_libraries(
  ${Boost_LIBRARIES}
)

cmh_end_module()
