cmake_minimum_required(VERSION 3.0)

project(streaming_connected_component_discovery)

set(ROOT ${CMAKE_CURRENT_LIST_DIR})

find_package(cmake_helper REQUIRED PATHS ${ROOT}/cmake_helper)
# The output directories must be set before including other packages.
# For example, cuda_sift will set the location of server_siftgpu.exe
cmh_set_output_directories()

# Libraries
find_package(assert PATHS ${ROOT}/lib_assert)
find_package(core PATHS ${ROOT}/lib_core)
find_package(dataset PATHS ${ROOT}/lib_dataset)
find_package(streaming_balanced_pcdb_file_reader PATHS ${ROOT}/lib_streaming_balanced_pcdb_file_reader)
find_package(streaming_image_loader PATHS ${ROOT}/lib_streaming_image_loader)
find_package(streaming_cuda_sift PATHS ${ROOT}/lib_streaming_cuda_sift)
find_package(streaming_visual_words PATHS ${ROOT}/lib_streaming_visual_words)
find_package(streaming_sift_file_writer PATHS ${ROOT}/lib_streaming_sift_file_writer)
find_package(descriptor_matcher PATHS ${ROOT}/lib_descriptor_matcher)
find_package(estimator PATHS ${ROOT}/lib_estimator)
find_package(google_breakpad_wrapper PATHS ${ROOT}/lib_google_breakpad_wrapper)
find_package(main_module PATHS ${ROOT}/lib_main_module)
find_package(feature_database PATHS ${ROOT}/lib_feature_database)

# Executables
find_package(create_empty_feature_database PATHS ${ROOT}/exe_create_empty_feature_database)
find_package(densification PATHS ${ROOT}/exe_densification)
find_package(list_cuda_devices PATHS ${ROOT}/exe_list_cuda_devices)
find_package(vocab_tree_learn PATHS ${ROOT}/exe_vocab_tree_learn)

# 3rd-Party Modules
find_package(sqlite_shell PATHS ${ROOT}/3rd_party/sqlite/shell)
find_package(Boost COMPONENTS log log_setup)

add_executable(${PROJECT_NAME}
  ${ROOT}/exe_streaming_connected_component_discovery/streaming_main_module.h
  ${ROOT}/exe_streaming_connected_component_discovery/streaming_main_module.cpp
  ${ROOT}/exe_streaming_connected_component_discovery/main_module_thread_handler.h
  ${ROOT}/exe_streaming_connected_component_discovery/main_module_thread_handler.cpp
  ${ROOT}/exe_streaming_connected_component_discovery/image_cache_entry.h
  ${ROOT}/exe_streaming_connected_component_discovery/image_cache_entry.cpp
  ${ROOT}/exe_streaming_connected_component_discovery/streaming_state.h
  ${ROOT}/exe_streaming_connected_component_discovery/batch_match_task.h
  ${ROOT}/exe_streaming_connected_component_discovery/geometric_verification_result.h
  ${ROOT}/exe_streaming_connected_component_discovery/logger.h
  ${ROOT}/exe_streaming_connected_component_discovery/logger.cpp
  ${ROOT}/exe_streaming_connected_component_discovery/main.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  ${Boost_INCLUDE_DIR}
)

target_link_libraries(${PROJECT_NAME} PUBLIC
  ${Boost_LIBRARIES}
)

cmh_link_modules(${PROJECT_NAME})
