cmake_minimum_required(VERSION 3.0)

find_package(Boost COMPONENTS filesystem system)

cmh_add_executable_module(create_empty_feature_database.cpp)

cmh_target_include_directories(${Boost_INCLUDE_DIR})

cmh_target_link_libraries(${Boost_LIBRARIES})
