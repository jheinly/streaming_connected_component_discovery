cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

cmh_set_as_third_party_module()

if(MSVC)
  message(STATUS "Microsoft compiler detected, using msinttypes for inttypes.h")
  cmh_new_module_with_dependencies()
else()
  message(STATUS "Non-Microsoft compiler detected, using the system-provided inttypes.h")
endif()
