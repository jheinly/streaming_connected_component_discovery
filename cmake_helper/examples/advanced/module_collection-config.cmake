cmake_minimum_required(VERSION 3.0)

find_package(cmake_helper REQUIRED)

# Get the name of this module collection.
cmh_get_module_name(CMH_MODULE_COLLECTION_NAME ${CMAKE_CURRENT_LIST_FILE})

if(${CMH_MODULE_COLLECTION_NAME}_FIND_COMPONENTS)
  # Iterate over the modules that were specified as components.
  foreach(COMPONENT ${${CMH_MODULE_COLLECTION_NAME}_FIND_COMPONENTS})
    # The following find_package() command assumes that this module collection file
    # is in the same directory as module folders, where for a given module name,
    # a ${CMAKE_CURRENT_LIST_DIR}/module_name/module_name-config.cmake file exists.
    find_package(${COMPONENT} PATHS ${CMAKE_CURRENT_LIST_DIR}/${COMPONENT})
  endforeach()
  unset(COMPONENT)
else()
  message(WARNING "No dependency modules specified. Please call find_package(${CMH_MODULE_COLLECTION_NAME} COMPONENTS ...) with a list of the dependency modules.")
endif()

unset(CMH_MODULE_COLLECTION_NAME)
