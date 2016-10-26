cmake_minimum_required(VERSION 3.0)

find_package(OpenCV COMPONENTS core calib3d)

set(SOURCE_FILES
  src/estimator/essential_matrix_estimator.h
  src/estimator/essential_matrix_estimator.cpp
  src/estimator/check_for_degenerate_match.h
)

set(V3D_CODE_FILES
  src/estimator/v3d_code/DataContainer.h
  src/estimator/v3d_code/essentialMat.h
  src/estimator/v3d_code/essentialMat.cpp
  src/estimator/v3d_code/Features.h
  src/estimator/v3d_code/Matcher.h
  src/estimator/v3d_code/Matcher.cpp
  src/estimator/v3d_code/ocv_arrsac.h
  src/estimator/v3d_code/ocv_arrsac.cpp
  src/estimator/v3d_code/ocv_arrsac_E_5pts.h
  src/estimator/v3d_code/ocv_arrsac_E_5pts.cpp
  src/estimator/v3d_code/OCV_utils.h
  src/estimator/v3d_code/OCV_utils.cpp
  src/estimator/v3d_code/triangulation.h
  src/estimator/v3d_code/triangulation.cpp
  src/estimator/v3d_code/v3d_cameramatrix.h
  src/estimator/v3d_code/v3d_distortion.h
  src/estimator/v3d_code/v3d_linear.h
  src/estimator/v3d_code/v3d_linear_lu.h
  src/estimator/v3d_code/v3d_linear_tnt.h
  src/estimator/v3d_code/v3d_linearbase.h
  src/estimator/v3d_code/v3d_mathutilities.h
  src/estimator/v3d_code/v3d_mathutilities.cpp
  src/estimator/v3d_code/v3d_mviewutilities.h
  src/estimator/v3d_code/v3d_mviewutilities.cpp
  src/estimator/v3d_code/v3d_ocv.h
  src/estimator/v3d_code/v3d_ocv.cpp
  src/estimator/v3d_code/v3d_optimization.h
  src/estimator/v3d_code/v3d_optimization.cpp
  src/estimator/v3d_code/v3d_poseutilities.h
  src/estimator/v3d_code/v3d_poseutilities.cpp
)

source_group(${CMH_MODULE_NAME} FILES ${SOURCE_FILES})
source_group(${CMH_MODULE_NAME}\\v3d_code FILES ${V3D_CODE_FILES})

add_library(${CMH_MODULE_NAME} ${SOURCE_FILES} ${V3D_CODE_FILES})

cmh_target_include_directories(
  ${CMAKE_CURRENT_LIST_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)

cmh_target_link_libraries(
  ${OpenCV_LIBS}
)

cmh_end_module()
