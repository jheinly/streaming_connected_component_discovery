#include <estimator/essential_matrix_estimator.h>

estimator::EssentialMatrixEstimator::EssentialMatrixEstimator(
  const int max_num_ransac_iterations,
  const int min_num_inliers,
  const float max_point_to_line_distance)
: m_max_num_ransac_iterations(max_num_ransac_iterations),
  m_min_num_inliers(min_num_inliers),
  m_max_point_to_line_distance(max_point_to_line_distance)
{
#ifdef USE_V3D_CODE_ESSENTIAL_MATRIX
  m_container.E_Estimator_ocv.init(5, max_point_to_line_distance, max_num_ransac_iterations, 10, 200);
#endif
}

estimator::EssentialMatrixEstimator::~EssentialMatrixEstimator()
{}
