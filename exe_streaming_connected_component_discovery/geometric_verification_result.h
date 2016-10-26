#pragma once
#ifndef GEOMETRIC_VERIFICATION_RESULT_H
#define GEOMETRIC_VERIFICATION_RESULT_H

struct GeometricVerificationResult
{
  GeometricVerificationResult()
    : successful_task_index(-1),
    num_inliers(0)
  {}

  GeometricVerificationResult(
    const int successful_task_idx,
    const int number_of_inliers)
    : successful_task_index(successful_task_idx),
    num_inliers(number_of_inliers)
  {}

  int successful_task_index;
  int num_inliers;
};

#endif // GEOMETRIC_VERIFICATION_RESULT_H
