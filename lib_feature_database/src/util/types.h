// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_UTIL_INTTYPES_H_
#define COLMAP_SRC_UTIL_INTTYPES_H_

#ifdef _MSC_VER
#if _MSC_VER >= 1600
#include <cstdint>
#else
typedef __int8 int8_t;
typedef __int16 int16_t;
typedef __int32 int32_t;
typedef __int64 int64_t;
typedef unsigned __int8 uint8_t;
typedef unsigned __int16 uint16_t;
typedef unsigned __int32 uint32_t;
typedef unsigned __int64 uint64_t;
#endif
#elif __GNUC__ >= 3
#include <cstdint>
#endif

#include <Eigen/Core>

namespace Eigen {

typedef Eigen::Matrix<double, 3, 4> Matrix3x4d;

}  // end namespace Eigen

namespace colmap {

////////////////////////////////////////////////////////////////////////////////
// Index types, determines the maximum number of objects
////////////////////////////////////////////////////////////////////////////////

// There is at most one camera per image, but multiple images might share the
// same camera
typedef uint32_t camera_t;
typedef uint32_t image_t;

// Each image pair gets a unique ID, see `FeatureDatabase::image_pair2pair_id`
typedef uint64_t image_pair_t;

// Feature track describing the pairwise match graph
typedef uint32_t track_t;

// Index per image, i.e. determines maximum number of 2D points per image
typedef uint32_t point2D_t;

// Unique ID per added 3D point. Since we add many 3D points, delete them, and
// possibly re-add them again, the maximum number of allowed unique indices
// should be larger here, see `StructureManager::num_added_points3D_`
typedef uint64_t point3D_t;

// Index of a reconstructed model
typedef uint32_t model_t;

const camera_t INVALID_CAMERA_ID = std::numeric_limits<camera_t>::max();
const image_t INVALID_IMAGE_ID = std::numeric_limits<image_t>::max();
const track_t INVALID_TRACK_ID = std::numeric_limits<track_t>::max();
const image_pair_t INVALID_IMAGE_PAIR_ID =
    std::numeric_limits<image_pair_t>::max();
const point2D_t INVALID_POINT2D_IDX = std::numeric_limits<point2D_t>::max();
const point3D_t INVALID_POINT3D_ID = std::numeric_limits<point3D_t>::max();
const model_t INVALID_MODEL_ID = std::numeric_limits<model_t>::max();

////////////////////////////////////////////////////////////////////////////////
// Matrix types
////////////////////////////////////////////////////////////////////////////////

// Use RowMajor order for compatibility with SiftGPU, and better data-locality,
// since we store one keypoint / descriptor entry row-wise.
// (Eigen's default is ColMajor)

// Per row: x, y, scale, orientation
typedef Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor>
    FeatureKeypoints;
// Per row: descriptor per feature
typedef Eigen::Matrix<uint8_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    FeatureDescriptors;
// Per row: idx1, idx2
typedef Eigen::Matrix<point2D_t, Eigen::Dynamic, 2, Eigen::RowMajor>
    FeatureMatches;

}  // end namespace colmap

#endif  // COLMAP_SRC_UTIL_INTTYPES_H_
