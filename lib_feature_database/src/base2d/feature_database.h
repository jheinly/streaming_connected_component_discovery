// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#ifndef COLMAP_SRC_BASE2D_FEATURE_DATABASE_H_
#define COLMAP_SRC_BASE2D_FEATURE_DATABASE_H_

#include "base2d/image_data.h"
#include "base2d/camera.h"
#include "estimators/two_view_geometry.h"
#if 0 // Disabled as this code is not necessary for the feature database.
#include "ext/SQLite/sqlite3.h"
#else
#include <sqlite/sqlite3.h>
#endif // Disabled as this code is not necessary for the feature database.
#include "util/types.h"
#include "util/sqlite3_utils.h"

#include <vector>
#include <unordered_map>

#include <Eigen/Core>

namespace colmap {

class FeatureDatabase {
 public:
  const static int SCHEMA_VERSION = 1;

  // The maximum number of images, that can be stored in the database.
  // This limitation arises due to the fact, that we generate unique IDs for
  // image pairs manually. Note: do not change this to
  // another type than `size_t`.
  const static size_t MAX_NUM_IMAGES;

  FeatureDatabase();
  ~FeatureDatabase();

  void init(const std::string& path);
  void close();

  void begin_transaction() const;
  void end_transaction() const;

  /*
    Check if entry already exists in database.

    For image pairs, the order of `image_id1` and `image_id2` does not matter.
  */
  bool exists_camera(const camera_t camera_id) const;
  bool exists_image(const image_t image_id) const;
  bool exists_image(std::string name) const;
  bool exists_keypoints(const image_t image_id) const;
  bool exists_descriptors(const image_t image_id) const;
  bool exists_matches(image_t image_id1, image_t image_id2) const;
  bool exists_inlier_matches(image_t image_id1, image_t image_id2) const;

  /*
    Number of rows in camera table.
  */
  size_t get_num_cameras() const;

  /*
    Number of rows in image table.
  */
  size_t get_num_images() const;

  /*
    Sum of `rows` column in `keypoints` table,
    i.e. number of total keypoints.
  */
  size_t get_num_keypoints() const;

  /*
    Sum of `rows` column in `descriptors` table,
    i.e. number of total descriptors.
  */
  size_t get_num_descriptors() const;

  /*
    Sum of `rows` column in `matches` table,
    i.e. number of total matches.
  */
  size_t get_num_matches() const;

  /*
    Sum of `rows` column in `inlier_matches` table,
    i.e. number of total inlier matches.
  */
  size_t get_num_inlier_matches() const;

  size_t get_num_pairs() const;
  size_t get_num_inlier_pairs() const;

  /*
    Each image pair is assigned a unique ID in the `matches` and
    `inlier_matches` table. We intentionally avoid to store the pairs in a
    separate table by using e.g. AUTOINCREMENT, since the overhead of querying
    the unique pair ID is significant.
  */
  static inline image_pair_t image_pair_to_pair_id(image_t image_id1,
                                                   image_t image_id2) {
    swap_image_pair(image_id1, image_id2);
    return MAX_NUM_IMAGES * image_id1 + image_id2;
  }

  static inline void pair_id_to_image_pair(const image_pair_t pair_id,
                                           image_t& image_id1,
                                           image_t& image_id2) {
    image_id2 = static_cast<image_t>(pair_id % MAX_NUM_IMAGES);
    image_id1 = static_cast<image_t>((pair_id - image_id2) / MAX_NUM_IMAGES);
  }

  /*
    To avoid duplicate pair entries in the database, we enforce that
    `image_id1 < image_id2`.
  */
  static bool swap_image_pair(image_t& image_id1, image_t& image_id2);

  /*
    Swap the columns of the matches
   */
  static void swap_matches(FeatureMatches& matches);

  /*
    Add new camera and return its new ID.
  */
  camera_t add_camera(const Camera& camera_data,
                      const bool use_camera_id = false) const;

  /*
    Add new image and return its new ID
  */
  image_t add_image(const ImageData& image_data,
                    const bool use_image_id = false) const;

  /*
    Update an existing entry in the database.

    You are responsible for making sure that the entry actually exists.
  */
  void update_camera(const camera_t camera_id, const Camera& camera);
  void update_image(const ImageData& image_data);

  /*
    Read an existing entry in the database.

    You are responsible for making sure that the entry actually exists.

    For image pairs, the order of `image_id1` and `image_id2` does not matter.
  */
  Camera read_camera(const camera_t camera_id) const;
  std::unordered_map<camera_t, Camera> read_cameras() const;

  ImageData read_image(const image_t image_id) const;
  ImageData read_image(std::string name) const;
  std::vector<ImageData> read_images() const;

  FeatureKeypoints read_keypoints(const image_t image_id) const;
  FeatureDescriptors read_descriptors(const image_t image_id) const;

  FeatureMatches read_matches(image_t image_id1, image_t image_id2) const;
  std::vector<std::pair<image_pair_t, FeatureMatches>> read_matches() const;

  TwoViewGeometry read_inlier_matches(image_t image_id1,
                                      image_t image_id2) const;
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> read_inlier_matches()
      const;

  void read_inlier_matches_graph(
      std::vector<std::pair<image_t, image_t>>& image_pairs,
      std::vector<int>& num_inliers) const;

  /*
    Write a new entry in the database.

    You are responsible for making sure that the entry does not yet exists.

    For image pairs, the order of `image_id1` and `image_id2` does not matter.
  */
  void write_keypoints(const image_t image_id,
                       const FeatureKeypoints& keypoints) const;
  void write_descriptors(const image_t image_id,
                         const FeatureDescriptors& descriptors) const;
  void write_matches(image_t image_id1, image_t image_id2,
                     FeatureMatches matches) const;
  void write_inlier_matches(image_t image_id1, image_t image_id2,
                            TwoViewGeometry two_view_geometry) const;

 private:
  // Prepare SQL statements once at construction of the database, and reuse
  // the statements for multiple queries by resetting their states.
  void prepare_sql_stmts_();
  void finalize_sql_stmts_();

  // Create database tables, if not existing. This is done at construction.
  void create_tables_() const;
  void create_camera_table_() const;
  void create_image_table_() const;
  void create_keypoints_table_() const;
  void create_descriptors_table_() const;
  void create_matches_table_() const;
  void create_inlier_matches_table_() const;

  void update_schema_() const;

  bool exists_row_(sqlite3_stmt* sql_stmt, const size_t row_id) const;
  bool exists_row_(sqlite3_stmt* sql_stmt, const std::string& row_entry) const;

  size_t count_rows_(const std::string& table) const;
  size_t sum_column_(const std::string& column, const std::string& table) const;

  sqlite3* database_;

  std::vector<sqlite3_stmt*> sql_stmts_;

  // exists_*
  sqlite3_stmt* sql_stmt_exists_camera_;
  sqlite3_stmt* sql_stmt_exists_image_id_;
  sqlite3_stmt* sql_stmt_exists_image_name_;
  sqlite3_stmt* sql_stmt_exists_keypoints_;
  sqlite3_stmt* sql_stmt_exists_descriptors_;
  sqlite3_stmt* sql_stmt_exists_matches_;
  sqlite3_stmt* sql_stmt_exists_inlier_matches_;

  // add_*
  sqlite3_stmt* sql_stmt_add_camera_;
  sqlite3_stmt* sql_stmt_add_image_;

  // update_*
  sqlite3_stmt* sql_stmt_update_camera_;
  sqlite3_stmt* sql_stmt_update_image_;

  // read_*
  sqlite3_stmt* sql_stmt_read_camera_;
  sqlite3_stmt* sql_stmt_read_cameras_;
  sqlite3_stmt* sql_stmt_read_image_id_;
  sqlite3_stmt* sql_stmt_read_image_name_;
  sqlite3_stmt* sql_stmt_read_images_;
  sqlite3_stmt* sql_stmt_read_keypoints_;
  sqlite3_stmt* sql_stmt_read_descriptors_;
  sqlite3_stmt* sql_stmt_read_matches_;
  sqlite3_stmt* sql_stmt_read_matches_all_;
  sqlite3_stmt* sql_stmt_read_inlier_matches_;
  sqlite3_stmt* sql_stmt_read_inlier_matches_all_;
  sqlite3_stmt* sql_stmt_read_inlier_matches_graph_;

  // write_*
  sqlite3_stmt* sql_stmt_write_keypoints_;
  sqlite3_stmt* sql_stmt_write_descriptors_;
  sqlite3_stmt* sql_stmt_write_matches_;
  sqlite3_stmt* sql_stmt_write_inlier_matches_;
};

}  // end namespace colmap

#endif  // COLMAP_SRC_BASE2D_FEATURE_DATABASE_H_
