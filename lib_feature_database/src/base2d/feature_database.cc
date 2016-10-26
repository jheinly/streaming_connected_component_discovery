// Author: Johannes L. Schoenberger <jsch at cs.unc.edu>

#include "base2d/feature_database.h"

#include <fstream>

#include <boost/lexical_cast.hpp>

namespace colmap {

template <typename MatrixType>
MatrixType read_matrix_blob_(sqlite3_stmt* sql_stmt, const int rc,
                             const int col) {
  assert(col >= 0);

  MatrixType matrix;

  if (rc == SQLITE_ROW) {
    const size_t rows =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 0));
    const size_t cols =
        static_cast<size_t>(sqlite3_column_int64(sql_stmt, col + 1));

    assert(rows >= 0);
    assert(cols >= 0);

    matrix = MatrixType(rows, cols);
    const size_t num_bytes =
        static_cast<size_t>(sqlite3_column_bytes(sql_stmt, col + 2));
    memcpy(reinterpret_cast<char*>(matrix.data()),
           sqlite3_column_blob(sql_stmt, col + 2), num_bytes);
  } else {
    const size_t rows = (MatrixType::RowsAtCompileTime == Eigen::Dynamic)
                        ? 0
                        : MatrixType::RowsAtCompileTime;
    const size_t cols = (MatrixType::ColsAtCompileTime == Eigen::Dynamic)
                        ? 0
                        : MatrixType::ColsAtCompileTime;
    matrix = MatrixType(rows, cols);
  }

  return matrix;
}

template <typename MatrixType>
void write_matrix_blob_(sqlite3_stmt* sql_stmt, const MatrixType& matrix,
                        const int col) {
  assert(matrix.rows() >= 0);
  assert(matrix.cols() >= 0);
  assert(col >= 0);

  const size_t num_bytes =
      matrix.size() * sizeof(typename MatrixType::Scalar);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 0, matrix.rows()));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt, col + 1, matrix.cols()));
  SQLITE3_CALL(sqlite3_bind_blob(
      sql_stmt, col + 2,
      reinterpret_cast<const char*>(matrix.derived().data()),
      static_cast<int>(num_bytes), SQLITE_STATIC));
}

Camera read_camera_row_(sqlite3_stmt* sql_stmt) {
  Camera camera;

  camera.model = sqlite3_column_int(sql_stmt, 1);
  camera.width = static_cast<int>(sqlite3_column_int64(sql_stmt, 2));
  camera.height = static_cast<int>(sqlite3_column_int64(sql_stmt, 3));

  const size_t num_params_bytes =
      static_cast<size_t>(sqlite3_column_bytes(sql_stmt, 4));
  camera.params.resize(num_params_bytes / sizeof(double));
  memcpy(camera.params.data(), sqlite3_column_blob(sql_stmt, 4),
         num_params_bytes);

  camera.prior_focal = sqlite3_column_int(sql_stmt, 5) != 0;

  return camera;
}

ImageData read_image_row_(sqlite3_stmt* sql_stmt) {
  ImageData image_data;
  image_data.image_id = static_cast<image_t>(sqlite3_column_int64(sql_stmt, 0));
  image_data.name = std::string(
      reinterpret_cast<const char*>(sqlite3_column_text(sql_stmt, 1)));
  image_data.camera_id =
      static_cast<camera_t>(sqlite3_column_int64(sql_stmt, 2));
  image_data.roll = sqlite3_column_double(sql_stmt, 3);
  image_data.pitch = sqlite3_column_double(sql_stmt, 4);
  image_data.yaw = sqlite3_column_double(sql_stmt, 5);
  image_data.x = sqlite3_column_double(sql_stmt, 6);
  image_data.y = sqlite3_column_double(sql_stmt, 7);
  image_data.z = sqlite3_column_double(sql_stmt, 8);

  return image_data;
}

const size_t FeatureDatabase::MAX_NUM_IMAGES =
    static_cast<size_t>(std::numeric_limits<int32_t>::max());

FeatureDatabase::FeatureDatabase() { database_ = nullptr; }

FeatureDatabase::~FeatureDatabase() { close(); }

void FeatureDatabase::init(const std::string& path) {
  close();

  // SQLITE_OPEN_NOMUTEX specifies that the connection should not have a
  // mutex (so that we don't serialize the connection's operations).
  // Modifications to the database will still be serialized, but multiple
  // connections can read concurrently.
  SQLITE3_CALL(sqlite3_open_v2(
      path.c_str(), &database_,
      SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE | SQLITE_OPEN_NOMUTEX,
      nullptr));

  // Don't wait for the operating system to write the changes to disk
  SQLITE3_EXEC(database_, "PRAGMA synchronous=OFF", nullptr);

  // Use faster journaling mode
  SQLITE3_EXEC(database_, "PRAGMA journal_mode=WAL", nullptr);

  // Store temporary tables and indices in memory
  SQLITE3_EXEC(database_, "PRAGMA temp_store=MEMORY", nullptr);

  // Disabled by default
  SQLITE3_EXEC(database_, "PRAGMA foreign_keys=ON", nullptr);

  create_tables_();
  update_schema_();

  prepare_sql_stmts_();
}

void FeatureDatabase::close() {
  if (database_ != nullptr) {
    finalize_sql_stmts_();
    sqlite3_close_v2(database_);
    database_ = nullptr;
  }
}

void FeatureDatabase::begin_transaction() const {
  SQLITE3_EXEC(database_, "BEGIN TRANSACTION", nullptr);
}

void FeatureDatabase::end_transaction() const {
  SQLITE3_EXEC(database_, "END TRANSACTION", nullptr);
}

bool FeatureDatabase::exists_camera(const camera_t camera_id) const {
  return exists_row_(sql_stmt_exists_camera_, camera_id);
}

bool FeatureDatabase::exists_image(const image_t image_id) const {
  return exists_row_(sql_stmt_exists_image_id_, image_id);
}

bool FeatureDatabase::exists_image(std::string name) const {
  return exists_row_(sql_stmt_exists_image_name_, name);
}

bool FeatureDatabase::exists_keypoints(const image_t image_id) const {
  return exists_row_(sql_stmt_exists_keypoints_, image_id);
}

bool FeatureDatabase::exists_descriptors(const image_t image_id) const {
  return exists_row_(sql_stmt_exists_descriptors_, image_id);
}

bool FeatureDatabase::exists_matches(image_t image_id1,
                                     image_t image_id2) const {
  return exists_row_(sql_stmt_exists_matches_,
                     image_pair_to_pair_id(image_id1, image_id2));
}

bool FeatureDatabase::exists_inlier_matches(image_t image_id1,
                                            image_t image_id2) const {
  return exists_row_(sql_stmt_exists_inlier_matches_,
                     image_pair_to_pair_id(image_id1, image_id2));
}

size_t FeatureDatabase::get_num_cameras() const {
  return count_rows_("cameras");
}

size_t FeatureDatabase::get_num_images() const {
  return count_rows_("images");
}

size_t FeatureDatabase::get_num_keypoints() const {
  return sum_column_("rows", "keypoints");
}

size_t FeatureDatabase::get_num_descriptors() const {
  return sum_column_("rows", "keypoints");
}

size_t FeatureDatabase::get_num_matches() const {
  return sum_column_("rows", "matches");
}

size_t FeatureDatabase::get_num_inlier_matches() const {
  return sum_column_("rows", "inlier_matches");
}

size_t FeatureDatabase::get_num_pairs() const {
  return count_rows_("matches");
}

size_t FeatureDatabase::get_num_inlier_pairs() const {
  return count_rows_("inlier_matches");
}

bool FeatureDatabase::swap_image_pair(image_t& image_id1, image_t& image_id2) {
  if (image_id1 > image_id2) {
    std::swap(image_id1, image_id2);
    return true;
  }
  return false;
}

void FeatureDatabase::swap_matches(FeatureMatches& matches) {
  matches.col(0).swap(matches.col(1));
}

camera_t FeatureDatabase::add_camera(const Camera& camera,
                                     const bool use_camera_id) const {
  if (use_camera_id) {
    if (exists_image(camera.camera_id)) {
      throw std::invalid_argument("`camera_id` already exists");
    }
    SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 1, camera.camera_id));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_camera_, 1));
  }

  SQLITE3_CALL(sqlite3_bind_int(sql_stmt_add_camera_, 2, camera.model));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.width)));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_add_camera_, 4,
                                  static_cast<sqlite3_int64>(camera.height)));

  const size_t num_params_bytes = sizeof(double) * camera.params.size();
  SQLITE3_CALL(sqlite3_bind_blob(sql_stmt_add_camera_, 5, camera.params.data(),
                                 static_cast<int>(num_params_bytes),
                                 SQLITE_STATIC));

  SQLITE3_CALL(sqlite3_bind_int(sql_stmt_add_camera_, 6, camera.prior_focal));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_camera_));

  return static_cast<camera_t>(sqlite3_last_insert_rowid(database_));
}

image_t FeatureDatabase::add_image(const ImageData& image_data,
                                   const bool use_image_id) const {
  if (use_image_id) {
    if (exists_image(image_data.image_id)) {
      throw std::invalid_argument("`image_id` already exists");
    }
    SQLITE3_CALL(
        sqlite3_bind_int64(sql_stmt_add_image_, 1, image_data.image_id));
  } else {
    SQLITE3_CALL(sqlite3_bind_null(sql_stmt_add_image_, 1));
  }

  std::string name = image_data.name;
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_add_image_, 2, name.c_str(),
                                 static_cast<int>(name.size()), SQLITE_STATIC));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_add_image_, 3, image_data.camera_id));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 4, image_data.roll));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 5, image_data.pitch));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 6, image_data.yaw));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 7, image_data.x));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 8, image_data.y));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_add_image_, 9, image_data.z));

  SQLITE3_CALL(sqlite3_step(sql_stmt_add_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_add_image_));

  return static_cast<image_t>(sqlite3_last_insert_rowid(database_));
}

void FeatureDatabase::update_camera(const camera_t camera_id,
                                    const Camera& camera) {
  SQLITE3_CALL(sqlite3_bind_int(sql_stmt_update_camera_, 1, camera.model));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 2,
                                  static_cast<sqlite3_int64>(camera.width)));
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 3,
                                  static_cast<sqlite3_int64>(camera.height)));

  const size_t num_params_bytes = sizeof(double) * camera.params.size();
  SQLITE3_CALL(
      sqlite3_bind_blob(sql_stmt_update_camera_, 4, camera.params.data(),
                        static_cast<int>(num_params_bytes), SQLITE_STATIC));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_camera_, 5, camera.prior_focal));

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_update_camera_, 6, camera_id));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_camera_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_camera_));
}

void FeatureDatabase::update_image(const ImageData& image_data) {
  std::string name = image_data.name;
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_update_image_, 1, name.c_str(),
                                 static_cast<int>(name.size()), SQLITE_STATIC));
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_image_, 2, image_data.camera_id));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_update_image_, 3, image_data.roll));
  SQLITE3_CALL(
      sqlite3_bind_double(sql_stmt_update_image_, 4, image_data.pitch));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_update_image_, 5, image_data.yaw));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_update_image_, 6, image_data.x));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_update_image_, 7, image_data.y));
  SQLITE3_CALL(sqlite3_bind_double(sql_stmt_update_image_, 8, image_data.z));

  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt_update_image_, 9, image_data.image_id));

  SQLITE3_CALL(sqlite3_step(sql_stmt_update_image_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_update_image_));
}

Camera FeatureDatabase::read_camera(const camera_t camera_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_camera_, 1, camera_id));

  Camera camera;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_camera_));
  if (rc == SQLITE_ROW) {
    camera = read_camera_row_(sql_stmt_read_camera_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_camera_));

  return camera;
}

std::unordered_map<camera_t, Camera> FeatureDatabase::read_cameras() const {
  std::unordered_map<camera_t, Camera> cameras;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_cameras_)) == SQLITE_ROW) {
    const camera_t camera_id =
        static_cast<camera_t>(sqlite3_column_int64(sql_stmt_read_cameras_, 0));
    cameras[camera_id] = read_camera_row_(sql_stmt_read_cameras_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_cameras_));

  return cameras;
}

ImageData FeatureDatabase::read_image(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_image_id_, 1, image_id));

  ImageData image_data;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_id_));
  if (rc == SQLITE_ROW) {
    image_data = read_image_row_(sql_stmt_read_image_id_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_id_));

  return image_data;
}

ImageData FeatureDatabase::read_image(std::string name) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt_read_image_name_, 1, name.c_str(),
                                 static_cast<int>(name.size()), SQLITE_STATIC));

  ImageData image_data;

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_image_name_));
  if (rc == SQLITE_ROW) {
    image_data = read_image_row_(sql_stmt_read_image_name_);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_image_name_));

  return image_data;
}

std::vector<ImageData> FeatureDatabase::read_images() const {
  std::vector<ImageData> image_data;

  while (SQLITE3_CALL(sqlite3_step(sql_stmt_read_images_)) == SQLITE_ROW) {
    image_data.push_back(read_image_row_(sql_stmt_read_images_));
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_images_));

  return image_data;
}

FeatureKeypoints FeatureDatabase::read_keypoints(const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_keypoints_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_keypoints_));
  FeatureKeypoints keypoints =
      read_matrix_blob_<FeatureKeypoints>(sql_stmt_read_keypoints_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_keypoints_));

  return keypoints;
}

FeatureDescriptors FeatureDatabase::read_descriptors(
    const image_t image_id) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_descriptors_, 1, image_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_descriptors_));
  FeatureDescriptors descriptors =
      read_matrix_blob_<FeatureDescriptors>(sql_stmt_read_descriptors_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_descriptors_));

  return descriptors;
}

FeatureMatches FeatureDatabase::read_matches(image_t image_id1,
                                             image_t image_id2) const {
  const image_pair_t pair_id = image_pair_to_pair_id(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_));
  FeatureMatches matches =
      read_matrix_blob_<FeatureMatches>(sql_stmt_read_matches_, rc, 0);

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_));

  if (swap_image_pair(image_id1, image_id2)) {
    swap_matches(matches);
  }

  return matches;
}

std::vector<std::pair<image_pair_t, FeatureMatches>>
FeatureDatabase::read_matches() const {
  std::vector<std::pair<image_pair_t, FeatureMatches>> all_matches;

  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_matches_all_))) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_matches_all_, 0));
    const FeatureMatches matches =
        read_matrix_blob_<FeatureMatches>(sql_stmt_read_matches_all_, rc, 1);
    all_matches.emplace_back(pair_id, matches);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_matches_all_));

  return all_matches;
}

TwoViewGeometry FeatureDatabase::read_inlier_matches(image_t image_id1,
                                                     image_t image_id2) const {
  const image_pair_t pair_id = image_pair_to_pair_id(image_id1, image_id2);
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_read_inlier_matches_, 1, pair_id));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_inlier_matches_));

  TwoViewGeometry two_view_geometry;

  two_view_geometry.inlier_matches =
      read_matrix_blob_<FeatureMatches>(sql_stmt_read_inlier_matches_, rc, 0);

  two_view_geometry.config =
      static_cast<int>(sqlite3_column_int64(sql_stmt_read_inlier_matches_, 3));

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_));

  if (swap_image_pair(image_id1, image_id2)) {
    swap_matches(two_view_geometry.inlier_matches);
  }

  return two_view_geometry;
}

std::vector<std::pair<image_pair_t, TwoViewGeometry>>
FeatureDatabase::read_inlier_matches() const {
  std::vector<std::pair<image_pair_t, TwoViewGeometry>> results;

  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(sql_stmt_read_inlier_matches_all_))) ==
         SQLITE_ROW) {
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_all_, 0));

    TwoViewGeometry two_view_geometry;

    two_view_geometry.inlier_matches = read_matrix_blob_<FeatureMatches>(
        sql_stmt_read_inlier_matches_all_, rc, 1);

    two_view_geometry.config = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_, 4));

    results.emplace_back(pair_id, two_view_geometry);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_all_));

  return results;
}

void FeatureDatabase::read_inlier_matches_graph(
    std::vector<std::pair<image_t, image_t>>& image_pairs,
    std::vector<int>& num_inliers) const {
  const auto num_inlier_matches = get_num_inlier_matches();
  image_pairs.reserve(num_inlier_matches);
  num_inliers.reserve(num_inlier_matches);

  int rc;
  while ((rc = SQLITE3_CALL(sqlite3_step(
              sql_stmt_read_inlier_matches_graph_))) == SQLITE_ROW) {
    image_t image_id1;
    image_t image_id2;
    const image_pair_t pair_id = static_cast<image_pair_t>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_graph_, 0));
    pair_id_to_image_pair(pair_id, image_id1, image_id2);
    image_pairs.emplace_back(image_id1, image_id2);

    const int rows = static_cast<int>(
        sqlite3_column_int64(sql_stmt_read_inlier_matches_graph_, 1));
    num_inliers.push_back(rows);
  }

  SQLITE3_CALL(sqlite3_reset(sql_stmt_read_inlier_matches_graph_));
}

void FeatureDatabase::write_keypoints(const image_t image_id,
                                      const FeatureKeypoints& keypoints) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_keypoints_, 1, image_id));
  write_matrix_blob_(sql_stmt_write_keypoints_, keypoints, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_keypoints_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_keypoints_));
}

void FeatureDatabase::write_descriptors(
    const image_t image_id, const FeatureDescriptors& descriptors) const {
  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_descriptors_, 1, image_id));
  write_matrix_blob_(sql_stmt_write_descriptors_, descriptors, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_descriptors_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_descriptors_));
}

void FeatureDatabase::write_matches(image_t image_id1, image_t image_id2,
                                    FeatureMatches matches) const {
  if (swap_image_pair(image_id1, image_id2)) {
    swap_matches(matches);
  }

  const image_pair_t pair_id = image_pair_to_pair_id(image_id1, image_id2);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_matches_, 1, pair_id));
  write_matrix_blob_(sql_stmt_write_matches_, matches, 2);

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_matches_));
}

void FeatureDatabase::write_inlier_matches(
    image_t image_id1, image_t image_id2,
    TwoViewGeometry two_view_geometry) const {
  if (swap_image_pair(image_id1, image_id2)) {
    swap_matches(two_view_geometry.inlier_matches);
  }

  const image_pair_t pair_id = image_pair_to_pair_id(image_id1, image_id2);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_inlier_matches_, 1, pair_id));

  write_matrix_blob_(sql_stmt_write_inlier_matches_,
                     two_view_geometry.inlier_matches, 2);

  SQLITE3_CALL(sqlite3_bind_int64(sql_stmt_write_inlier_matches_, 5,
                                  two_view_geometry.config));

  SQLITE3_CALL(sqlite3_step(sql_stmt_write_inlier_matches_));
  SQLITE3_CALL(sqlite3_reset(sql_stmt_write_inlier_matches_));
}

void FeatureDatabase::prepare_sql_stmts_() {
  sql_stmts_.clear();

  std::string sql;

  //////////////////////////////////////////////////////////////////////////////
  // exists_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT 1 FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_camera_, 0));
  sql_stmts_.push_back(sql_stmt_exists_camera_);

  sql = "SELECT 1 FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_id_);

  sql = "SELECT 1 FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_exists_image_name_);

  sql = "SELECT 1 FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_exists_keypoints_);

  sql = "SELECT 1 FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_exists_descriptors_);

  sql = "SELECT 1 FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_matches_, 0));
  sql_stmts_.push_back(sql_stmt_exists_matches_);

  sql = "SELECT 1 FROM inlier_matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_exists_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_exists_inlier_matches_);

  //////////////////////////////////////////////////////////////////////////////
  // add_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "INSERT INTO cameras(camera_id, model, width, height, params, "
      "prior_focal) VALUES(?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_camera_, 0));
  sql_stmts_.push_back(sql_stmt_add_camera_);

  sql =
      "INSERT INTO images(image_id, name, camera_id, roll, pitch, yaw, x, y, z)"
      " VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?);";
  SQLITE3_CALL(
      sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt_add_image_, 0));
  sql_stmts_.push_back(sql_stmt_add_image_);

  //////////////////////////////////////////////////////////////////////////////
  // update_*
  //////////////////////////////////////////////////////////////////////////////
  sql =
      "UPDATE cameras SET model=?, width=?, height=?, params=?, prior_focal=? "
      "WHERE camera_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_camera_, 0));
  sql_stmts_.push_back(sql_stmt_update_camera_);

  sql =
      "UPDATE images SET name=?, camera_id=?, roll=?, pitch=?, yaw=?, x=?, "
      "y=?, z=? WHERE image_id=?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_update_image_, 0));
  sql_stmts_.push_back(sql_stmt_update_image_);

  //////////////////////////////////////////////////////////////////////////////
  // read_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "SELECT * FROM cameras WHERE camera_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_camera_, 0));
  sql_stmts_.push_back(sql_stmt_read_camera_);

  sql = "SELECT * FROM cameras;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_cameras_, 0));
  sql_stmts_.push_back(sql_stmt_read_cameras_);

  sql = "SELECT * FROM images WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_id_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_id_);

  sql = "SELECT * FROM images WHERE name = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_image_name_, 0));
  sql_stmts_.push_back(sql_stmt_read_image_name_);

  sql = "SELECT * FROM images;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_images_, 0));
  sql_stmts_.push_back(sql_stmt_read_images_);

  sql = "SELECT rows, cols, data FROM keypoints WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_read_keypoints_);

  sql = "SELECT rows, cols, data FROM descriptors WHERE image_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_read_descriptors_);

  sql = "SELECT rows, cols, data FROM matches WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_);

  sql = "SELECT * FROM matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_matches_all_);

  sql =
      "SELECT rows, cols, data, config FROM inlier_matches "
      "WHERE pair_id = ?;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_);

  sql = "SELECT * FROM inlier_matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_all_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_all_);

  sql = "SELECT pair_id, rows FROM inlier_matches WHERE rows > 0;";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_read_inlier_matches_graph_, 0));
  sql_stmts_.push_back(sql_stmt_read_inlier_matches_graph_);

  //////////////////////////////////////////////////////////////////////////////
  // write_*
  //////////////////////////////////////////////////////////////////////////////
  sql = "INSERT INTO keypoints(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_keypoints_, 0));
  sql_stmts_.push_back(sql_stmt_write_keypoints_);

  sql =
      "INSERT INTO descriptors(image_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_descriptors_, 0));
  sql_stmts_.push_back(sql_stmt_write_descriptors_);

  sql = "INSERT INTO matches(pair_id, rows, cols, data) VALUES(?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_matches_);

  sql =
      "INSERT INTO inlier_matches(pair_id, rows, cols, data, config) "
      "VALUES(?, ?, ?, ?, ?);";
  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1,
                                  &sql_stmt_write_inlier_matches_, 0));
  sql_stmts_.push_back(sql_stmt_write_inlier_matches_);
}

void FeatureDatabase::finalize_sql_stmts_() {
  for (const auto& sql_stmt : sql_stmts_) {
    SQLITE3_CALL(sqlite3_finalize(sql_stmt));
  }
}

void FeatureDatabase::create_tables_() const {
  create_camera_table_();
  create_image_table_();
  create_keypoints_table_();
  create_descriptors_table_();
  create_matches_table_();
  create_inlier_matches_table_();
}

void FeatureDatabase::create_camera_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS cameras"
      "   (camera_id     INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    model         INTEGER                             NOT NULL,"
      "    width         INTEGER                             NOT NULL,"
      "    height        INTEGER                             NOT NULL,"
      "    params        BLOB,"
      "    prior_focal   INTEGER                             NOT NULL);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::create_image_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS images"
      "   (image_id   INTEGER  PRIMARY KEY AUTOINCREMENT  NOT NULL,"
      "    name       TEXT                                NOT NULL UNIQUE,"
      "    camera_id  INTEGER                             NOT NULL,"
      "    roll       REAL,"
      "    pitch      REAL,"
      "    yaw        REAL,"
      "    x          REAL,"
      "    y          REAL,"
      "    z          REAL,"
      "FOREIGN KEY(camera_id) REFERENCES cameras(camera_id) ON DELETE CASCADE);"
      "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::create_keypoints_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS keypoints"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::create_descriptors_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS descriptors"
      "   (image_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows      INTEGER               NOT NULL,"
      "    cols      INTEGER               NOT NULL,"
      "    data      BLOB,"
      "FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::create_matches_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::create_inlier_matches_table_() const {
  const std::string sql =
      "CREATE TABLE IF NOT EXISTS inlier_matches"
      "   (pair_id  INTEGER  PRIMARY KEY  NOT NULL,"
      "    rows     INTEGER               NOT NULL,"
      "    cols     INTEGER               NOT NULL,"
      "    data     BLOB,"
      "    config   INTEGER               NOT NULL);";

  SQLITE3_EXEC(database_, sql.c_str(), nullptr);
}

void FeatureDatabase::update_schema_() const {
  // Query user_version
  const std::string query_user_version_sql = "PRAGMA user_version;";
  sqlite3_stmt* query_user_version_sql_stmt;
  SQLITE3_CALL(sqlite3_prepare_v2(database_, query_user_version_sql.c_str(), -1,
                                  &query_user_version_sql_stmt, 0));

  // Update schema, if user_version < SCHEMA_VERSION
  if (SQLITE3_CALL(sqlite3_step(query_user_version_sql_stmt)) == SQLITE_ROW) {
    const int user_version = sqlite3_column_int(query_user_version_sql_stmt, 0);
    // user_version == 0: initial value from SQLite, nothing to do, since all
    // tables were created in `FeatureDatabase::create_tables_`
    if (user_version > 0) {
      // if (user_version < 2) {}
    }
  }

  SQLITE3_CALL(sqlite3_finalize(query_user_version_sql_stmt));

  // Update user_version
  const std::string update_user_version_sql =
      "PRAGMA user_version = " + std::to_string(SCHEMA_VERSION) + ";";
  SQLITE3_EXEC(database_, update_user_version_sql.c_str(), nullptr);
}

bool FeatureDatabase::exists_row_(sqlite3_stmt* sql_stmt,
                                  const size_t row_id) const {
  SQLITE3_CALL(
      sqlite3_bind_int64(sql_stmt, 1, static_cast<sqlite3_int64>(row_id)));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

  const bool exists = rc == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

bool FeatureDatabase::exists_row_(sqlite3_stmt* sql_stmt,
                                  const std::string& row_entry) const {
  SQLITE3_CALL(sqlite3_bind_text(sql_stmt, 1, row_entry.c_str(),
                                 static_cast<int>(row_entry.size()),
                                 SQLITE_STATIC));

  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));

  const bool exists = rc == SQLITE_ROW;

  SQLITE3_CALL(sqlite3_reset(sql_stmt));

  return exists;
}

size_t FeatureDatabase::count_rows_(const std::string& table) const {
  const std::string sql = "SELECT COUNT(*) FROM " + table + ";";
  sqlite3_stmt* sql_stmt;

  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t count = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    count = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return count;
}

size_t FeatureDatabase::sum_column_(const std::string& column,
                                    const std::string& table) const {
  const std::string sql = "SELECT SUM(" + column + ") FROM " + table + ";";
  sqlite3_stmt* sql_stmt;

  SQLITE3_CALL(sqlite3_prepare_v2(database_, sql.c_str(), -1, &sql_stmt, 0));

  size_t sum = 0;
  const int rc = SQLITE3_CALL(sqlite3_step(sql_stmt));
  if (rc == SQLITE_ROW) {
    sum = static_cast<size_t>(sqlite3_column_int64(sql_stmt, 0));
  }

  SQLITE3_CALL(sqlite3_finalize(sql_stmt));

  return sum;
}

}  // end namespace colmap
