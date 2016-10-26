#include <core/file_helper.h>
#include <core/manager_worker_synchronization.h>
#include <core/shared_counter.h>
#include <core/image_dimensions.h>
#include <core/timer.h>
#include <main_module/streaming_ipc2sfm_types.h>
#include <main_module/feature_database_wrapper.h>
#include <main_module/sift_matcher_thread_group.h>
#include <main_module/main_args.h>
#include <estimator/essential_matrix_estimator.h>
#include <estimator/check_for_degenerate_match.h>
#include <v3d_support/pcdb.h>
#include <v3d_support/sift_parser.h>
#include <vocab_tree/VocabTree.h>
#include <base2d/feature_database.h>
#include <base3d/camera_models.h>
#include <boost/thread/thread.hpp>
#include <boost/unordered_map.hpp>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem/operations.hpp>
#include <iostream>
#include <utility>
#include <vector>
#include <string>
#include <algorithm>

struct ReadImageListThreadArgs
{
  std::string image_list_path;
  std::vector<std::string> * image_names;
  core::Checkpoint * checkpoint;
};

void read_image_list_thread_run(
  ReadImageListThreadArgs args)
{
  args.image_names->clear();
  FILE * file = core::file_helper::open_file(args.image_list_path, "r");

  const int line_size = 256;
  char line[line_size];

  const int format_size = 32;
  char format[format_size];
#ifdef WIN32
  sprintf_s(format, format_size, "%%%ds\n", line_size - 1);
#else
  sprintf(format, "%%%ds\n", line_size - 1);
#endif

  for (;;)
  {
#ifdef WIN32
    const int num_read = fscanf_s(file, format, line, line_size);
#else
    const int num_read = fscanf(file, format, line);
#endif
    if (num_read != 1)
    {
      break;
    }
    args.image_names->push_back(line);
  }
  if (!feof(file))
  {
    std::cerr << "ERROR: format error in image list after reading " <<
      args.image_names->size() << " lines," << std::endl;
    std::cerr << args.image_list_path << std::endl;
    fclose(file);
    exit(EXIT_FAILURE);
  }

  fclose(file);

  args.checkpoint->wait();
}

int compute_connected_component_size(
  const component_index_t component_index,
  boost::unordered_map<component_index_t, std::vector<cluster_index_t> > & component_to_cluster_indices,
  boost::unordered_map<cluster_index_t, std::vector<image_index_t> > & cluster_to_image_indices)
{
  size_t size = 0;
  const std::vector<cluster_index_t> & clusters =
    component_to_cluster_indices[component_index];
  for (size_t i = 0; i < clusters.size(); ++i)
  {
    const cluster_index_t cluster_index = clusters[i];
    const std::vector<image_index_t> & images =
      cluster_to_image_indices[cluster_index];
    size += images.size();
  }
  return static_cast<int>(size);
}

void get_connected_component_images(
  const component_index_t component_index,
  std::vector<image_index_t> & component_images,
  boost::unordered_map<component_index_t, std::vector<cluster_index_t> > & component_to_cluster_indices,
  boost::unordered_map<cluster_index_t, std::vector<image_index_t> > & cluster_to_image_indices,
  const int max_num_images = -1)
{
  component_images.clear();

  // Get the image indices for this component.
  const std::vector<cluster_index_t> & clusters =
    component_to_cluster_indices[component_index];
  for (const auto & cluster_index : clusters)
  {
    const std::vector<image_index_t> & images =
      cluster_to_image_indices[cluster_index];
    component_images.insert(
      component_images.end(),
      images.begin(),
      images.end());

    if (max_num_images > 0 && static_cast<int>(component_images.size()) >= max_num_images)
    {
      component_images.resize(max_num_images);
      break;
    }
  }
}

void save_component_images_visualization(
  const std::string & output_path,
  const std::vector<image_index_t> & image_indices,
  const std::vector<std::string> & image_names,
  const v3d_support::PCDB & pcdb,
  const int max_num_images = 0)
{
  std::ofstream file;
  core::file_helper::open_file(output_path, file);

  file << "<html>" << std::endl;
  file << "<head>" << std::endl;
  file << "<title>Component Images</title>" << std::endl;
  file << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">" << std::endl;
  file << "</head>" << std::endl;

  file << "<h1 style=\"display:inline;\">";
  file << image_indices.size() << " Images";
  file << "</h1> <br/>" << std::endl;

  int num_images = 0;
  for (const auto & image_index : image_indices)
  {
    const std::string & image_name = image_names.at(image_index);
    const std::string & thumbnail_path =
      pcdb.get_indexed_file(v3d_support::PCDB::thumbnail, image_name);
    file << "<img src=\"file:///" << thumbnail_path << "\"/>" << std::endl;
    ++num_images;
    if (max_num_images > 0 && num_images == max_num_images)
    {
      break;
    }
  }

  file << "</html>" << std::endl;
  file.close();
}

struct MainThreadArgs
{
  enum Command {AddImagesToVocTree, GetVocTreeKnn, RunGeometricVerification};

  const std::vector<std::string> * image_names;
  const v3d_support::PCDB * pcdb;
  VocabTree * vocab_tree;
  SiftMatcherThreadGroup * sift_matcher_thread_group;
  FeatureDatabaseWrapper * feature_database;
  core::ManagerWorkerSynchronization * syncrhonization;
  Command * command;
  bool * should_exit;
  core::SharedCounter * counter;
  int min_num_inliers_for_export;
  int num_query_knn;

  std::vector<image_index_t> * image_indices_arg;
  std::vector<std::vector<features::SiftKeypoint> > * images_keypoints_arg;
  std::vector<std::vector<features::SiftDescriptorUChar> > * images_descriptors_arg;
  std::vector<std::vector<int> > * images_visual_words_arg;
  std::vector<std::vector<VocabTree::QueryResult> > * query_results_arg;
  std::vector<std::pair<image_index_t, image_index_t> > * match_tasks_arg;
  boost::unordered_map<image_index_t, int> * image_index_to_task_index_arg;
  std::vector<core::ImageDimensions> * images_dimensions_arg;
  std::vector<float> * images_focals_arg;
};

template<typename Keypoint1, typename Keypoint2>
struct FeatureMatchesAccessor
{
  inline const int num_matches() const
  { return m_num_matches; }

  inline const float x1(const int i) const
  { return (*m_keypoints1).at((*m_matches).at(i).first).x; }

  inline const float y1(const int i) const
  { return (*m_keypoints1).at((*m_matches).at(i).first).y; }

  inline const float x2(const int i) const
  { return (*m_keypoints2).at((*m_matches).at(i).second).x; }

  inline const float y2(const int i) const
  { return (*m_keypoints2).at((*m_matches).at(i).second).y; }

  int m_num_matches;
  const std::vector<Keypoint1> * m_keypoints1;
  const std::vector<Keypoint2> * m_keypoints2;
  const std::vector<std::pair<int, int> > * m_matches;
};

void main_thread_run(MainThreadArgs args)
{
  estimator::EssentialMatrixEstimator essential_estimator(
    400,
    25,
    4.0f);

  VocabTree::ThreadSafeStorage thread_safe_storage(args.vocab_tree);

  std::vector<std::pair<int, int> > matches;
  std::vector<int> inlier_match_indices;

  for (;;)
  {
    args.syncrhonization->wait_for_signal_from_manager();

    if (*args.should_exit)
    {
      break;
    }

    if (*args.command == MainThreadArgs::AddImagesToVocTree)
    {
      for (;;)
      {
        const int task_index = args.counter->get_value_and_increment();
        if (task_index >= static_cast<int>(args.image_indices_arg->size()))
        {
          break;
        }

        const image_index_t image_index = args.image_indices_arg->at(task_index);
        const std::string image_name = args.image_names->at(image_index);
        const std::string sift_path = args.pcdb->get_indexed_file(
          v3d_support::PCDB::sift, image_name);

        std::vector<features::SiftKeypoint> & keypoints =
          args.images_keypoints_arg->at(task_index);
        std::vector<features::SiftDescriptorUChar> & descriptors =
          args.images_descriptors_arg->at(task_index);
        std::vector<int> & visual_words =
          args.images_visual_words_arg->at(task_index);

        keypoints.clear();
        descriptors.clear();
        visual_words.clear();

        v3d_support::sift_parser::read_from_file(
          sift_path,
          keypoints,
          descriptors);

        args.vocab_tree->v2_convert_features_to_visual_words_thread_safe(
          static_cast<int>(descriptors.size()),
          descriptors[0].uchars,
          visual_words);

        args.vocab_tree->v2_add_image_to_database_thread_safe(
          image_index,
          visual_words,
          true,
          thread_safe_storage);
      }
    } // End AddImagesToVocTree
    else if (*args.command == MainThreadArgs::GetVocTreeKnn)
    {
      const int num_knn =
        std::min(args.num_query_knn, args.vocab_tree->num_images_in_database());

      for (;;)
      {
        const int task_index = args.counter->get_value_and_increment();
        if (task_index >= static_cast<int>(args.image_indices_arg->size()))
        {
          break;
        }

        const std::vector<int> & query_visual_words =
          args.images_visual_words_arg->at(task_index);

        std::vector<VocabTree::QueryResult> & query_results =
          args.query_results_arg->at(task_index);
        query_results.clear();

        args.vocab_tree->v2_query_database_thread_safe(
          query_visual_words,
          query_results,
          num_knn,
          true,
          thread_safe_storage);
      }
    } // End GetVocTreeKnn
    else if (*args.command == MainThreadArgs::RunGeometricVerification)
    {
      for (;;)
      {
        const int task_index = args.counter->get_value_and_increment();
        if (task_index >= static_cast<int>(args.match_tasks_arg->size()))
        {
          break;
        }

        const std::pair<image_index_t, image_index_t> & match_task =
          args.match_tasks_arg->at(task_index);

        const auto found1 = args.image_index_to_task_index_arg->find(
          match_task.first);
        const auto found2 = args.image_index_to_task_index_arg->find(
          match_task.second);

        ASSERT(found1 != args.image_index_to_task_index_arg->end());
        ASSERT(found2 != args.image_index_to_task_index_arg->end());

        const int task_index1 = found1->second;
        const int task_index2 = found2->second;

        // Attempt to find an existing match between these images.
        {
          boost::unique_lock<boost::mutex> database_lock(args.feature_database->mutex);
          const bool already_exists = args.feature_database->database.exists_inlier_matches(
            colmap::image_t(match_task.first),
            colmap::image_t(match_task.second));
          if (already_exists)
          {
            continue;
          }
        }

        matches.clear();
        int num_matches = 0;

        num_matches = args.sift_matcher_thread_group->match_descriptors(
          args.images_descriptors_arg->at(task_index1).at(0).uchars,
          static_cast<int>(args.images_descriptors_arg->at(task_index1).size()),
          args.images_descriptors_arg->at(task_index2).at(0).uchars,
          static_cast<int>(args.images_descriptors_arg->at(task_index2).size()),
          matches);

        // TODO: revisit this and figure out how to deal with duplicate images
        const bool are_duplicate_images = estimator::check_matches_for_duplicate_images(
          args.images_keypoints_arg->at(task_index1),
          args.images_keypoints_arg->at(task_index2),
          matches);
        if (are_duplicate_images)
        {
          num_matches = 0;
          matches.clear();
        }

        // If this pair has too few matches, skip to the next match task.
        if (num_matches < args.min_num_inliers_for_export)
        {
          /*FeatureMatches colmap_feature_matches(0, 0);
          {
            boost::lock_guard<boost::mutex> database_lock(args.feature_database->mutex);
            args.feature_database->database.write_inlier_matches(
              image_t(match_task.first),
              image_t(match_task.second),
              colmap_feature_matches);
          }*/

          continue;
        }

        inlier_match_indices.clear();

        FeatureMatchesAccessor<features::SiftKeypoint, features::SiftKeypoint> feature_matches_accessor;
        feature_matches_accessor.m_num_matches = num_matches;
        feature_matches_accessor.m_keypoints1 = &args.images_keypoints_arg->at(task_index1);
        feature_matches_accessor.m_keypoints2 = &args.images_keypoints_arg->at(task_index2);
        feature_matches_accessor.m_matches = &matches;

        float essential[9] = { 0, 0, 0, 0, 0, 0, 0, 0, 0 };
        int num_inliers = essential_estimator.estimate(
          feature_matches_accessor,
          args.images_focals_arg->at(task_index1),
          args.images_dimensions_arg->at(task_index1).width,
          args.images_dimensions_arg->at(task_index1).height,
          args.images_focals_arg->at(task_index2),
          args.images_dimensions_arg->at(task_index2).width,
          args.images_dimensions_arg->at(task_index2).height,
          inlier_match_indices,
          essential);

        if (num_inliers < args.min_num_inliers_for_export)
        {
          /*num_inliers = 0;
          inlier_match_indices.clear();*/

          continue;
        }

        const bool is_border_match = estimator::check_inliers_for_border_matches(
          args.images_dimensions_arg->at(task_index1),
          args.images_dimensions_arg->at(task_index2),
          args.images_keypoints_arg->at(task_index1),
          args.images_keypoints_arg->at(task_index2),
          matches,
          inlier_match_indices);
        if (is_border_match)
        {
          /*num_inliers = 0;
          inlier_match_indices.clear();*/

          continue;
        }

        // Save inlier matches to the database.
        colmap::TwoViewGeometry colmap_two_view_geometry;
        colmap_two_view_geometry.config = colmap::TwoViewGeometry::CALIBRATED;
        colmap_two_view_geometry.inlier_matches.resize(num_inliers, 2);
        for (int i = 0; i < num_inliers; ++i)
        {
          colmap_two_view_geometry.inlier_matches(i, 0) = matches.at(inlier_match_indices.at(i)).first;
          colmap_two_view_geometry.inlier_matches(i, 1) = matches.at(inlier_match_indices.at(i)).second;
        }
        {
          boost::lock_guard<boost::mutex> database_lock(args.feature_database->mutex);
          args.feature_database->database.write_inlier_matches(
            colmap::image_t(match_task.first),
            colmap::image_t(match_task.second),
            colmap_two_view_geometry);
        }
      }
    } // End RunGeometricVerification
  }
}

int main(const int argc, const char ** argv)
{
  MainArgs main_args;
  handle_main_args(argc, argv, main_args);

  const std::string export_sqlite_py = "STREAMING_IPC2SFM_PACKAGE_SCRIPTS_FOLDER/export_colmap_database_subset.py";
  const std::string create_empty_sqlite_exe = "CREATE_EMPTY_FEATURE_DATABASE_BIN_FOLDER/create_empty_feature_database.exe";
  const std::string sfm_visualization_path = main_args.densification_output_sfm_folder + "/sfm_tasks.html";
  const std::string output_sfm_prefix = "sfm_";
  const std::string output_visualization_name = "images.html";
  const std::string output_visualization_subset_name = "images_subset.html";
  const std::string output_sqlite_name = "sqlite.db";
  const std::string output_image_indices_name = "image_indices.txt";
  const std::string output_mapper_project_name = "mapper_project.ini";
  const std::string output_batch_name = "run_mapper.bat";
  const std::string output_all_batch_name = "all_mapper_tasks.txt";
  const std::string output_console_name = "mapper_console_output.txt";

  core::Timer total_timer;
  total_timer.start();

  /////////////////////////////////////////////////////////////////////////////
  // Read the components and clusters from disk.

  std::cout << std::endl;
  std::cout << "Reading components and clusters from disk..." << std::endl;

  boost::unordered_map<component_index_t, std::vector<cluster_index_t> > component_to_cluster_indices;
  boost::unordered_map<cluster_index_t, std::vector<image_index_t> > cluster_to_image_indices;
  {
    std::ifstream components_file;
    core::file_helper::open_file(main_args.densification_components_path, components_file);
    std::string line;
    std::vector<cluster_index_t> clusters;
    component_index_t component_index = -1;
    int num_components = 0;
    int num_clusters = 0;
    while (components_file >> line)
    {
      if (line[0] == '#')
      {
        if (clusters.size() > 0)
        {
          ++num_components;
          num_clusters += static_cast<int>(clusters.size());
          component_to_cluster_indices[component_index] = clusters;
          clusters.clear();
        }
        component_index = boost::lexical_cast<component_index_t>(line.substr(1));
      }
      else
      {
        clusters.push_back(boost::lexical_cast<cluster_index_t>(line));
      }
    }
    if (clusters.size() > 0)
    {
      ++num_components;
      num_clusters += static_cast<int>(clusters.size());
      component_to_cluster_indices[component_index] = clusters;
    }
    components_file.close();

    std::cout << "Loaded " << num_components << " components (" << num_clusters << " clusters)" << std::endl;
  }
  {
    std::ifstream clusters_file;
    core::file_helper::open_file(main_args.densification_clusters_path, clusters_file);
    std::string line;
    std::vector<image_index_t> images;
    cluster_index_t cluster_index = -1;
    int num_clusters = 0;
    int num_images = 0;
    while (clusters_file >> line)
    {
      if (line[0] == '#')
      {
        if (images.size() > 0)
        {
          ++num_clusters;
          num_images += static_cast<int>(images.size());
          cluster_to_image_indices[cluster_index] = images;
          images.clear();
        }
        cluster_index = boost::lexical_cast<cluster_index_t>(line.substr(1));
      }
      else
      {
        images.push_back(boost::lexical_cast<image_index_t>(line));
      }
    }
    if (images.size() > 0)
    {
      ++num_clusters;
      num_images += static_cast<int>(images.size());
      cluster_to_image_indices[cluster_index] = images;
    }
    clusters_file.close();
    
    std::cout << "Loaded " << num_clusters << " clusters (" << num_images << " images)" << std::endl;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Read the image lists from disk.

  std::cout << std::endl;
  std::cout << "Reading image names from disk..." << std::endl;

  std::vector<std::string> image_names;
  {
    std::vector<std::string> image_list_names;
    std::ifstream image_lists_file;
    core::file_helper::open_file(main_args.image_lists_path, image_lists_file);
    std::string line;
    for (;;)
    {
      line.clear();
      std::getline(image_lists_file, line);
      if (line.length() == 0)
      {
        break;
      }
      image_list_names.push_back(line);
    }
    image_lists_file.close();

    const int num_threads = static_cast<int>(image_list_names.size());
    core::Checkpoint checkpoint(num_threads + 1);
    std::vector<std::vector<std::string> > image_names_subsets(num_threads);
    boost::thread * threads = new boost::thread[num_threads];
    for (int i = 0; i < num_threads; ++i)
    {
      ReadImageListThreadArgs args;
      args.image_list_path = image_list_names[i];
      args.image_names = &image_names_subsets[i];
      args.checkpoint = &checkpoint;
      threads[i] = boost::thread(read_image_list_thread_run, args);
    }
    checkpoint.wait();
    for (int i = 0; i < num_threads; ++i)
    {
      std::cout << i << ": Read " << image_names_subsets[i].size() << " image names" << std::endl;
      image_names.insert(
        image_names.end(),
        image_names_subsets[i].begin(),
        image_names_subsets[i].end());
    }

    std::cout << "Loaded " << image_names.size() << " image names" << std::endl;
  }

  /////////////////////////////////////////////////////////////////////////////
  // Load PCDB file.

  std::cout << std::endl;
  std::cout << "Loading PCDB file..." << std::endl;

  v3d_support::PCDB pcdb(main_args.pcdb_path);

  /////////////////////////////////////////////////////////////////////////////
  // Load voc-tree.

  std::cout << std::endl;
  std::cout << "Loading voc-tree..." << std::endl;

  VocabTree vocab_tree;
  vocab_tree.Read(main_args.vocab_tree_path);
  vocab_tree.set_distance_type(DistanceMin);
  vocab_tree.SetInteriorNodeWeight(0.0f);
  vocab_tree.SetConstantLeafWeights();
  vocab_tree.ClearDatabase();

  /////////////////////////////////////////////////////////////////////////////
  // Sort the components by their size.

  std::cout << std::endl;
  std::cout << "Sorting components by size..." << std::endl;

  std::vector<std::pair<int, component_index_t> > component_sizes_and_indices;
  for (const auto & entry : component_to_cluster_indices)
  {
    const int component_size = compute_connected_component_size(
      entry.first,
      component_to_cluster_indices,
      cluster_to_image_indices);
    component_sizes_and_indices.push_back(std::make_pair(
      component_size, entry.first));
  }
  std::sort(
    component_sizes_and_indices.begin(),
    component_sizes_and_indices.end(),
    std::greater<std::pair<int, component_index_t> >());

  /////////////////////////////////////////////////////////////////////////////
  // Visualize sorted components.

  std::cout << std::endl;
  std::cout << "Visualizing components..." << std::endl;
  {
    std::ofstream file;
    core::file_helper::open_file(sfm_visualization_path, file);

    std::vector<image_index_t> image_indices;

    file << "<html>" << std::endl;
    file << "<head>" << std::endl;
    file << "<title>Component Images</title>" << std::endl;
    file << "<meta http-equiv=\"Content-Type\" content=\"text/html; charset=iso-8859-1\">" << std::endl;
    file << "</head>" << std::endl;

    int num_output_components = -1;
    for (const auto & component_size_and_index : component_sizes_and_indices)
    {
      ++num_output_components;

      if (component_size_and_index.first > main_args.densification_max_component_size)
      {
        continue;
      }
      if (component_size_and_index.first < main_args.densification_min_component_size)
      {
        break;
      }

      get_connected_component_images(
        component_size_and_index.second,
        image_indices,
        component_to_cluster_indices,
        cluster_to_image_indices,
        main_args.densification_sfm_visualization_size);

      const int buffer_size = 256;
      char buffer[buffer_size];
#ifdef WIN32
      sprintf_s(buffer, buffer_size, "%s%04d", output_sfm_prefix.c_str(), num_output_components);
#else
      sprintf(buffer, "%s%04d", output_sfm_prefix.c_str(), num_output_components);
#endif

      file << "<h1 style=\"display:inline;\">";
      file << buffer << ", " << component_size_and_index.first << " Images";
      file << "</h1> <br/>" << std::endl;

      for (const auto & image_index : image_indices)
      {
        const std::string & image_name = image_names.at(image_index);
        const std::string & thumbnail_path =
          pcdb.get_indexed_file(v3d_support::PCDB::thumbnail, image_name);
        file << "<img src=\"file:///" << thumbnail_path << "\"/>" << std::endl;
      }

      file << "<br/> <hr />" << std::endl;
    }

    file << "</html>" << std::endl;
    file.close();
  }

  /////////////////////////////////////////////////////////////////////////////
  // Iterate over the components.

  std::cout << std::endl;
  std::cout << "Iterating over connected components..." << std::endl;

  core::Timer timer;

  const std::string all_batch_tasks_file_path =
    main_args.densification_output_sfm_folder + std::string("/") + output_all_batch_name;

  std::ofstream all_batch_tasks_file;
  core::file_helper::open_file(all_batch_tasks_file_path, all_batch_tasks_file);

  SiftMatcherThreadGroup sift_matcher_thread_group(
    main_args.densification_num_gpu_threads,
    main_args.densification_gpu_nums,
    main_args.max_num_features,
    0.7f,
    0.8f);

  FeatureDatabaseWrapper sfm_feature_database;
  core::ManagerWorkerSynchronization synchronization(main_args.densification_num_cpu_threads);
  MainThreadArgs::Command command;
  bool should_exit = false;
  core::SharedCounter counter;

  std::vector<image_index_t> image_indices_arg;
  std::vector<std::vector<features::SiftKeypoint> > images_keypoints;
  std::vector<std::vector<features::SiftDescriptorUChar> > images_descriptors;
  std::vector<std::vector<int> > images_visual_words_arg;
  std::vector<std::vector<VocabTree::QueryResult> > query_results_arg;
  std::vector<std::pair<image_index_t, image_index_t> > match_tasks_arg;
  boost::unordered_map<image_index_t, int> image_index_to_task_index;
  std::vector<core::ImageDimensions> images_dimensions;
  std::vector<float> images_focals;

  MainThreadArgs args;
  args.image_names = &image_names;
  args.pcdb = &pcdb;
  args.vocab_tree = &vocab_tree;
  args.sift_matcher_thread_group = &sift_matcher_thread_group;
  args.feature_database = &sfm_feature_database;
  args.syncrhonization = &synchronization;
  args.command = &command;
  args.should_exit = &should_exit;
  args.counter = &counter;
  args.min_num_inliers_for_export = main_args.densification_min_num_inliers_for_export;

  args.image_indices_arg = &image_indices_arg;
  args.images_keypoints_arg = &images_keypoints;
  args.images_descriptors_arg = &images_descriptors;
  args.images_visual_words_arg = &images_visual_words_arg;
  args.query_results_arg = &query_results_arg;
  args.match_tasks_arg = &match_tasks_arg;
  args.image_index_to_task_index_arg = &image_index_to_task_index;
  args.images_dimensions_arg = &images_dimensions;
  args.images_focals_arg = &images_focals;
  args.num_query_knn = main_args.densification_vocab_tree_num_knn;

  boost::thread * main_threads = new boost::thread[main_args.densification_num_cpu_threads];
  for (int i = 0; i < main_args.densification_num_cpu_threads; ++i)
  {
    main_threads[i] = boost::thread(main_thread_run, args);
  }

  synchronization.wait_for_workers_to_be_ready_to_be_signaled();

  int num_output_components = -1;
  for (const auto & component_size_and_index : component_sizes_and_indices)
  {
    ++num_output_components;

    if (component_size_and_index.first > main_args.densification_max_component_size)
    {
      continue;
    }
    if (component_size_and_index.first < main_args.densification_min_component_size)
    {
      break;
    }

    std::cout << std::endl;
    std::cout << "----------------------------------------" << std::endl;
    std::cout << "Connected Component # " << num_output_components << std::endl;
    std::cout << std::endl;

    // ------------------------------------------------------------------------
    // Create a new folder on disk for this component.

    std::string output_folder;
    {
      const int buffer_size = 256;
      char buffer[buffer_size];
#ifdef WIN32
      sprintf_s(buffer, buffer_size, "%s%04d", output_sfm_prefix.c_str(), num_output_components);
#else
      sprintf(buffer, "%s%04d", output_sfm_prefix.c_str(), num_output_components);
#endif
      output_folder = main_args.densification_output_sfm_folder + std::string("/") + std::string(buffer);
      if (!core::file_helper::path_exists(output_folder))
      {
        boost::filesystem::create_directory(output_folder);
      }
    }

    // ------------------------------------------------------------------------
    // Get the image indices for this component, and determine which images
    // have valid SIFT files on disk.
    std::vector<image_index_t> image_indices;
    {
      // Get the image indices for this component.
      std::vector<image_index_t> component_images;
      get_connected_component_images(
        component_size_and_index.second,
        component_images,
        component_to_cluster_indices,
        cluster_to_image_indices);

      std::cout << "Verifying image SIFT..." << std::endl;

      // Determine which image indices have valid SIFT files on disk.
      for (const auto & image_index : component_images)
      {
        const std::string & image_name = image_names.at(image_index);
        const std::string sift_path = pcdb.get_indexed_file(
          v3d_support::PCDB::sift, image_name);
        if (core::file_helper::path_exists(sift_path))
        {
          image_indices.push_back(image_index);
        }
        else
        {
          std::cout << "MISSING SIFT: " << sift_path << std::endl;
        }
      }
    }

    std::cout << image_indices.size() << " images" << std::endl;

    // ------------------------------------------------------------------------
    // Visualize component images.

    std::cout << "Visualizing images..." << std::endl;

    {
      const std::string visualization_path =
        output_folder + std::string("/") + output_visualization_name;
      const std::string visualization_subset_path =
        output_folder + std::string("/") + output_visualization_subset_name;

      save_component_images_visualization(
        visualization_path, image_indices, image_names, pcdb);
      save_component_images_visualization(
        visualization_subset_path, image_indices, image_names, pcdb,
        main_args.densification_visualization_subset_size);
    }

    // ------------------------------------------------------------------------
    // Export SQLite database subset.
    
    std::cout << "Exporting SQLite subset..." << std::endl;
    
    const std::string sfm_sqlite_path =
      output_folder + std::string("/") + output_sqlite_name;
    {
      const std::string image_indices_path =
        output_folder + std::string("/") + output_image_indices_name;
      std::ofstream file;
      core::file_helper::open_file(image_indices_path, file);
      for (const auto & index : image_indices)
      {
        file << index << std::endl;
      }
      file.close();

      const std::string space = " ";
      const std::string command =
        main_args.densification_python_exe_path + space +
        export_sqlite_py + space +
        main_args.sqlite_database_path + space +
        image_indices_path + space +
        boost::lexical_cast<std::string>(main_args.densification_min_num_inliers_for_export) + space +
        sfm_sqlite_path + space +
        create_empty_sqlite_exe;
      system(command.c_str());
    }

    sfm_feature_database.database.init(sfm_sqlite_path);

    // ------------------------------------------------------------------------
    // Add images to voc-tree (get keypoints, descriptors, and visual words).

    std::cout << "Adding images to voc-tree..." << std::endl;

    timer.start();

    command = MainThreadArgs::AddImagesToVocTree;
    image_indices_arg = image_indices;
    images_keypoints.resize(image_indices.size());
    images_descriptors.resize(image_indices.size());
    images_visual_words_arg.resize(image_indices.size());
    counter.reset();
    synchronization.signal_workers();
    synchronization.wait_for_workers_to_be_ready_to_be_signaled();

    timer.stop();
    timer.print("Add Images");
    timer.print_items_per_second(image_indices.size(), "Add Images");

    // ------------------------------------------------------------------------
    // Get voc-tree KNN within component.

    std::cout << "Getting voc-tree KNN..." << std::endl;

    timer.start();

    command = MainThreadArgs::GetVocTreeKnn;
    query_results_arg.resize(image_indices.size());
    counter.reset();
    synchronization.signal_workers();
    synchronization.wait_for_workers_to_be_ready_to_be_signaled();

    timer.stop();
    timer.print("Get KNN");
    timer.print_items_per_second(image_indices.size(), "Get KNN");

    // ------------------------------------------------------------------------
    // Form unique list of match tasks.

    std::cout << "Forming match tasks..." << std::endl;

    match_tasks_arg.clear();
    for (size_t i = 0; i < image_indices.size(); ++i)
    {
      for (size_t j = 0; j < query_results_arg.at(i).size(); ++j)
      {
        std::pair<image_index_t, image_index_t> match_task(
          image_indices.at(i), query_results_arg.at(i).at(j).image_index);

        // Make sure the smaller of the two indices comes first.
        if (match_task.first > match_task.second)
        {
          std::swap(match_task.first, match_task.second);
        }
        else if (match_task.first == match_task.second)
        {
          continue;
        }
        match_tasks_arg.push_back(match_task);
      }
    }

    std::sort(
      match_tasks_arg.begin(),
      match_tasks_arg.end());

    std::vector<std::pair<int, int> >::const_iterator iter = std::unique(
      match_tasks_arg.begin(),
      match_tasks_arg.end());
    match_tasks_arg.resize(iter - match_tasks_arg.begin());

    std::cout << match_tasks_arg.size() << " match tasks" << std::endl;

    // ------------------------------------------------------------------------
    // Run geometric verification on KNN.

    std::cout << "Preparing for geometric verification..." << std::endl;

    image_index_to_task_index.clear();
    images_dimensions.resize(image_indices.size());
    images_focals.resize(image_indices.size());
    for (size_t i = 0; i < image_indices.size(); ++i)
    {
      image_index_to_task_index[image_indices.at(i)] =
        static_cast<int>(i);

      const colmap::ImageData colmap_image = sfm_feature_database.database.read_image(image_indices.at(i));
      const colmap::Camera colmap_camera = sfm_feature_database.database.read_camera(colmap_image.camera_id);
      images_dimensions.at(i).width = static_cast<int>(colmap_camera.width);
      images_dimensions.at(i).height = static_cast<int>(colmap_camera.height);
      images_focals.at(i) = static_cast<float>(colmap_camera.params[colmap::SimpleRadialCameraModel::focal_idxs[0]]);
    }

    std::cout << "Running geometric verification..." << std::endl;

    timer.start();

    command = MainThreadArgs::RunGeometricVerification;
    counter.reset();
    synchronization.signal_workers();
    synchronization.wait_for_workers_to_be_ready_to_be_signaled();

    timer.stop();
    timer.print("Geometric Verification");
    timer.print_items_per_second(match_tasks_arg.size(), "Geometric Verification");

    // ------------------------------------------------------------------------
    // Export SIFT keypoints to SQLite database subset.

    std::cout << "Exporting SIFT keypoints..." << std::endl;

    sfm_feature_database.database.begin_transaction();
    for (size_t i = 0; i < image_indices.size(); ++i)
    {
      const std::vector<features::SiftKeypoint> & keypoints =
        images_keypoints.at(i);
      Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 4, Eigen::RowMajor> > eigen_keypoints(
        &keypoints.at(0).x, keypoints.size(), 4);
      sfm_feature_database.database.write_keypoints(image_indices.at(i), eigen_keypoints);
    }
    sfm_feature_database.database.end_transaction();

    // ------------------------------------------------------------------------
    // Write mapper task batch file.

    std::cout << "Writing mapper task file..." << std::endl;

    {
      const std::string mapper_file_path =
        output_folder + std::string("/") + output_mapper_project_name;

      std::ofstream output;
      core::file_helper::open_file(mapper_file_path, output);

      std::ifstream input;
      core::file_helper::open_file(main_args.densification_mapper_project_template_path, input);

      std::string line;
      while (input >> line)
      {
        if (line == "database_path=")
        {
          output << line << sfm_sqlite_path << std::endl;
        }
        else if (line == "image_path=")
        {
          output << line << main_args.densification_image_links_folder << std::endl;
        }
        else if (line == "export_path=")
        {
          output << line << output_folder << std::endl;
        }
        else
        {
          output << line << std::endl;
        }
      }

      input.close();
      output.close();

      const std::string batch_file_path =
        output_folder + std::string("/") + output_batch_name;
      std::ofstream batch_file;
      core::file_helper::open_file(batch_file_path, batch_file);

      const std::string console_output_path =
        output_folder + std::string("/") + output_console_name;

      batch_file << "@echo off" << std::endl;
      batch_file << main_args.densification_mapper_exe_path;
      batch_file << " --project_path " << mapper_file_path;
      batch_file << " > " << console_output_path << " 2>&1" << std::endl;

      batch_file.close();

      all_batch_tasks_file << batch_file_path << std::endl;
    }

    // ------------------------------------------------------------------------
    // Clear.

    std::cout << "Done." << std::endl;

    vocab_tree.ClearDatabase();
    sfm_feature_database.database.close();
  }

  all_batch_tasks_file.close();

  std::cout << std::endl;
  total_timer.print("TOTAL ELAPSED");
  std::cout << std::endl;

  should_exit = true;
  synchronization.signal_workers();

  return EXIT_SUCCESS;
}
