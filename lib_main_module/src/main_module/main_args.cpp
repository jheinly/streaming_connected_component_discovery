#include "main_args.h"
#include <assert/assert.h>
#include <core/file_helper.h>
#include <boost/program_options.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <vector>

class ArgList
{
  public:
    template<typename T>
    void add_arg(
      const std::string & name,
      const T & value,
      const bool is_required,
      const std::string & comment)
    {
      std::stringstream stream;
      stream.precision(6);
      stream << "# " << comment;
      if (is_required)
      {
        stream << " (required)";
      }
      else
      {
        stream << " (optional)";
      }
      stream << std::endl;
      stream << name << "=" << value << std::endl;
      lines.push_back(stream.str());
    }

    void add_blank_line()
    {
      std::stringstream stream;
      stream << std::endl;
      lines.push_back(stream.str());
    }

    void write(std::ostream & stream)
    {
      for (size_t i = 0; i < lines.size(); ++i)
      {
        stream << lines[i];
      }
    }

  private:
    std::vector<std::string> lines;
};

#define ADD_ARG(arg_list_obj, arg_desc_obj, arg_obj, name, default_arg, is_required, comment) \
  add_arg_helper(arg_list_obj, arg_desc_obj, #name, &arg_obj.name, default_arg, is_required, comment)

template<typename T, typename S>
void add_arg_helper(
  ArgList & arg_list,
  boost::program_options::options_description & args_description,
  const std::string & name,
  T * variable,
  const S & default_value,
  const bool is_required,
  const std::string & comment)
{
  if (is_required)
  {
    args_description.add_options()
      (name.c_str(),
       boost::program_options::value(variable)->default_value(default_value)->required(),
       comment.c_str());
  }
  else
  {
    args_description.add_options()
      (name.c_str(),
      boost::program_options::value(variable)->default_value(default_value),
      comment.c_str());
  }
  arg_list.add_arg(name, default_value, is_required, comment);
}

template<typename T>
void parse_comma_separated_list(
  const std::string & str,
  std::vector<T> & values)
{
  std::vector<std::string> strs;
  boost::split(strs, str, boost::is_any_of(","));
  values.clear();
  for (size_t i = 0; i < strs.size(); ++i)
  {
    boost::trim(strs[i]);
    values.push_back(boost::lexical_cast<T>(strs[i]));
  }
}

void handle_main_args(
  const int argc,
  const char ** argv,
  MainArgs & main_args)
{
  try
  {
    std::string input_config_path;

    // Handle command-line input.
    {
      boost::program_options::options_description input_description("Command-line options");
      input_description.add_options()
        ("help",
        "Produce help message")
        ("input_config_path",
        boost::program_options::value<std::string>(&input_config_path)->required(),
        "Set path to input config file")
        ;

      try
      {
        boost::program_options::variables_map input_variables;
        boost::program_options::store(
          boost::program_options::parse_command_line(argc, argv, input_description),
          input_variables);

        if (input_variables.count("help"))
        {
          std::cout << input_description << std::endl;
          exit(EXIT_SUCCESS);
        }

        boost::program_options::notify(input_variables);
      }
      catch (boost::program_options::error & e)
      {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << input_description << std::endl;
        exit(EXIT_FAILURE);
      }
    } // End handling command-line input.

    // Handle config-file input.
    {
      boost::program_options::options_description args_description("Config-file options");

      ArgList arg_list;

      ADD_ARG(arg_list, args_description, main_args, sqlite_database_path, "", true,
        "Path where SQLite database should be created");
      ADD_ARG(arg_list, args_description, main_args, vocab_tree_path, "", true,
        "Path to voc-tree file");
      ADD_ARG(arg_list, args_description, main_args, pcdb_path, "", true,
        "Path to PCDB file");
      ADD_ARG(arg_list, args_description, main_args, image_lists_path, "", true,
        "Path to file containing paths of image lists");
      ADD_ARG(arg_list, args_description, main_args, output_visualization_folder, "", true,
        "Folder where visualizations should be saved");
      ADD_ARG(arg_list, args_description, main_args, output_backup_folder, "", true,
        "Folder where backup files should be saved");
      ADD_ARG(arg_list, args_description, main_args, output_log_folder, "", true,
        "Folder where log information should be saved");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, dim_list_path, "", false,
        "Path to image DIM file");
      ADD_ARG(arg_list, args_description, main_args, focal_list_path, "", false,
        "Path to image FOCAL file");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, num_files_to_skip, 0, false,
        "Number of files to skip processing at the beginning");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, batch_size, 256, true,
        "Streaming batch size");
      ADD_ARG(arg_list, args_description, main_args, max_match_attempts_per_image, 2, true,
        "Maximum number of times to try to match a batch image");
      ADD_ARG(arg_list, args_description, main_args, max_matches_per_connected_component, 1, true,
        "Maximum number of times to successfully match a batch image to the same connected component");
      ADD_ARG(arg_list, args_description, main_args, cluster_discard_rate_in_images, 100000, true,
        "Number of images after which a cluster is expected to have grown by at least 1 image");
      arg_list.add_blank_line();
      
      ADD_ARG(arg_list, args_description, main_args, iconic_selection_enabled, true, true,
        "Whether or not iconic selection should be used for a cluster (1 = yes, 0 = no)");
      ADD_ARG(arg_list, args_description, main_args, cluster_merging_enabled, true, true,
        "Whether or not cluster merging should be used (1 = yes, 0 = no)");
      ADD_ARG(arg_list, args_description, main_args, augmented_visual_words_enabled, true, true,
        "Whether or not the cluster representations should augment their visual words (1 = yes, 0 = no)");
      arg_list.add_blank_line();
      
      ADD_ARG(arg_list, args_description, main_args, main_registration_stats_display_rate_in_batches, 100, true,
        "Number of batches between display of the streaming registration stats.");
      ADD_ARG(arg_list, args_description, main_args, main_visualization_rate_in_batches, 100, true,
        "Number of batches between visualization of the streaming state");
      ADD_ARG(arg_list, args_description, main_args, main_backup_rate_in_batches, 1000, true,
        "Number of batches between backup of the streaming state");
      ADD_ARG(arg_list, args_description, main_args, main_visualization_num_clusters, 100, true,
        "Number of clusters to visualize");
      ADD_ARG(arg_list, args_description, main_args, main_visualization_num_images_per_cluster, 100, true,
        "Number of images per cluster to visualize");
      ADD_ARG(arg_list, args_description, main_args, main_visualization_num_components, 100, true,
        "Number of components to visualize");
      ADD_ARG(arg_list, args_description, main_args, main_visualization_num_images_per_component, 100, true,
        "Number of images per component to visualize");
      ADD_ARG(arg_list, args_description, main_args, main_backup_min_images_per_component, 2, true,
        "Only backup components that have greater than or equal to this number of images in them.");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, num_file_buffers, 5, true,
        "Streaming file buffer size in batches");
      ADD_ARG(arg_list, args_description, main_args, num_image_buffers, 4, true,
        "Streaming image buffer size in batches");
      ADD_ARG(arg_list, args_description, main_args, num_sift_features_buffers, 4, true,
        "Streaming SIFT features buffer size in batches");
      ADD_ARG(arg_list, args_description, main_args, num_visual_words_buffers, 4, true,
        "Streaming visual words buffer size in batches");
      ADD_ARG(arg_list, args_description, main_args, num_output_sift_buffers, 4, true,
        "Streaming output SIFT buffer size in batches");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, num_image_loader_threads, 4, true,
        "Number of streaming image loader threads");
      ADD_ARG(arg_list, args_description, main_args, num_cuda_sift_threads, 4, true,
        "Number of streaming CUDA SIFT threads");
      ADD_ARG(arg_list, args_description, main_args, num_visual_words_threads, 4, true,
        "Number of streaming visual words threads");
      ADD_ARG(arg_list, args_description, main_args, num_main_cpu_threads, 16, true,
        "Number of CPU threads for the main module");
      ADD_ARG(arg_list, args_description, main_args, num_main_gpu_threads, 8, true,
        "Number of GPU threads for the main module");
      ADD_ARG(arg_list, args_description, main_args, num_output_sift_threads, 4, true,
        "Number of GPU threads for the main module");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, sift_gpu_nums_list, "0,1,2,3", true,
        "Comma-separated list of GPU indices for SIFT computation");
      ADD_ARG(arg_list, args_description, main_args, main_gpu_nums_list, "0,1,2,3", true,
        "Comma-separated list of GPU indices for main computation (SIFT matching)");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, max_image_dimension, 640, true,
        "Maximum allowed image dimension, all larger images will be resized to this dimension");
      ADD_ARG(arg_list, args_description, main_args, min_image_dimension, 240, true,
        "Minimum required image dimension, all smaller images will be ignored");
      ADD_ARG(arg_list, args_description, main_args, max_aspect_ratio, 2.0f, true,
        "Maximum required aspect ratio, all images with larger ratios will be ignored");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, max_num_features, 4096, true,
        "Maximum allowed number of SIFT features, any additional features will be ignored");
      ADD_ARG(arg_list, args_description, main_args, min_num_features, 100, true,
        "Minimum required number of SIFT features, any image with fewer features will be ignored");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, min_matching_distance, 0.7f, true,
        "Minimum SIFT matching distance");
      ADD_ARG(arg_list, args_description, main_args, max_matching_ratio, 0.8f, true,
        "Maximum SIFT matching ratio");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, max_num_ransac_iterations, 400, true,
        "Maximum number of RANSAC iterations");
      ADD_ARG(arg_list, args_description, main_args, max_pixel_error_ransac, 4.0f, true,
        "Maximum pixel error in order to be a RANSAC inlier");
      ADD_ARG(arg_list, args_description, main_args, min_num_inliers_for_database, 18, true,
        "Minimum number of inliers in order to store in SQLite database");
      ADD_ARG(arg_list, args_description, main_args, min_num_inliers_for_successful_match, 30, true,
        "Minimum number of inliers in order to be considered a successful match");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, vocab_tree_num_knn, 100, true,
        "Initial number of nearest neighbors to query");
      arg_list.add_blank_line();

      ADD_ARG(arg_list, args_description, main_args, densification_components_path, "", false,
        "Path to the components file to densify, which is typically the last components file in the backup folder");
      ADD_ARG(arg_list, args_description, main_args, densification_clusters_path, "", false,
        "Path to the clusters file to densify, which is typically the last clusters file in the backup folder");
      ADD_ARG(arg_list, args_description, main_args, densification_output_sfm_folder, "", false,
        "Folder where the densified output will be written, typically called 'output_sfm'");
      ADD_ARG(arg_list, args_description, main_args, densification_image_links_folder, "", false,
        "Folder containing the top-level hierarchy (000, 001, etc.) of images, or a folder with symbolic links to all of these top-level folders");
      ADD_ARG(arg_list, args_description, main_args, densification_mapper_project_template_path, "", false,
        "Path to the mapper project template file, where an example is provided in streaming_ipc2sfm_package/data");
      ADD_ARG(arg_list, args_description, main_args, densification_python_exe_path, "", false,
        "Path to this system's python executable");
      ADD_ARG(arg_list, args_description, main_args, densification_mapper_exe_path, "", false,
        "Path to the mapper.exe executable");
      ADD_ARG(arg_list, args_description, main_args, densification_vocab_tree_num_knn, 51, false,
        "Number of nearest neighbors to query and match against in order to densify the connections within the component (add 1 for returning self in query results)");
      ADD_ARG(arg_list, args_description, main_args, densification_max_component_size, 100000, false,
        "The maximum component size that will be densified, any larger components will be ignored");
      ADD_ARG(arg_list, args_description, main_args, densification_min_component_size, 20, false,
        "The minimum component size that will be densified, any smaller components will be ignored");
      ADD_ARG(arg_list, args_description, main_args, densification_min_num_inliers_for_export, 25, false,
        "The minimum number of inliers required for that match to be exported in the sfm task's sqlite database");
      ADD_ARG(arg_list, args_description, main_args, densification_num_cpu_threads, 16, false,
        "The number of CPU threads to use for densification");
      ADD_ARG(arg_list, args_description, main_args, densification_num_gpu_threads, 8, false,
        "The number of GPU threads to use for densification");
      ADD_ARG(arg_list, args_description, main_args, densification_gpu_nums_list, "0,1,2,3", false,
        "The list of GPUs to use during densification");
      ADD_ARG(arg_list, args_description, main_args, densification_sfm_visualization_size, 4, false,
        "The number of images to visualize per sfm task in the summary sfm tasks file");
      ADD_ARG(arg_list, args_description, main_args, densification_visualization_subset_size, 100, false,
        "The number of images to visualize per component in a component's images subset file");

      try
      {
        if (input_config_path.empty())
        {
          std::cerr << "ERROR: input_config_path needs to be specified" << std::endl;
          exit(EXIT_FAILURE);
        }

        if (!core::file_helper::path_exists(input_config_path))
        {
          std::cerr << "WARNING: config file does not exist, attempting to create default file at path:" << std::endl;
          std::cerr << input_config_path << std::endl;
          std::cerr << std::endl;

          std::ofstream output_config_file;
          core::file_helper::open_file(input_config_path, output_config_file);
          arg_list.write(output_config_file);
          output_config_file.close();

          exit(EXIT_FAILURE);
        }

        std::ifstream input_config_file;
        core::file_helper::open_file(input_config_path, input_config_file);

        boost::program_options::variables_map args_variables;
        boost::program_options::store(
          boost::program_options::parse_config_file(input_config_file, args_description),
          args_variables);

        boost::program_options::notify(args_variables);
      }
      catch (boost::program_options::error & e)
      {
        std::cerr << "ERROR: " << e.what() << std::endl;
        std::cerr << std::endl;
        std::cerr << args_description << std::endl;
        exit(EXIT_FAILURE);
      }
    } // End handling config-file input.
  }
  catch (std::exception & e)
  {
    std::cerr << "ERROR: unhandled exception in handle_main_args():" << std::endl;
    std::cerr << e.what() << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.vocab_tree_path))
  {
    std::cerr << "ERROR: vocab_tree_path is not set to a valid path," << std::endl;
    std::cerr << "'" << main_args.vocab_tree_path << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.pcdb_path))
  {
    std::cerr << "ERROR: pcdb_path is not set to a valid path," << std::endl;
    std::cerr << "'" << main_args.pcdb_path << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.image_lists_path))
  {
    std::cerr << "ERROR: image_lists_path is not set to a valid path," << std::endl;
    std::cerr << "'" << main_args.image_lists_path << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.output_visualization_folder))
  {
    std::cerr << "ERROR: output_visualization_folder is not set to a valid folder," << std::endl;
    std::cerr << "'" << main_args.output_visualization_folder << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.output_backup_folder))
  {
    std::cerr << "ERROR: output_backup_folder is not set to a valid folder," << std::endl;
    std::cerr << "'" << main_args.output_backup_folder << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  if (!core::file_helper::path_exists(main_args.output_log_folder))
  {
    std::cerr << "ERROR: output_log_folder is not set to a valid folder," << std::endl;
    std::cerr << "'" << main_args.output_log_folder << "'" << std::endl;
    exit(EXIT_FAILURE);
  }

  parse_comma_separated_list(main_args.sift_gpu_nums_list, main_args.sift_gpu_nums);
  parse_comma_separated_list(main_args.main_gpu_nums_list, main_args.main_gpu_nums);
  parse_comma_separated_list(main_args.densification_gpu_nums_list, main_args.densification_gpu_nums);
}
