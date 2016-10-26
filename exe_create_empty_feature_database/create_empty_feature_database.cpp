#include <boost/filesystem.hpp>
#include <base2d/feature_database.h>
#include <string>
#include <iostream>
#include <cstdlib>

// TODO: modify util/inttypes.h and change point2D_t to be a uint16_t in order to save space.

int main(int argc, char ** argv)
{
  if (argc != 2)
  {
    std::cerr << "USAGE: <database>" << std::endl;
    return EXIT_FAILURE;
  }

  const std::string database_path = argv[1];

  if (boost::filesystem::exists(database_path))
  {
    std::cerr << "ERROR: a file with the provided name already exists," << std::endl;
    std::cerr << "       please specify a new name for the sqlite database" << std::endl;
    return EXIT_FAILURE;
  }
  
  colmap::FeatureDatabase feature_database;
  feature_database.init(database_path);
  feature_database.close();

  return EXIT_SUCCESS;
}
