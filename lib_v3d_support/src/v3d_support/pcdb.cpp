#include <v3d_support/pcdb.h>
#include <boost/lexical_cast.hpp>
#include <boost/algorithm/string.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>

v3d_support::PCDB::PCDB()
{
  init_file_types();
}

v3d_support::PCDB::PCDB(
  const std::string & pcdb_file_path)
{
  init_file_types();
  load_pcdb_file(pcdb_file_path);
}

v3d_support::PCDB::PCDB(
  const std::string & pcdb_file_path,
  const std::string & image_list_file_path)
{
  init_file_types();
  load_pcdb_file(pcdb_file_path);
  load_image_list_file(image_list_file_path);
}

v3d_support::PCDB::~PCDB()
{}

void v3d_support::PCDB::load_pcdb_file(
  const std::string & pcdb_file_path)
{
  m_paths.clear();
  m_file_extensions.clear();
  m_num_digits_per_directory.clear();

  m_paths.resize(MAX_NUM_FILE_TYPES);
  m_file_extensions.resize(MAX_NUM_FILE_TYPES);
  m_num_digits_per_directory.resize(MAX_NUM_FILE_TYPES);

  std::ifstream file(pcdb_file_path.c_str());
  if (file.fail())
  {
    std::cerr << "ERROR: PCDB::load_pcdb_file() - failed to open file for reading," << std::endl;
    std::cerr << pcdb_file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  // Read the first line and check for the PCDB header.
  std::string line;
  std::getline(file, line);
  if (file.fail())
  {
    std::cerr << "ERROR: PCDB::load_pcdb_file() - the file appears to be empty," << std::endl;
    std::cerr << pcdb_file_path << std::endl;
    file.close();
    exit(EXIT_FAILURE);
  }
  if (line != "#DB_VERSION_0")
  {
    std::cerr << "ERROR: PCDB::load_pcdb_file() - unrecognized first line in file," << std::endl;
    std::cerr << "\"" << line << ", expected \"#DB_VERSION_0\"" << std::endl;
    file.close();
    exit(EXIT_FAILURE);
  }

  std::string token;
  std::string original_line;
  std::getline(file, line);
  while (!file.fail())
  {
    // Prepare the line for processing by removing any leading or trailing whitespace.
    // NOTE: don't convert the line to lowercase at this point as the line may contain
    // a path that is case-sensitive.
    boost::trim(line);
    original_line = line;
    
    // If the line is empty, or contains a comment, skip to the next line.
    if (line.length() == 0)
    {
      std::getline(file, line);
      continue;
    }
    if (line[0] == '#')
    {
      std::getline(file, line);
      continue;
    }

    // NOTE: don't split the line into all of its tokens at this point, and instead opt to
    // remove one token at a time as the line may contain a path that has a space in it.
    token = get_next_token_from_string(line);
    boost::to_lower(token);
    const int file_type_index = file_type_to_index(token);

    token = get_next_token_from_string(line);
    boost::to_lower(token);    
    if (token == "file")
    {
      // This line specifies the path to a single file.
      m_paths[file_type_index].resize(1);
      m_paths[file_type_index][0] = line;
    }
    else if (token == "dir")
    {
      // Store this directory's file extension.
      token = get_next_token_from_string(line);
      m_file_extensions[file_type_index] = token;

      token = get_next_token_from_string(line);
      const int num_subdirectories = boost::lexical_cast<int>(token);
      if (num_subdirectories < 0)
      {
        std::cerr << "ERROR: PCDB::load_pcdb_file() - expected first value to be >= 0," << std::endl;
        std::cerr << original_line << std::endl;
        exit(EXIT_FAILURE);
      }
      else if (num_subdirectories == 0)
      {
        // This file type has all files in one folder.
        m_num_digits_per_directory[file_type_index].resize(1);
        m_num_digits_per_directory[file_type_index][0] = 0;
      }
      else
      {
        // This file type indexes into different folders, so read the number of
        // digits used to specify each one.
        m_num_digits_per_directory[file_type_index].resize(num_subdirectories);
        for (int i = 0; i < num_subdirectories; ++i)
        {
          token = get_next_token_from_string(line);
          const int num_digits = boost::lexical_cast<int>(token);
          if (num_digits < 1)
          {
            std::cerr << "ERROR: PCDB::load_pcdb_file() - expected num digits per folder to be > 0," << std::endl;
            std::cerr << original_line << std::endl;
            exit(EXIT_FAILURE);
          }
          m_num_digits_per_directory[file_type_index][i] = num_digits;
        }
      }
      if (line.length() > 0)
      {
        std::cerr << "ERROR: PCDB::load_pcdb_file() - unexpected tokens at end of line," << std::endl;
        std::cerr << original_line << std::endl;
        exit(EXIT_FAILURE);
      }
    }
    else if (token == "def")
    {
      if (m_num_digits_per_directory[file_type_index].size() == 0)
      {
        std::cerr << "ERROR: PCDB::load_pcdb_file() - \"def\" keyword provided before \"dir\"," << std::endl;
        std::cerr << original_line << std::endl;
        exit(EXIT_FAILURE);
      }
      if (m_num_digits_per_directory[file_type_index][0] == 0)
      {
        // This file type has all files in one folder.
        m_paths[file_type_index].resize(1);
        m_paths[file_type_index][0] = line;
      }
      else
      {
        // This file type is indexed into folders, so read the range that the current line specifies.
        token = get_next_token_from_string(line);
        const int begin = boost::lexical_cast<int>(token);
        token = get_next_token_from_string(line);
        const int end = boost::lexical_cast<int>(token);
        if (begin < 0 || end < 0 || begin > end)
        {
          std::cerr << "ERROR: PCDB::load_pcdb_file() - invalid integer range," << std::endl;
          std::cerr << original_line << std::endl;
          exit(EXIT_FAILURE);
        }

        // If this range specifies new folders, increase the size of m_paths.
        if (end + 1 > static_cast<int>(m_paths[file_type_index].size()))
        {
          m_paths[file_type_index].resize(end + 1);
        }

        // Assign the path to all folders in the range.
        for (int i = begin; i <= end; ++i)
        {
          if (m_paths[file_type_index][i] != "")
          {
            std::cerr << "ERROR: PCDB::load_pcdb_file() - duplicate definition of subfolder range," << std::endl;
            std::cerr << original_line << std::endl;
            exit(EXIT_FAILURE);
          }
          m_paths[file_type_index][i] = line;
        }
      }
    }
    else
    {
      std::cerr << "ERROR: PCDB::load_pcdb_file() - unexpected 2nd token in line," << std::endl;
      std::cerr << original_line << std::endl;
      exit(EXIT_FAILURE);
    }

    std::getline(file, line);
  }
}

void v3d_support::PCDB::load_image_list_file(
  const std::string & image_list_file_path)
{
  m_image_names.clear();
  std::ifstream file(image_list_file_path.c_str());
  if (file.fail())
  {
    std::cerr << "ERROR: PCDB::load_image_list_file() - failed to open file for reading," << std::endl;
    std::cerr << image_list_file_path << std::endl;
    exit(EXIT_FAILURE);
  }

  std::string image_name;
  while (file >> image_name)
  {
    m_image_names.push_back(image_name);
  }

  file.close();
}

std::string v3d_support::PCDB::get_single_file(
  const PCDB::FileType file_type) const
{
  std::string returned_file_path;
  get_single_file(file_type, returned_file_path);
  return returned_file_path;
}

void v3d_support::PCDB::get_single_file(
  const PCDB::FileType file_type,
  std::string & returned_file_path) const
{
  returned_file_path = m_paths[file_type][0];
}

std::string v3d_support::PCDB::get_indexed_file(
  const PCDB::FileType file_type,
  const int image_index) const
{
  std::string returned_file_path;
  get_indexed_file(file_type, image_index, returned_file_path);
  return returned_file_path;
}

void v3d_support::PCDB::get_indexed_file(
  const PCDB::FileType file_type,
  const int image_index,
  std::string & returned_file_path) const
{
  if (image_index >= static_cast<int>(m_image_names.size()))
  {
    std::cerr << "ERROR: requesting PCDB image index which is out of range, " <<
      image_index << std::endl;
    exit(EXIT_FAILURE);
  }
  const std::string & image_name = m_image_names[image_index];
  get_indexed_file(file_type, image_name, returned_file_path);
}

std::string v3d_support::PCDB::get_indexed_file(
  const PCDB::FileType file_type,
  const std::string & image_name) const
{
  std::string returned_file_path;
  get_indexed_file(file_type, image_name, returned_file_path);
  return returned_file_path;
}

void v3d_support::PCDB::get_indexed_file(
  const PCDB::FileType file_type,
  const std::string & image_name,
  std::string & returned_file_path) const
{
  const std::vector<int> & num_digits_per_directory = m_num_digits_per_directory[file_type];

  if (num_digits_per_directory.size() == 0)
  {
    std::cerr << "ERROR: requesting file of type '" << m_file_types[file_type] <<
      "', but this type has not been defined in the current PCDB file" << std::endl;
    exit(EXIT_FAILURE);
  }

  const int first_folder_num_digits = num_digits_per_directory[0];
  int first_folder_num = 0;
  std::string first_folder_str;
  if (first_folder_num_digits > 0)
  {
    first_folder_str = image_name.substr(0, first_folder_num_digits);
    first_folder_num = boost::lexical_cast<int>(first_folder_str);
  }

  if (first_folder_num >= static_cast<int>(m_paths[file_type].size()))
  {
    std::cerr << "ERROR: requesting path for file '" << image_name <<
      "', but it is out of range of the current PCDB file" << std::endl;
    exit(EXIT_FAILURE);
  }

  const std::string & folder = m_paths[file_type][first_folder_num];
  const std::string & extension = m_file_extensions[file_type];

  returned_file_path.clear();
  returned_file_path.append(folder);
  returned_file_path.append("/");
  
  if (first_folder_num_digits > 0)
  {
    returned_file_path.append(first_folder_str);
    returned_file_path.append("/");
    int image_name_position = first_folder_num_digits;
    for (size_t i = 1; i < num_digits_per_directory.size(); ++i)
    {
      const int num_digits = num_digits_per_directory[i];
      returned_file_path.append(image_name.substr(image_name_position, num_digits));
      returned_file_path.append("/");
      image_name_position += num_digits;
    }
  }

  returned_file_path.append(image_name);
  returned_file_path.append(".");
  returned_file_path.append(extension);
}

std::string v3d_support::PCDB::get_image_name(
  const int image_index) const
{
  return m_image_names[image_index];
}

void v3d_support::PCDB::get_image_name(
  const int image_index,
  std::string & returned_image_name) const
{
  returned_image_name = m_image_names[image_index];
}

void v3d_support::PCDB::init_file_types()
{
  // The order of the file types should match the order of PCDB::FileType.
  m_file_types.push_back("image");
  m_file_types.push_back("thumbnail");
  m_file_types.push_back("sift");
  m_file_types.push_back("harris_brief");
  m_file_types.push_back("dim");
  m_file_types.push_back("focal");
  m_file_types.push_back("gps");
  if (m_file_types.size() != MAX_NUM_FILE_TYPES)
  {
    std::cerr << "ERROR: PCDB::init_file_types() needs to be updated to match entries in PCDB::FileType" << std::endl;
    exit(EXIT_FAILURE);
  }
}

int v3d_support::PCDB::file_type_to_index(const std::string & file_type) const
{
  for (int i = 0; i < MAX_NUM_FILE_TYPES; ++i)
  {
    if (file_type == m_file_types[i])
    {
      return i;
    }
  }
  std::cerr << "ERROR: PCDB::file_type_to_index() - unknown file type," << std::endl;
  std::cerr << file_type << std::endl;
  exit(EXIT_FAILURE);
}

std::string v3d_support::PCDB::get_next_token_from_string(std::string & line) const
{
  size_t found = line.find_first_of("\t ");
  if (found == std::string::npos)
  {
    if (line.length() > 0)
    {
      std::string token = line;
      line.clear();
      return token;
    }
    else
    {
      std::cerr << "ERROR: PCDB::get_next_token_from_string() - called with empty string";
      exit(EXIT_FAILURE);
    }
  }
  else
  {
    std::string token = line.substr(0, found);
    line = line.substr(found + 1, std::string::npos);
    boost::trim(line);
    return token;
  }
}
