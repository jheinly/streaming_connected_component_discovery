#pragma once
#ifndef PCDB_H
#define PCDB_H

#include <string>
#include <vector>

namespace v3d_support {

class PCDB
{
  public:
    // NOTE: if this list of file types changes, PCDB::init_file_types()
    // needs to be updated as well.
    enum FileType
    {
      image = 0,
      thumbnail,
      sift,
      harris_brief,
      dim,
      focal,
      gps,
      MAX_NUM_FILE_TYPES // This is used to keep track of the number of different file types.
    };

    explicit PCDB();
    explicit PCDB(
      const std::string & pcdb_file_path);
    explicit PCDB(
      const std::string & pcdb_file_path,
      const std::string & image_list_file_path);
    ~PCDB();

    void load_pcdb_file(
      const std::string & pcdb_file_path);
    void load_image_list_file(
      const std::string & image_list_file_path);

    // Get the path to a single file like dim, focal, or gps.
    std::string get_single_file(
      const FileType file_type) const;
    void get_single_file(
      const FileType file_type,
      std::string & returned_file_path) const;

    // Get the path to an indexed file like image, thumbnail, or sift using
    // an integer index (this assumes an image list has been loaded).
    std::string get_indexed_file(
      const FileType file_type,
      const int image_index) const;
    void get_indexed_file(
      const FileType file_type,
      const int image_index,
      std::string & returned_file_path) const;

    // Get the path to an indexed file like image, thumbnail, or sift using
    // an image name (an image list does not have to be loaded).
    std::string get_indexed_file(
      const FileType file_type,
      const std::string & image_name) const;
    void get_indexed_file(
      const FileType file_type,
      const std::string & image_name,
      std::string & returned_file_path) const;

    // Get the name of an image using an integer index (this assumes an image
    // list has been loaded).
    std::string get_image_name(
      const int image_index) const;
    void get_image_name(
      const int image_index,
      std::string & returned_image_name) const;

    // Get the number of images that are indexed by this object (this assumes
    // an image list has been loaded).
    int get_num_images() const
    { return static_cast<int>(m_image_names.size()); }

  private:
    void init_file_types();
    int file_type_to_index(const std::string & file_type) const;
    std::string get_next_token_from_string(std::string & line) const;

    // This contains string versions of the file types names listed in FileType.
    std::vector<std::string> m_file_types;

    // This contains the list image names loaded from the current image list.
    std::vector<std::string> m_image_names;
    
    // For each file type, this contains a single path, or list of paths.
    // For example, for the image file type, m_paths[image][2] contains
    // the path to folder 002. For the dim file type, m_paths[dim][0]
    // contains the path to the dim file.
    std::vector<std::vector<std::string> > m_paths;

    // This contains the file extensions for each of the file types.
    std::vector<std::string> m_file_extensions;

    // For each file type, this contains a list of the number of digits
    // used to name each directory in the folder hierarchy.
    std::vector<std::vector<int> > m_num_digits_per_directory;
};

} // namespace v3d_support

#endif // PCDB_H
