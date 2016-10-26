/* 
* Copyright 2011-2012 Noah Snavely, Cornell University
* (snavely@cs.cornell.edu).  All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions
* are met:

* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 
* 2. Redistributions in binary form must reproduce the above
*    copyright notice, this list of conditions and the following
*    disclaimer in the documentation and/or other materials provided
*    with the distribution.
*
* THIS SOFTWARE IS PROVIDED BY NOAH SNAVELY ''AS IS'' AND ANY EXPRESS
* OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
* ARE DISCLAIMED. IN NO EVENT SHALL NOAH SNAVELY OR CONTRIBUTORS BE
* LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
* CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
* OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
* BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
* LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
* (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE
* USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
* DAMAGE.
* 
* The views and conclusions contained in the software and
* documentation are those of the authors and should not be
* interpreted as representing official policies, either expressed or
* implied, of Cornell University.
*
*/

/* VocabLearn.cpp */
/* Driver for learning a vocabulary tree through hierarchical kmeans */

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>

#include <vocab_tree/VocabTree.h>
#include <imagelib/defines.h>

#include <v3d_support/pcdb.h>
#include <v3d_support/sift_parser.h>

#include <boost/lexical_cast.hpp>

#include <core/timer.h>

#include <omp.h>

//#define MAX_ARRAY_SIZE 8388608 // 2 ** 23

int main(int argc, char ** argv)
{
  const int num_expected_args = 8;
  if (argc != num_expected_args + 1)
  {
    std::cerr << "USAGE: <pcdb>" << std::endl;
    std::cerr << "       <image_list>" << std::endl;
    std::cerr << "       <target_num_training_sift>" << std::endl;
    std::cerr << "       <image_skip>" << std::endl;
    std::cerr << "       <tree_depth>" << std::endl;
    std::cerr << "       <tree_branching_factor>" << std::endl;
    std::cerr << "       <num_kmeans_trials>" << std::endl;
    std::cerr << "       <output_filename>" << std::endl;
    return -1;
  }

  std::cout << "VocabLearn" << std::endl;
  std::cout << std::endl;

  const std::string & pcdb_path = argv[1];
  const std::string & image_list_path = argv[2];
  const long long int target_num_training_sift = boost::lexical_cast<long long int>(argv[3]);
  const int image_skip = boost::lexical_cast<int>(argv[4]);
  const int tree_depth = boost::lexical_cast<int>(argv[5]);
  const int tree_branching_factor = boost::lexical_cast<int>(argv[6]);
  const int num_kmeans_trials = boost::lexical_cast<int>(argv[7]);
  const std::string output_path = argv[8];

  v3d_support::PCDB pcdb(pcdb_path, image_list_path);
  const int original_num_images = pcdb.get_num_images();
  const int new_num_images = ((original_num_images - 1) / image_skip) + 1;

  const int sift_dim = 128;
  const long long int target_num_training_sift_per_image = static_cast<long long int>(ceil(double(target_num_training_sift) / new_num_images));
  const long long int new_target_num_training_sift = target_num_training_sift_per_image * new_num_images;

  std::cout << "Target num training SIFT:     " << target_num_training_sift << std::endl;
  std::cout << "New target num training SIFT: " << new_target_num_training_sift << std::endl;
  std::cout << "Tree depth:                   " << tree_depth << std::endl;
  std::cout << "Tree branching factor:        " << tree_branching_factor << std::endl;
  std::cout << "Num visual words:             " << static_cast<int>(pow(double(tree_branching_factor), tree_depth)) << std::endl;
  std::cout << "Num k-means trials:           " << num_kmeans_trials << std::endl;
  std::cout << std::endl;

  core::Timer timer;
  double elapsed = 0;

  long long int num_training_sift_vals = sift_dim * new_target_num_training_sift;

  // Allocate space for all of the training SIFT descriptors.
  std::cout << "Allocating " << num_training_sift_vals * sizeof(unsigned char) << " bytes" << std::endl;
  std::cout << std::endl;
  unsigned char * training_sift = new unsigned char[num_training_sift_vals];

  std::cout << "Loading SIFT descriptors..." << std::endl;
  long long int num_training_sift = 0;
  int num_loaded_images = 0;
  timer.start();

  #pragma omp parallel for shared(pcdb, num_training_sift, num_loaded_images)
  for (int img_idx = 0; img_idx < original_num_images; img_idx += image_skip)
  {
    std::vector<features::SiftKeypoint> keypoints;
    std::vector<features::SiftDescriptorUChar> descriptors;
    std::vector<int> sift_indices;

    // Load SIFT features from disk.
    const std::string sift_path = pcdb.get_indexed_file(v3d_support::PCDB::sift, img_idx);
    v3d_support::sift_parser::read_from_file(sift_path, keypoints, descriptors);

    // Randomly suffle feature order.
    sift_indices.clear();
    for (int i = 0; i < descriptors.size(); ++i)
    {
      sift_indices.push_back(i);
    }
    std::random_shuffle(sift_indices.begin(), sift_indices.end());

    const long long int num_sift_to_load = std::min(target_num_training_sift_per_image, static_cast<long long int>(sift_indices.size()));

    long long int current_training_sift_idx = -1;

    #pragma omp flush(num_training_sift)
    #pragma omp critical
    {
      if (num_loaded_images % 1000 == 0)
      {
        std::cout << num_loaded_images << " / " << new_num_images << std::endl;
      }
      ++num_loaded_images;

      current_training_sift_idx = num_training_sift;
      num_training_sift += num_sift_to_load;
    }
    #pragma omp flush(num_training_sift)

    for (int i = 0; i < num_sift_to_load; ++i)
    {
      memcpy(
        training_sift + sift_dim * current_training_sift_idx, // destination
        descriptors[sift_indices[i]].uchars, // source
        sift_dim); // num bytes

      ++current_training_sift_idx;
    }
  }
  elapsed = timer.elapsed();
  std::cout << "Done loading SIFT: " << elapsed << " sec" << std::endl;
  std::cout << std::endl;

  // The k-means code expects an array of pointers, where each pointer points to
  // a different descriptor.
  std::cout << "Allocating " << num_training_sift * sizeof(unsigned char *) << " bytes" << std::endl;
  std::cout << std::endl;
  unsigned char ** training_sift_ptrs = new unsigned char *[num_training_sift];
  for (long long int i = 0; i < num_training_sift; ++i)
  {
    training_sift_ptrs[i] = training_sift + sift_dim * i;
  }

  VocabTree tree;

  std::cout << "Building vocabulary tree..." << std::endl;
  std::cout << std::endl;
  timer.start();
  tree.Build(static_cast<int>(num_training_sift), sift_dim, tree_depth - 1, tree_branching_factor, num_kmeans_trials, training_sift_ptrs);
  elapsed = timer.elapsed();
  std::cout << "Done building vocabulary tree: " << elapsed << " sec" << std::endl;
  std::cout << std::endl;

  std::cout << "Saving vocabulary tree to disk..." << std::endl;
  std::cout << std::endl;
  timer.start();
  tree.Write(output_path.c_str());
  elapsed = timer.elapsed();
  std::cout << "Done saving vocabulary tree to disk: " << elapsed << " sec" << std::endl;
  std::cout << std::endl;

  //delete [] training_sift_ptrs; // This memory is freed in VocabTree::Build
  delete [] training_sift;

  return 0;
}
