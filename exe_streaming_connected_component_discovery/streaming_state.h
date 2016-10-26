#pragma once
#ifndef STREAMING_STATE_H
#define STREAMING_STATE_H

#include <main_module/streaming_ipc2sfm_types.h>
#include <boost/unordered_map.hpp>
#include <vector>

struct StreamingState
{
  // Map: images -> clusters -> connected components
  boost::unordered_map<image_index_t, cluster_index_t> image_index_to_cluster_index;
  boost::unordered_map<cluster_index_t, component_index_t> cluster_index_to_component_index;

  // Map: connected components -> clusters -> images
  boost::unordered_map<component_index_t, std::vector<cluster_index_t> > component_index_to_cluster_indices;
  boost::unordered_map<cluster_index_t, std::vector<image_index_t> > cluster_index_to_image_indices;

  boost::unordered_map<cluster_index_t, int> cluster_index_to_batch_index;
};

#endif // STREAMING_STATE_H
