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

/* VocabTree.cpp */
/* Build a vocabulary tree from a set of vectors */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <limits>

#include "VocabTree.h"
#include <imagelib/defines.h>
#include <imagelib/util.h>

#ifdef USE_SSE
  #include <emmintrin.h>
#endif

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

#include <assert/assert.h>

/* Useful utility function for computing the squared distance between
* two vectors a and b of length dim */
static unsigned int vec_diff_normsq(
  int dim, const unsigned char *a, const unsigned char *b)
{
#ifdef USE_SSE

  // Set 16, 8-bit values to zero.
  __m128i zero8 = _mm_set1_epi8(0);

  // Set 8, 16-bit values to zero.
  __m128i zero16 = _mm_set1_epi16(0);

  // Set 4, 32-bit values to zero.
  __m128i sum32 = _mm_set1_epi32(0);

  __m128i a8;
  __m128i b8;
  __m128i a16;
  __m128i b16;
  __m128i sub16;
  __m128i mul16;
  __m128i mul32;

  // We will process 16 values at a time.
  const int dim_16 = dim / 16;
  for (int i = 0; i < dim_16; ++i)
  {
    // Load 16, 8-bit values from each descriptor without assuming anything
    // about the alignment of the underlying pointers.
    a8 = _mm_loadu_si128((__m128i *)(a + 16 * i));
    b8 = _mm_loadu_si128((__m128i *)(b + 16 * i));

    /////////////////////////////////////////////////
    // Process the lower 8 values of the descriptors.

    // Interleave the lower 8, 8-bit values of the descriptors with zeros,
    // effectively converting the 8 values to 16-bit values.
    a16 = _mm_unpacklo_epi8(a8, zero8);
    b16 = _mm_unpacklo_epi8(b8, zero8);

    // Compute the difference between the 8, 16-bit values.
    sub16 = _mm_sub_epi16(a16, b16);

    // Compute the squared difference between the 8, 16-bit values.
    mul16 = _mm_mullo_epi16(sub16, sub16);

    // Interleave the lower 4, 16-bit values with zeros to effectively
    // convert to 4, 32-bit values.
    mul32 = _mm_unpacklo_epi16(mul16, zero16);

    // Add the current 4 values to the running sum that stores 4, 32-bit values.
    sum32 = _mm_add_epi32(sum32, mul32);

    // Interleave the upper 4, 16-bit values with zeros to effectively
    // convert to 4, 32-bit values.
    mul32 = _mm_unpackhi_epi16(mul16, zero16);

    // Add the current 4 values to the running sum that stores 4, 32-bit values.
    sum32 = _mm_add_epi32(sum32, mul32);

    /////////////////////////////////////////////////
    // Process the upper 8 values of the descriptors.

    // Interleave the upper 8, 8-bit values of the descriptors with zeros,
    // effectively converting the 8 values to 16-bit values.
    a16 = _mm_unpackhi_epi8(a8, zero8);
    b16 = _mm_unpackhi_epi8(b8, zero8);

    // Compute the difference between the 8, 16-bit values.
    sub16 = _mm_sub_epi16(a16, b16);

    // Compute the squared difference between the 8, 16-bit values.
    mul16 = _mm_mullo_epi16(sub16, sub16);

    // Interleave the lower 4, 16-bit values with zeros to effectively
    // convert to 4, 32-bit values.
    mul32 = _mm_unpacklo_epi16(mul16, zero16);

    // Add the current 4 values to the running sum that stores 4, 32-bit values.
    sum32 = _mm_add_epi32(sum32, mul32);

    // Interleave the upper 4, 16-bit values with zeros to effectively
    // convert to 4, 32-bit values.
    mul32 = _mm_unpackhi_epi16(mul16, zero16);

    // Add the current 4 values to the running sum that stores 4, 32-bit values.
    sum32 = _mm_add_epi32(sum32, mul32);
  }

  // Copy the running sum of the squared differences to 4, 32-bit values without
  // assuming anything about the alignment of the pointer.
  unsigned int sum[4];
  _mm_storeu_si128((__m128i *)sum, sum32);

  // Manually sum the 4, 32-bit values, and return the result.
  return sum[0] + sum[1] + sum[2] + sum[3];

#else

  int i;
  unsigned int normsq = 0;

  for (i = 0; i < dim; ++i)
  {
    int d = (int) a[i] - (int) b[i];
    normsq += d * d;
  }

  return normsq;

#endif
}

#if 0
VocabTreeInteriorNode::~VocabTreeInteriorNode()
{
  if (m_children != NULL)
  {
    delete [] m_children;
  }

  if (m_desc != NULL)
  {
    delete [] m_desc;
  }
}

VocabTreeLeaf::~VocabTreeLeaf()
{
  if (m_desc != NULL)
  {
    delete [] m_desc;
  }
}
#endif

int VocabTreeInteriorNode::ConvertFeatureToVisualWord(
  const unsigned char * feature,
  const int feature_dim,
  const int branch_factor) const
{
  unsigned int min_dist = std::numeric_limits<unsigned int>::max();
  int best_idx = 0;

  for (int i = 0; i < branch_factor; ++i)
  {
    if (m_children[i] != NULL)
    {
      const unsigned int dist = vec_diff_normsq(feature_dim, m_children[i]->m_desc, feature);

      if (dist < min_dist)
      {
        min_dist = dist;
        best_idx = i;
      }
    }
  }    

  return m_children[best_idx]->ConvertFeatureToVisualWord(
    feature,
    feature_dim,
    branch_factor);
}

int VocabTreeLeaf::ConvertFeatureToVisualWord(
  const unsigned char * /*feature*/,
  const int /*feature_dim*/,
  const int /*branch_factor*/) const
{
  return m_visual_word_index;
}

void VocabTreeInteriorNode::ComputeTFIDFWeights(int bf, double n)
{
  /* Compute TFIDF weights for all leaf nodes (visual words) */
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->ComputeTFIDFWeights(bf, n);
    }
  }
}

void VocabTreeLeaf::ComputeTFIDFWeights(int /*bf*/, double n)
{
  std::vector<VocabTree::InvertedIndexEntry> & current_inverted_indices =
    m_vocab_tree->m_leaf_inverted_indices[m_visual_word_index];
  
  const int len = (int)current_inverted_indices.size();

  if (len > 0)
  {
    m_vocab_tree->m_leaf_weights[m_visual_word_index] =
      static_cast<float>(log((double) n / (double) len));
  }
  else
  {
    m_vocab_tree->m_leaf_weights[m_visual_word_index] = 0.0;
  }

  /* We'll pre-apply weights to the count values (TF scores) in the
  * inverted file.  We took care of this for you. */
  // printf("weight = %0.3f\n", m_weight);
  const float weight = m_vocab_tree->m_leaf_weights[m_visual_word_index];
  for (std::vector<VocabTree::InvertedIndexEntry>::iterator iter = current_inverted_indices.begin();
       iter != current_inverted_indices.end();
       ++iter)
  {
    iter->score *= weight;
  }
}

float ComputeMagnitude(DistanceType dtype, float dim)
{
  switch (dtype)
  {
    case DistanceDot:
      return dim * dim;
    case DistanceMin:
      return dim;
    default:
      printf("[ComputeMagnitude] No case value found!\n");
      return 0.0;
  }
}

/* Implementations of driver functions */

void VocabTree::v2_initialize_thread_safe_storage(
  ThreadSafeStorage & storage) const
{
  storage.histogram.clear();
  storage.histogram_counts.clear();
  storage.scores.clear();

  storage.histogram_counts.resize(m_num_leaves, 0);
}

void VocabTree::v2_convert_features_to_visual_words_thread_safe(
  const int num_features,
  const unsigned char * features,
  std::vector<int> & visual_words) const
{
  visual_words.resize(num_features);

  for (int i = 0; i < num_features; ++i)
  {
    visual_words[i] = m_root->ConvertFeatureToVisualWord(
      features + i * m_dim,
      m_dim,
      m_branch_factor);
  }
}

void VocabTree::v2_add_image_to_database_thread_safe(
  const int image_index,
  const std::vector<int> & visual_words,
  bool normalize_image,
  ThreadSafeStorage & storage)
{
  v2_convert_visual_words_to_histogram_thread_safe(
    visual_words,
    storage.histogram,
    storage.histogram_counts);
  const size_t histogram_size = storage.histogram.size();
  ASSERT(histogram_size > 0);

  int storage_index = -1;
  {
    boost::lock_guard<boost::mutex> auto_lock(m_storage_mutex);
    storage_index = m_database_image_storage.add(
      DatabaseImage(image_index, 1.0f));
    ASSERT(m_image_to_storage_indices.find(image_index) == m_image_to_storage_indices.end());
    m_image_to_storage_indices[image_index] = storage_index;
  }

  float magnitude = 0.0f;
  switch (m_distance_type)
  {
    case DistanceDot:
      for (size_t i = 0; i < histogram_size; ++i)
      {
        const int visual_word = storage.histogram[i].visual_word;
        const int frequency = storage.histogram[i].frequency;
        const float score = static_cast<float>(frequency) * m_leaf_weights[visual_word];
        magnitude += score * score;

        ASSERT(visual_word < m_leaf_inverted_indices.size());
        boost::lock_guard<boost::mutex> auto_lock(*m_leaf_inverted_indices_mutexes[visual_word]);
        m_leaf_inverted_indices[visual_word].push_back(InvertedIndexEntry(storage_index, score));
      }
      magnitude = sqrtf(magnitude);
      break;
    case DistanceMin:
      for (size_t i = 0; i < histogram_size; ++i)
      {
        const int visual_word = storage.histogram[i].visual_word;
        const int frequency = storage.histogram[i].frequency;
        const float score = static_cast<float>(frequency) * m_leaf_weights[visual_word];
        magnitude += score;

        ASSERT(visual_word < m_leaf_inverted_indices.size());
        boost::lock_guard<boost::mutex> auto_lock(*m_leaf_inverted_indices_mutexes[visual_word]);
        m_leaf_inverted_indices[visual_word].push_back(InvertedIndexEntry(storage_index, score));
      }
      break;
    default:
      printf("VocabTree::v2_add_image_to_database - no case value found");
      exit(EXIT_FAILURE);
  }

  if (normalize_image)
  {
    boost::lock_guard<boost::mutex> auto_lock(m_storage_mutex);
    m_database_image_storage[storage_index].inverse_magnitude = 1.0f / magnitude;
  }
}

void VocabTree::v2_add_visual_words_to_image_thread_safe(
  const int image_index,
  const std::vector<int> & new_visual_words,
  bool normalize_image,
  ThreadSafeStorage & storage)
{
  v2_convert_visual_words_to_histogram_thread_safe(
    new_visual_words,
    storage.histogram,
    storage.histogram_counts);
  ASSERT(storage.histogram.size() > 0);

  if (m_image_to_storage_indices.find(image_index) == m_image_to_storage_indices.end())
  {
    std::cout << "ERROR: image_index = " << image_index << std::endl;
  }
  ASSERT(m_image_to_storage_indices.find(image_index) != m_image_to_storage_indices.end());
  const int storage_index = m_image_to_storage_indices[image_index];
  ASSERT(storage_index >= 0);
  ASSERT(storage_index < m_database_image_storage.capacity());

  float magnitude = 0.0f;
  switch (m_distance_type)
  {
    case DistanceDot:
      for (int i = 0; i < static_cast<int>(storage.histogram.size()); ++i)
      {
        const int visual_word = storage.histogram[i].visual_word;
        const int frequency = storage.histogram[i].frequency;
        ASSERT(visual_word >= 0);
        ASSERT(visual_word < m_leaf_weights.size());
        const float score = static_cast<float>(frequency) * m_leaf_weights[visual_word];
        magnitude += score * score;

        ASSERT(visual_word < m_leaf_inverted_indices_mutexes.size());
        boost::lock_guard<boost::mutex> auto_lock(*m_leaf_inverted_indices_mutexes[visual_word]);
        ASSERT(visual_word < m_leaf_inverted_indices.size());
        std::vector<InvertedIndexEntry> & current_inverted_indices = m_leaf_inverted_indices[visual_word];
        bool found = false;
        for (std::vector<InvertedIndexEntry>::iterator iter = current_inverted_indices.begin();
             iter != current_inverted_indices.end();
             ++iter)
        {
          ASSERT(iter->storage_index < m_database_image_storage.capacity());
          if (iter->storage_index == storage_index)
          {
            iter->score += score;
            found = true;
            break;
          }
        }
        if (!found)
        {
          current_inverted_indices.push_back(InvertedIndexEntry(storage_index, score));
        }
      }
      magnitude = sqrtf(magnitude);
      break;
    case DistanceMin:
      for (int i = 0; i < static_cast<int>(storage.histogram.size()); ++i)
      {
        const int visual_word = storage.histogram[i].visual_word;
        const int frequency = storage.histogram[i].frequency;
        ASSERT(visual_word >= 0);
        ASSERT(visual_word < m_leaf_weights.size());
        const float score = static_cast<float>(frequency) * m_leaf_weights[visual_word];
        magnitude += score;

        ASSERT(visual_word < m_leaf_inverted_indices_mutexes.size());
        boost::lock_guard<boost::mutex> auto_lock(*m_leaf_inverted_indices_mutexes[visual_word]);
        ASSERT(visual_word < m_leaf_inverted_indices.size());
        std::vector<InvertedIndexEntry> & current_inverted_indices = m_leaf_inverted_indices[visual_word];
        bool found = false;
        for (std::vector<InvertedIndexEntry>::iterator iter = current_inverted_indices.begin();
             iter != current_inverted_indices.end();
             ++iter)
        {
          ASSERT(iter->storage_index < m_database_image_storage.capacity());
          if (iter->storage_index == storage_index)
          {
            iter->score += score;
            found = true;
            break;
          }
        }
        if (!found)
        {
          current_inverted_indices.push_back(InvertedIndexEntry(storage_index, score));
        }
      }
      break;
    default:
      printf("VocabTree::v2_add_visual_words_to_image - no case value found");
      exit(EXIT_FAILURE);
  }

  if (normalize_image)
  {
    boost::lock_guard<boost::mutex> auto_lock(m_storage_mutex);
    const float existing_magnitude =
      1.0f / m_database_image_storage[storage_index].inverse_magnitude;
    m_database_image_storage[storage_index].inverse_magnitude =
      1.0f / (existing_magnitude + magnitude);
  }
}

void VocabTree::v2_remove_image_from_database_thread_safe(
  const int image_index,
  const std::vector<int> & visual_words,
  ThreadSafeStorage & storage)
{
  v2_convert_visual_words_to_histogram_thread_safe(
    visual_words,
    storage.histogram,
    storage.histogram_counts);

  int storage_index = -1;
  {
    boost::lock_guard<boost::mutex> auto_lock(m_storage_mutex);
    ASSERT(m_image_to_storage_indices.find(image_index) != m_image_to_storage_indices.end());
    storage_index = m_image_to_storage_indices[image_index];
    ASSERT(storage_index >= 0);
    ASSERT(storage_index < m_database_image_storage.capacity());
    m_image_to_storage_indices.erase(image_index);
    m_database_image_storage.remove(storage_index);
  }

  for (size_t i = 0; i < storage.histogram.size(); ++i)
  {
    const int visual_word = storage.histogram[i].visual_word;

    boost::lock_guard<boost::mutex> auto_lock(*m_leaf_inverted_indices_mutexes[visual_word]);
    std::vector<InvertedIndexEntry> & current_inverted_indices = m_leaf_inverted_indices[visual_word];

    for (std::vector<InvertedIndexEntry>::iterator iter = current_inverted_indices.begin();
         iter != current_inverted_indices.end();
         ++iter)
    {
      if (iter->storage_index == storage_index)
      {
        current_inverted_indices.erase(iter);
        break;
      }
    }
  }
}

bool VocabTree::v2_is_image_in_database(
  const int image_index)
{
  boost::lock_guard<boost::mutex> auto_lock(m_storage_mutex);
  const boost::unordered_map<int, int>::iterator found =
    m_image_to_storage_indices.find(image_index);
  if (found == m_image_to_storage_indices.end())
  {
    return false;
  }
  else
  {
    return true;
  }
}

void VocabTree::v2_query_database_thread_safe(
  const std::vector<int> & visual_words,
  std::vector<QueryResult> & query_results,
  const int num_knn,
  bool normalize_query,
  ThreadSafeStorage & storage) const
{
  ASSERT(num_knn > 0);
  ASSERT(num_knn <= m_database_image_storage.num_entries());
  if (num_knn > m_database_image_storage.num_entries())
  {
    std::cerr << "ERROR: v2_query_database_thread_safe() - num_knn is greater than number of entries in voc-tree," << std::endl;
    std::cerr << "  num_knn = " << num_knn << ", num voc-tree entries = " << m_database_image_storage.num_entries() << std::endl;
    exit(EXIT_FAILURE);
  }

  ASSERT(m_database_image_storage.capacity() > 0);
  query_results.resize(m_database_image_storage.capacity());
  for (int i = 0; i < m_database_image_storage.capacity(); ++i)
  {
    query_results[i] = QueryResult(m_database_image_storage[i].image_index, 0.0f);
  }

  v2_convert_visual_words_to_histogram_thread_safe(
    visual_words,
    storage.histogram,
    storage.histogram_counts);
  const size_t histogram_size = storage.histogram.size();
  ASSERT(histogram_size > 0);

  storage.scores.resize(histogram_size);
  for (size_t i = 0; i < histogram_size; ++i)
  {
    const int visual_word = storage.histogram[i].visual_word;
    const int frequency = storage.histogram[i].frequency;
    //ASSERT(visual_word >= 0);
    //ASSERT(visual_word < m_leaf_weights.size());
    storage.scores[i] = static_cast<float>(frequency) * m_leaf_weights[visual_word];
  }

  if (normalize_query)
  {
    float magnitude = 0.0f;
    switch (m_distance_type)
    {
      case DistanceDot:
        for (size_t i = 0; i < histogram_size; ++i)
        {
          const float score = storage.scores[i];
          magnitude += score * score;
        }
        magnitude = sqrtf(magnitude);
        break;
      case DistanceMin:
        for (size_t i = 0; i < histogram_size; ++i)
        {
          magnitude += storage.scores[i];
        }
        break;
      default:
        printf("VocabTree::v2_query_database - no case value found");
        exit(EXIT_FAILURE);
    }

    const float inv_magnitude = 1.0f / magnitude;
    for (size_t i = 0; i < histogram_size; ++i)
    {
      storage.scores[i] *= inv_magnitude;
    }
  }

  for (size_t i = 0; i < histogram_size; ++i)
  {
    const int visual_word = storage.histogram[i].visual_word;
    const float score = storage.scores[i];
    //ASSERT(visual_word >= 0);
    //ASSERT(visual_word < m_leaf_inverted_indices.size());
    const std::vector<InvertedIndexEntry> & current_inverted_indices = m_leaf_inverted_indices[visual_word];

    switch (m_distance_type)
    {
      case DistanceDot:
        for (std::vector<InvertedIndexEntry>::const_iterator idx_iter = current_inverted_indices.begin();
             idx_iter != current_inverted_indices.end();
             ++idx_iter)
        {
          const int storage_index = idx_iter->storage_index;
          //ASSERT(storage_index >= 0);
          //ASSERT(storage_index < query_results.size());
          query_results[storage_index].score +=
            score * idx_iter->score * m_database_image_storage[storage_index].inverse_magnitude;
        }
        break;
      case DistanceMin:
        for (std::vector<InvertedIndexEntry>::const_iterator idx_iter = current_inverted_indices.begin();
             idx_iter != current_inverted_indices.end();
             ++idx_iter)
        {
          const int storage_index = idx_iter->storage_index;
          //ASSERT(storage_index >= 0);
          //ASSERT(storage_index < m_database_image_storage.capacity());
          //ASSERT(storage_index < query_results.size());
          query_results[storage_index].score +=
            MIN(score, idx_iter->score * m_database_image_storage[storage_index].inverse_magnitude);
        }
        break;
    }
  }

  std::partial_sort(
    query_results.begin(),
    query_results.begin() + num_knn,
    query_results.end(),
    VocabTree::QueryResult::greater);
  query_results.resize(num_knn);
}

// NOTE: temp_histogram_counts should have a size equal to the number of visual words,
//       and the values should be initialized to all zeros.
void VocabTree::v2_convert_visual_words_to_histogram_thread_safe(
  const std::vector<int> & visual_words,
  std::vector<HistogramEntry> & histogram,
  std::vector<int> & temp_histogram_counts) const
{
  histogram.clear();

  for (std::vector<int>::const_iterator iter = visual_words.begin();
       iter != visual_words.end();
       ++iter)
  {
    ++temp_histogram_counts[*iter];
  }

  for (std::vector<int>::const_iterator iter = visual_words.begin();
       iter != visual_words.end();
       ++iter)
  {
    const int visual_word = *iter;
    ASSERT(visual_word >= 0);
    ASSERT(visual_word < temp_histogram_counts.size());
    if (temp_histogram_counts[visual_word] != 0)
    {
      histogram.push_back(HistogramEntry(visual_word, temp_histogram_counts[visual_word]));
      temp_histogram_counts[visual_word] = 0;
    }
  }
}

void VocabTree::ComputeTFIDFWeights(unsigned int num_db_images)
{
  if (m_root != NULL)
  {
    // double n = m_root->CountFeatures(m_branch_factor);
    // printf("[VocabTree::ComputeTFIDFWeights] Found %lf features\n", n);
    m_root->ComputeTFIDFWeights(m_branch_factor, num_db_images);
  }
}

unsigned long g_leaf_counter = 0;

void VocabTreeInteriorNode::PopulateLeaves(int bf, int dim, 
                                           VocabTreeNode **leaves)
{
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->PopulateLeaves(bf, dim, leaves);
    }
  }
}

void VocabTreeLeaf::PopulateLeaves(int /*bf*/, int /*dim*/, 
                                   VocabTreeNode **leaves)
{
  leaves[g_leaf_counter++] = this;
}

int VocabTree::Flatten()
{
  if (m_root == NULL)
  {
    return -1;
  }

  // ANNpointArray pts = annAllocPts(num_leaves, dim);

  // m_root->FillDescriptors(num_leaves, dim, pts[0]);

  int num_leaves = CountLeaves();

  VocabTreeFlatNode *new_root = new VocabTreeFlatNode;
  new_root->m_children = new VocabTreeNode *[num_leaves];

  for (int i = 0; i < num_leaves; ++i)
  {
    new_root->m_children[i] = NULL;
  }

  g_leaf_counter = 0;
  m_root->PopulateLeaves(m_branch_factor, m_dim, new_root->m_children);
  new_root->BuildANNTree(num_leaves, m_dim);
  new_root->m_desc = new unsigned char[m_dim];
  memset(new_root->m_desc, 0, m_dim);
  new_root->m_id = 0;

  m_root = new_root;

  /* Reset the branch factor */
  m_branch_factor = num_leaves;

  return 0;
}

VocabTree::~VocabTree()
{
  for (size_t i = 0; i < m_leaf_inverted_indices_mutexes.size(); ++i)
  {
    delete m_leaf_inverted_indices_mutexes[i];
  }
  m_leaf_inverted_indices_mutexes.resize(0);
}
