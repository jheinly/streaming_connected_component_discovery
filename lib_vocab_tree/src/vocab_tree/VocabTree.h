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

/* VocabTree.h */
/* Vocabulary tree classes */

#ifndef __vocab_tree_h__
#define __vocab_tree_h__

// Using SSE currently gives around a 2x speedup.
// TODO: find references to m_desc, and enforce 16-byte allignment (_aligned_malloc, aligned_alloc, etc).
#define USE_SSE

#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>

#include "ANN_char/ANN.h"

#include <core/indexed_storage.h>

#include <boost/thread/mutex.hpp>
#include <boost/unordered_map.hpp>

/* Types of distances supported */
typedef enum {
    DistanceDot  = 0,
    DistanceMin = 1
} DistanceType;

/* ImageCount class used in the inverted file */
class ImageCount {
public:
    ImageCount() : m_index(0), m_count(0.0) { }
    ImageCount(unsigned int index, float count) : 
        m_index(index), m_count(count) { }

    unsigned int m_index; /* Index of the database image this entry
                           * corresponds to */
    float m_count; /* (Weighted, normalized) count of how many times this
                    * feature appears */
};

// Forward declaration of VocabTree.
class VocabTree;

/* Abstract class for a node of the vocabulary tree */
class VocabTreeNode {
public:
    /* Constructor */
    VocabTreeNode() : m_desc(NULL) { }
    /* Destructor */
    virtual ~VocabTreeNode() { }

    /* I/O routines */
    virtual int Read(FILE *f, int bf, int dim, VocabTree * vocab_tree) = 0;
    virtual int WriteNode(FILE *f, int bf, int dim) const = 0;
    virtual int Write(FILE *f, int bf, int dim) const = 0;
    virtual int WriteFlat(FILE *f, int bf, int dim) const = 0;
    virtual int WriteASCII(FILE *f, int bf, int dim) const = 0;

    virtual int ConvertFeatureToVisualWord(
      const unsigned char * feature,
      const int feature_dim,
      const int branch_factor) const = 0;

    /* Recursively build the vocabulary tree via kmeans clustering */
    /* Inputs: 
     *   n      : number of features to cluster
     *   dim    : dimensionality of each feature vector (e.g., 128)
     *   depth  : total depth of the tree to create
     *   depth_curr : current depth
     *   bf     : branching factor of the tree (children per node)
     *   restarts   : number of random restarts during clustering
     *   v      : list of arrays representing the features
     *
     *   means      : work array for storing means that get passed to kmeans
     *   clustering : work array for storing clustering in kmeans
     */
    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr, 
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering) = 0;

    /* Compute TFIDF weights for each visual word
     * 
     * Inputs:
     *   bf : branching factor of the tree
     *   n  : number of documents in the database 
     */
    virtual void ComputeTFIDFWeights(int /*bf*/, double /*n*/)
        {}

    /* Fill a memory buffer with the descriptors scored in the leaves
     * of the tree */
    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const = 0;
    virtual void PopulateLeaves(int bf, int dim, 
                                VocabTreeNode **leaves) = 0;

    /* Utility functions */
    virtual int PrintWeights(int /*depth_curr*/, int /*bf*/) const
        { return 0; }
    virtual unsigned long ComputeIDs(int /*bf*/, unsigned long /*id*/) 
        { return 0; }        
    virtual unsigned long CountNodes(int bf) const = 0;
    virtual unsigned long CountLeaves(int bf) const = 0;
    virtual int ClearDatabase(int /*bf*/)
        { return 0; }
    virtual int SetInteriorNodeWeight(int /*bf*/, float /*weight*/)
        { return 0; }
    virtual int SetInteriorNodeWeight(int /*bf*/, int /*dist_from_leaves*/, 
                                      float /*weight*/)
        { return 0; }
    virtual int SetConstantLeafWeights(int /*bf*/) 
        { return 0; }
        
    /* Member variables */
    unsigned char *m_desc; /* Descriptor for this node */
    unsigned long m_id;    /* ID of this node */
};

/* Class representing an interior node of the vocab tree */
class VocabTreeInteriorNode : public VocabTreeNode {
public:
    VocabTreeInteriorNode() : VocabTreeNode(), m_children(NULL) { }
    virtual ~VocabTreeInteriorNode() { };

    virtual int Read(FILE *f, int bf, int dim, VocabTree * vocab_tree);
    virtual int WriteNode(FILE *f, int bf, int dim) const;
    virtual int Write(FILE *f, int bf, int dim) const;
    virtual int WriteFlat(FILE *f, int bf, int dim) const;
    virtual int WriteASCII(FILE *f, int bf, int dim) const;

    virtual int ConvertFeatureToVisualWord(
      const unsigned char * feature,
      const int feature_dim,
      const int branch_factor) const;

    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr, 
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering);

    virtual void ComputeTFIDFWeights(int bf, double n);

    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const;
    virtual void PopulateLeaves(int bf, int dim, VocabTreeNode **leaves);

    virtual int PrintWeights(int depth_curr, int bf) const;
    virtual unsigned long ComputeIDs(int bf, unsigned long id);
    virtual unsigned long CountNodes(int bf) const;
    virtual unsigned long CountLeaves(int bf) const;
    
    virtual int ClearDatabase(int bf);
    virtual int SetConstantLeafWeights(int bf);

    /* Member variables */
    VocabTreeNode **m_children; /* Array of child nodes */
};

/* Class representing a leaf of the vocab tree.  Each leaf represents
 * a visual word */
class VocabTreeLeaf : public VocabTreeNode
{
public:
    VocabTreeLeaf() : VocabTreeNode(), m_visual_word_index(-1), m_vocab_tree(NULL)
    {}

    virtual ~VocabTreeLeaf()
    {};

    /* I/O functions */
    virtual int Read(FILE *f, int bf, int dim, VocabTree * vocab_tree);
    virtual int WriteNode(FILE *f, int bf, int dim) const;
    virtual int Write(FILE *f, int bf, int dim) const;
    virtual int WriteFlat(FILE *f, int bf, int dim) const;
    virtual int WriteASCII(FILE *f, int bf, int dim) const;

    virtual int ConvertFeatureToVisualWord(
      const unsigned char * feature,
      const int feature_dim,
      const int branch_factor) const;

    virtual int BuildRecurse(int n, int dim, int depth, int depth_curr, 
                             int bf, int restarts, unsigned char **v,
                             double *means, unsigned int *clustering);

    virtual void ComputeTFIDFWeights(int bf, double n);

    virtual void FillDescriptors(int bf, int dim, unsigned long &id,
                                 unsigned char *desc) const;
    virtual void PopulateLeaves(int bf, int dim, VocabTreeNode **leaves);

    virtual int ClearDatabase(int bf);

    virtual int SetInteriorNodeWeight(int bf, float weight);
    virtual int SetInteriorNodeWeight(int bf, int dist_from_leaves, 
                                      float weight);
    virtual int SetConstantLeafWeights(int bf);

    virtual int PrintWeights(int depth_curr, int bf) const;

    virtual unsigned long ComputeIDs(int bf, unsigned long id);
    virtual unsigned long CountNodes(int bf) const;
    virtual unsigned long CountLeaves(int bf) const;

    /* Member variables */
    int m_visual_word_index;
    VocabTree * m_vocab_tree;
};


class VocabTreeFlatNode : public VocabTreeInteriorNode
{
public:
    VocabTreeFlatNode() : VocabTreeInteriorNode()
    { }

    void BuildANNTree(int num_leaves, int dim);

    ann_1_1_char::ANNkd_tree *m_tree; /* For finding nearest neighbors */
};

class VocabTree
{
  public:
    VocabTree()
    : m_num_nodes(0), m_num_leaves(0), m_distance_type(DistanceMin), m_root(NULL)
    {}

    void set_distance_type(DistanceType distance_type)
    { m_distance_type = distance_type; }

    int get_num_visual_words() const
    { return m_num_leaves; }

    ~VocabTree();

    /* I/O routines */
    int Read(const std::string & filename);
    int WriteHeader(FILE *f) const;
    int Write(const char *filename) const;
    int WriteFlat(const char *filename) const;
    int WriteASCII(const char *filename) const;

    int Flatten(); /* Flatten the tree to a single level */

    /* Build the vocabulary tree using kmeans 
     *
     * Inputs: 
     *  n        : number of features to cluster
     *  dim      : dimensionality of each feature vector (e.g., 128)
     *  depth    : total depth of the tree to create
     *  bf       : desired branching factor of the tree (children per node)
     *  restarts : number of random restarts during clustering
     *  vp       : array of pointers to arrays representing the features
     */
    int Build(int n, int dim, int depth, int bf, int restarts, 
              unsigned char **vp);

    struct HistogramEntry
    {
      HistogramEntry()
      : visual_word(-1),
        frequency(0)
      {}

      HistogramEntry(const int visual_word, const int frequency)
      : visual_word(visual_word),
        frequency(frequency)
      {}

      int visual_word;
      int frequency;
    };

    struct ThreadSafeStorage
    {
      ThreadSafeStorage(const VocabTree * vocab_tree)
      {
        vocab_tree->v2_initialize_thread_safe_storage(*this);
      }

      std::vector<HistogramEntry> histogram;
      std::vector<int> histogram_counts;
      std::vector<float> scores;
    };

    void v2_initialize_thread_safe_storage(
      ThreadSafeStorage & storage) const;

    void v2_convert_features_to_visual_words_thread_safe(
      const int num_features,
      const unsigned char * features,
      std::vector<int> & visual_words) const;

    void v2_add_image_to_database_thread_safe(
      const int image_index,
      const std::vector<int> & visual_words,
      bool normalize_image,
      ThreadSafeStorage & storage);

    void v2_add_visual_words_to_image_thread_safe(
      const int image_index,
      const std::vector<int> & new_visual_words,
      bool normalize_image,
      ThreadSafeStorage & storage);

    void v2_remove_image_from_database_thread_safe(
      const int image_index,
      const std::vector<int> & visual_words,
      ThreadSafeStorage & storage);

    bool v2_is_image_in_database(
      const int image_index);

    struct QueryResult
    {
      QueryResult()
      : image_index(-1),
        score(0.0f)
      {}

      QueryResult(const int image_index, const float score)
      : image_index(image_index),
        score(score)
      {}

      static inline bool greater(const QueryResult & a, const QueryResult & b)
      {
        return a.score > b.score;
      }

      int image_index;
      float score;
    };

    void v2_query_database_thread_safe(
      const std::vector<int> & visual_words,
      std::vector<QueryResult> & query_results,
      const int num_knn,
      bool normalize_query,
      ThreadSafeStorage & storage) const;

    int num_images_in_database() const
    { return m_database_image_storage.num_entries(); }

    /* Given a tree populated with database images, compute the TFIDF
     * weights */
    void ComputeTFIDFWeights(unsigned int num_db_images);

    /* Empty out the database */
    int ClearDatabase();

    /* Utility functions */
    int PrintWeights();
    unsigned long CountNodes() const;
    unsigned long CountLeaves() const;
    int SetInteriorNodeWeight(float weight);
    int SetInteriorNodeWeight(int dist_from_leaves, float weight);
    int SetConstantLeafWeights();
    int SetDistanceType(DistanceType type);

  private:
    // NOTE: temp_histogram_counts should have a size equal to the number of visual words,
    //       and the values should be initialized to all zeros.
    void v2_convert_visual_words_to_histogram_thread_safe(
      const std::vector<int> & visual_words,
      std::vector<HistogramEntry> & histogram,
      std::vector<int> & temp_histogram_counts) const;

    friend class VocabTreeLeaf;

    /* Member variables */
    
    int m_branch_factor;           /* Branching factor for tree */
    int m_depth;                   /* Depth of the tree */
    int m_dim;                     /* Dimension of the descriptors */
    unsigned long m_num_nodes;     /* Number of nodes in the tree */
    int m_num_leaves;              /* Number of leaves (visual words) in the tree */
    DistanceType m_distance_type;  /* Type of the distance measure */
    VocabTreeNode * m_root;         /* Root of the tree */

    struct InvertedIndexEntry
    {
      InvertedIndexEntry()
      : storage_index(-1),
        score(0.0f)
      {}

      InvertedIndexEntry(const int storage_index, const float score)
      : storage_index(storage_index),
        score(score)
      {}

      int storage_index;
      float score;
    };
    std::vector<std::vector<InvertedIndexEntry> > m_leaf_inverted_indices;
    std::vector<boost::mutex *> m_leaf_inverted_indices_mutexes;
    std::vector<float> m_leaf_weights;

    struct DatabaseImage
    {
      DatabaseImage()
      : image_index(-1),
        inverse_magnitude(1.0f)
      {}

      DatabaseImage(const int image_index, const float inverse_magnitude)
      : image_index(image_index),
        inverse_magnitude(inverse_magnitude)
      {}

      int image_index;
      float inverse_magnitude;
    };
    core::IndexedStorage<DatabaseImage> m_database_image_storage;
    boost::unordered_map<int, int> m_image_to_storage_indices;
    boost::mutex m_storage_mutex;
};

#endif /* __vocab_tree_h__ */
