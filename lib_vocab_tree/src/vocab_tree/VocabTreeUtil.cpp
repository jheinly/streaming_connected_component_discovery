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

/* VocabTreeUtil.cpp */
/* Utility functions for vocabulary tree */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "VocabTree.h"

#include <boost/thread/mutex.hpp>
#include <boost/thread/lock_guard.hpp>

unsigned long VocabTreeInteriorNode::CountNodes(int bf) const
{
  unsigned long num_nodes = 0;

  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      num_nodes += m_children[i]->CountNodes(bf);
    }
  }

  return num_nodes + 1;
}

unsigned long VocabTreeInteriorNode::CountLeaves(int bf) const
{
  unsigned long num_leaves = 0;

  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      num_leaves += m_children[i]->CountLeaves(bf);
    }
  }

  return num_leaves;
}

int VocabTreeInteriorNode::ClearDatabase(int bf)
{
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->ClearDatabase(bf);
    }
  }

  return 0;    
}

int VocabTreeInteriorNode::SetConstantLeafWeights(int bf) 
{
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->SetConstantLeafWeights(bf);
    }
  }

  return 0;    
}

unsigned long VocabTreeLeaf::CountNodes(int /*bf*/) const
{
  return 1;
}

unsigned long VocabTreeLeaf::CountLeaves(int /*bf*/) const
{
  return 1;
}

int VocabTreeLeaf::ClearDatabase(int /*bf*/)
{
  boost::lock_guard<boost::mutex> auto_lock(*m_vocab_tree->m_leaf_inverted_indices_mutexes[m_visual_word_index]);
  m_vocab_tree->m_leaf_inverted_indices[m_visual_word_index].clear();
  return 0;    
}

int VocabTreeLeaf::SetInteriorNodeWeight(int /*bf*/, float /*weight*/)
{
  return 0;
}

int VocabTreeLeaf::SetInteriorNodeWeight(int /*bf*/, int /*dist_from_leaves*/,
                                         float /*weight*/)
{
  return 1;
}

int VocabTreeLeaf::SetConstantLeafWeights(int /*bf*/) 
{
  m_vocab_tree->m_leaf_weights[m_visual_word_index] = 1.0f;
  return 0;
}

int VocabTreeInteriorNode::PrintWeights(int depth_curr, int bf) const
{
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->PrintWeights(depth_curr + 1, bf);
    }
  }

  return 0;
}

unsigned long VocabTreeInteriorNode::ComputeIDs(int bf, unsigned long id)
{
  m_id = id;

  unsigned long next_id = id + 1;
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      next_id = m_children[i]->ComputeIDs(bf, next_id);
    }
  }

  return next_id;
}

unsigned long VocabTreeLeaf::ComputeIDs(int /*bf*/, unsigned long id)
{
  m_id = id;
  return id + 1;
}

int VocabTreeLeaf::PrintWeights(int depth_curr, int /*bf*/) const
{
  for (int i = 0; i < depth_curr; ++i)
  {
    printf(" ");
  }

  printf("%0.3f\n", m_vocab_tree->m_leaf_weights[m_visual_word_index]);

  return 0;
}

unsigned long VocabTree::CountNodes() const
{
  if (m_root == NULL)
  {
    return 0;
  }
  else
  {
    return m_root->CountNodes(m_branch_factor);
  }
}

unsigned long VocabTree::CountLeaves() const
{
  if (m_root == NULL)
  {
    return 0;
  }
  else
  {
    return m_root->CountLeaves(m_branch_factor);
  }
}

int VocabTree::PrintWeights() 
{
  if (m_root != NULL)
  {
    m_root->PrintWeights(0, m_branch_factor);
  }

  return 0;
}

int VocabTree::ClearDatabase()
{
  if (m_root != NULL)
  {
    m_root->ClearDatabase(m_branch_factor);
  }

  m_database_image_storage.clear();
  m_image_to_storage_indices.clear();

  return 0;
}

int VocabTree::SetInteriorNodeWeight(float weight)
{
  if (m_root != NULL)
  {
    m_root->SetInteriorNodeWeight(m_branch_factor, weight);
  }

  return 0;
}

int VocabTree::SetInteriorNodeWeight(int dist_from_leaves, float weight)
{
  if (m_root != NULL)
  {
    m_root->SetInteriorNodeWeight(m_branch_factor, 
      dist_from_leaves,weight);
  }

  return 0;
}

int VocabTree::SetConstantLeafWeights() 
{
  if (m_root != NULL)
  {
    m_root->SetConstantLeafWeights(m_branch_factor);
  }

  return 0;
}

int VocabTree::SetDistanceType(DistanceType type)
{
  m_distance_type = type;
  return 0;
}

void VocabTreeInteriorNode::FillDescriptors(int bf, int dim, unsigned long &id,
                                            unsigned char *desc) const
{
  for (int i = 0; i < bf; ++i)
  {
    if (m_children[i] != NULL)
    {
      m_children[i]->FillDescriptors(bf, dim, id, desc);
    }
  }
}

void VocabTreeLeaf::FillDescriptors(int /*bf*/, int dim, unsigned long &id,
                                    unsigned char *desc) const
{
  memcpy(desc + id * dim, m_desc, dim);
  id++;
}
