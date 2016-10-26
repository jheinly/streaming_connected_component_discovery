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

/* VocabTreeIO.cpp */
/* I/O functions for vocabulary tree */

#include <assert.h>
#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include "VocabTree.h"

int VocabTreeInteriorNode::Write(FILE *f, int bf, int dim) const {
    WriteNode(f, bf, dim);

    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->Write(f, bf, dim);
        }
    }

    return 0;
}

int VocabTreeInteriorNode::WriteFlat(FILE *f, int bf, int dim) const
{
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->WriteFlat(f, bf, dim);
        }
    }

    return 0;
}

int VocabTreeInteriorNode::WriteASCII(FILE *f, int bf, int dim) const
{
    fprintf(f, "%lu ", m_id);
    for (int i = 0; i < dim; i++)
        fprintf(f, " %d", m_desc[i]);

    fprintf(f, "\n");

    /* Count children */
    int num_children = 0;
    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            num_children++;
        }
    }

    fprintf(f, "%d\n", num_children);

    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL) {
            m_children[i]->WriteASCII(f, bf, dim);
        }
    }

    return 0;
}

int VocabTreeInteriorNode::Read(FILE *f, int bf, int dim, VocabTree * vocab_tree)
{
    char *children = new char[bf];
    float dummy;

    m_desc = new unsigned char[dim];
    fread(m_desc, sizeof(unsigned char), dim, f);
    fread(&dummy, sizeof(float), 1, f);
    fread(children, sizeof(char), bf, f);

    m_children = new VocabTreeNode *[bf];

    for (int i = 0; i < bf; i++) {
        if (children[i] == 0) {
            m_children[i] = NULL;
        } else {
            /* Read the interior flag */
            char interior;
            fread(&interior, sizeof(char), 1, f);

            if (interior == 1) {
                m_children[i] = new VocabTreeInteriorNode();
            } else {
                m_children[i] = new VocabTreeLeaf();
            }
            
            m_children[i]->Read(f, bf, dim, vocab_tree);
        }
    }

    delete [] children;

    return 0;    
}

int VocabTreeInteriorNode::WriteNode(FILE *f, int bf, int dim) const
{
    char *children = new char[bf];
    char interior = 1;
    float dummy = 0.0;

    fwrite(&interior, sizeof(char), 1, f);
    fwrite(m_desc, sizeof(unsigned char), dim, f);
    fwrite(&dummy, sizeof(float), 1, f);

    for (int i = 0; i < bf; i++) {
        if (m_children[i] != NULL)
            children[i] = 1;
        else
            children[i] = 0;
    }

    fwrite(children, sizeof(char), bf, f);

    delete [] children;

    return 0;    
}

int VocabTreeLeaf::Write(FILE *f, int bf, int dim) const {
    return WriteNode(f, bf, dim);
}

int VocabTreeLeaf::WriteFlat(FILE *f, int /*bf*/, int dim) const {
    int num_rows = dim / 16 + (((dim % 16) != 0) ? 1 : 0);
    
    int count = 0;
    for (int i = 0; i < num_rows; i++) {
        for (int j = 0; j < 16; j++) {
            fprintf(f, " %d", m_desc[count]);
            count++;

            if (count >= dim)
                break;
        }
        fprintf(f, "\n");
    }

    return 0;
}

int VocabTreeLeaf::WriteASCII(FILE *f, int /*bf*/, int dim) const {
    fprintf(f, "%lu ", m_id);
    for (int i = 0; i < dim; i++)
        fprintf(f, " %d", m_desc[i]);

    fprintf(f, "\n0\n");

    return 0;
}

int VocabTreeLeaf::Read(FILE *f, int /*bf*/, int dim, VocabTree * vocab_tree)
{
    m_vocab_tree = vocab_tree;
    m_visual_word_index = static_cast<int>(m_vocab_tree->m_leaf_inverted_indices.size());
    m_vocab_tree->m_leaf_inverted_indices.push_back(std::vector<VocabTree::InvertedIndexEntry>());
    m_vocab_tree->m_leaf_inverted_indices_mutexes.push_back(new boost::mutex());

    m_desc = new unsigned char[dim];
    fread(m_desc, sizeof(unsigned char), dim, f);
    float weight;
    fread(&weight, sizeof(float), 1, f);
    m_vocab_tree->m_leaf_weights.push_back(weight);

    int num_images;
    fread(&num_images, sizeof(int), 1, f);

    if (num_images != 0)
    {
      std::cerr << "ERROR: expected 0 images in node" << std::endl;
    }

    return 0;
}

int VocabTreeLeaf::WriteNode(FILE *f, int /*bf*/, int dim) const
{
    char interior = 0;
    fwrite(&interior, sizeof(char), 1, f);
    fwrite(m_desc, sizeof(unsigned char), dim, f);
    fwrite(&m_vocab_tree->m_leaf_weights[m_visual_word_index], sizeof(float), 1, f);

    int num_images = 0;
    fwrite(&num_images, sizeof(int), 1, f);

    return 0;    
}

int VocabTree::Read(const std::string & filename) 
{
    FILE * f = NULL;
#ifdef WIN32
    fopen_s(&f, filename.c_str(), "rb");
#else
    f = fopen(filename.c_str(), "rb");
#endif
    
    if (f == NULL) {
        printf("[VocabTree::Read] Error opening file %s for reading\n",
               filename.c_str());
        return -1;
    }

    /* Read the fields for the tree */
    fread(&m_branch_factor, sizeof(int), 1, f);
    fread(&m_depth, sizeof(int), 1, f);
    fread(&m_dim, sizeof(int), 1, f);    
    
    m_root = new VocabTreeInteriorNode();

    /* Read one byte for the interior field */
    char interior;
    fread(&interior, sizeof(char), 1, f);

    m_root->Read(f, m_branch_factor, m_dim, this);

    //unsigned long next_id = m_root->ComputeIDs(m_branch_factor, 0);
    //unsigned long n = m_root->CountNodes(m_branch_factor);
    //printf("  Next id: %lu == %lu + 1\n", next_id, n);

    m_num_nodes = CountNodes();
    m_num_leaves = CountLeaves();

    fclose(f);

    return 0;
}

int VocabTree::WriteHeader(FILE *f) const
{
    /* Write the fields for the tree */
    fwrite(&m_branch_factor, sizeof(int), 1, f);
    fwrite(&m_depth, sizeof(int), 1, f);
    fwrite(&m_dim, sizeof(int), 1, f);    

    return 0;
}

int VocabTree::Write(const char *filename) const
{
    if (m_root == NULL)
    {
        return -1;
    }

    FILE * f = NULL;
#ifdef WIN32
    fopen_s(&f, filename, "wb");
#else
    f = fopen(filename, "wb");
#endif
    
    if (f == NULL) {
        printf("[VocabTree::Write] Error opening file %s for writing\n",
               filename);
        return -1;
    }

    WriteHeader(f);
    
    m_root->Write(f, m_branch_factor, m_dim);

    fclose(f);

    return 0;
}

int VocabTree::WriteFlat(const char *filename) const
{
    if (m_root == NULL)
    {
        return -1;
    }

    FILE * f = NULL;
#ifdef WIN32
    fopen_s(&f, filename, "w");
#else
    f = fopen(filename, "w");
#endif
    
    if (f == NULL) {
        printf("[VocabTree::WriteFlat] Error opening file %s for writing\n",
               filename);
        return -1;
    }
    
    unsigned long num_leaves = CountLeaves();

    fprintf(f, "%lu %d\n", num_leaves, m_dim);
    m_root->WriteFlat(f, m_branch_factor, m_dim);

    fclose(f);

    return 0;
}

int VocabTree::WriteASCII(const char *filename) const
{
    if (m_root == NULL)
    {
        return -1;
    }

    FILE * f = NULL;
#ifdef WIN32
    fopen_s(&f, filename, "w");
#else
    f = fopen(filename, "w");
#endif
    
    if (f == NULL) {
        printf("[VocabTree::WriteASCII] Error opening file %s for writing\n",
               filename);
        return -1;
    }
    
    unsigned long num_nodes = CountNodes();
    unsigned long num_leaves = CountLeaves();

    fprintf(f, "%lu %lu %d\n", num_nodes, num_leaves, m_dim);
    m_root->WriteASCII(f, m_branch_factor, m_dim);

    fclose(f);

    return 0;
}
