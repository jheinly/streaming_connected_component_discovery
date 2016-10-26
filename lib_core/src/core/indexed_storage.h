#pragma once
#ifndef INDEXED_STORAGE_H
#define INDEXED_STORAGE_H

#include <vector>

namespace core {

template<typename T>
class IndexedStorage
{
  public:
    IndexedStorage()
    : m_num_entries(0)
    {}
    
    ~IndexedStorage()
    {}
    
    inline int num_entries() const
    { return m_num_entries; }
    
    inline int capacity() const
    { return static_cast<int>(m_storage.size()); }
    
    inline const T & operator[](int index) const
    { return m_storage[index]; }
    
    inline T & operator[](int index)
    { return m_storage[index]; }
    
    int add(const T & t)
    {
      ++m_num_entries;
      if (m_free_indices.size() > 0)
      {
        const int index = m_free_indices.back();
        m_free_indices.pop_back();
        m_storage[index] = t;
        return index;
      }
      
      const int index = static_cast<int>(m_storage.size());
      m_storage.push_back(t);
      return index;
    }
    
    void remove(int index)
    {
      --m_num_entries;
      m_free_indices.push_back(index);
    }
    
    void reserve(int count)
    { m_storage.reserve(count); }
    
    void clear()
    {
      m_num_entries = 0;
      m_storage.clear();
      m_free_indices.clear();
    }
    
  private:
    int m_num_entries;
    std::vector<T> m_storage;
    std::vector<int> m_free_indices;
};

} // namespace core

#endif // INDEXED_STORAGE_H
