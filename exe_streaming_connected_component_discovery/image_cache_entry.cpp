#include "image_cache_entry.h"

void ImageCacheEntry::concatenate_all_visual_words(
  std::vector<int> & all_visual_words) const
{
  all_visual_words.clear();
  all_visual_words.reserve(visual_words.size());

  for (size_t i = 0; i < visual_words.size(); ++i)
  {
    all_visual_words.insert(
      all_visual_words.end(),
      visual_words[i].begin(),
      visual_words[i].end());
  }
}
