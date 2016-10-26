#pragma once
#ifndef BATCH_MATCH_TASK_H
#define BATCH_MATCH_TASK_H

struct BatchMatchTask
{
  BatchMatchTask()
  : image_index(-1),
    connected_component_index(-1)
  {}

  BatchMatchTask(
    const int image_idx,
    const int connected_component_idx)
  : image_index(image_idx),
    connected_component_index(connected_component_idx)
  {}

  int image_index;
  int connected_component_index;
};

#endif // BATCH_MATCH_TASK_H
