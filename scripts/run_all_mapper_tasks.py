import subprocess
import time
import sys
import os

num_expected_args = 2
if len(sys.argv) != 1 + num_expected_args:
  print('USAGE: <all_mapper_tasks.txt> <num_processes>')
  sys.exit()

mapper_tasks_path = sys.argv[1]
num_threads = int(sys.argv[2])

start_time = time.time()

mapper_tasks_file = open(mapper_tasks_path, 'r')
mapper_tasks = []
for mapper_task in mapper_tasks_file:
  mapper_task = mapper_task.strip()
  if not mapper_task:
    continue
  mapper_tasks.append(mapper_task)
mapper_tasks_file.close()
num_tasks = len(mapper_tasks)

print(str(num_tasks) + ' tasks')
print(str(num_threads) + ' threads')
print(' ')

num_started = 0
num_finished = 0

threads = [None] * num_threads
task_nums = [-1] * num_threads

while num_finished < num_tasks:
  for i in range(num_threads):
    if threads[i] is not None and threads[i].poll() is not None:
      num_finished += 1
      threads[i] = None
      task_path = mapper_tasks[task_nums[i]]
      task_name = os.path.basename(os.path.dirname(task_path))
      print('FINISHED ' + str(num_finished) + ' / ' + str(num_tasks) + ', TASK ' + task_name)
      task_nums[i] = -1
    if threads[i] is None and num_started < num_tasks:
      threads[i] = subprocess.Popen([mapper_tasks[num_started]])
      task_nums[i] = num_started
      num_started += 1
      print('STARTED ' + str(num_started) + ' / ' + str(num_tasks))
  time.sleep(1)

end_time = time.time()
print(' ')
print('TOTAL TIME: ' + str((end_time - start_time)) + ' sec')
print(' ')
