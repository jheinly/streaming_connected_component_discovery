import os
import sys
import time
import sqlite3
import subprocess

num_expected_args = 5
if len(sys.argv) != num_expected_args + 1:
  print('USAGE: <input_database> <image_indices_subset> <min_num_inliers>')
  print('       <output_database> <create_empty_colmap_database_exe>')
  sys.exit()

input_database_path = sys.argv[1]
image_indices_subset_path = sys.argv[2]
min_num_inliers = int(sys.argv[3])
output_database_path = sys.argv[4]
create_empty_colmap_database_exe = sys.argv[5]

if not os.path.exists(input_database_path):
  print('ERROR: could not find input database,')
  print(input_database_path)
  sys.exit()
if not os.path.exists(image_indices_subset_path):
  print('ERROR: could not find image indices subset,')
  print(image_indices_subset_path)
  sys.exit()
if os.path.exists(output_database_path):
  print('ERROR: output database already exists,')
  print(output_database_path)
  sys.exit()
if not os.path.exists(create_empty_colmap_database_exe):
  print('ERROR: could not find create_empty_colmap_database executable,')
  print(create_empty_colmap_database_exe)
  sys.exit()

def image_index_to_colmap_image_id(image_index):
  return int(image_index)

def colmap_pair_id_to_image_ids(pair_id):
  MAX_NUM_IMAGES = int(2147483647)
  image_id2 = pair_id % MAX_NUM_IMAGES;
  image_id1 = int(round((pair_id - image_id2) / MAX_NUM_IMAGES))
  return (image_id1, image_id2)

def colmap_image_ids_to_pair_id(image_id1, image_id2):
  MAX_NUM_IMAGES = int(2147483647)
  if image_id1 < image_id2:
    return image_id1 * MAX_NUM_IMAGES + image_id2
  else:
    return image_id2 * MAX_NUM_IMAGES + image_id1

image_ids_subset_set = set()
image_ids_subset_list = []
min_image_id = -1
max_image_id = -1
image_indices_subset_file = open(image_indices_subset_path, 'r')
for line in image_indices_subset_file:
  line = line.strip()
  if len(line) == 0:
    continue
  image_index = int(line)
  image_id = image_index_to_colmap_image_id(image_index)
  if image_id < min_image_id or min_image_id == -1:
    min_image_id = image_id
  if image_id > max_image_id or max_image_id == -1:
    max_image_id = image_id
  image_ids_subset_set.add(image_id)
  image_ids_subset_list.append(image_id)
image_indices_subset_file.close()

subprocess.call([create_empty_colmap_database_exe, output_database_path])

input_connection = sqlite3.connect(input_database_path)
input_cursor = input_connection.cursor()

output_connection = sqlite3.connect(output_database_path)
output_cursor = output_connection.cursor()

start = time.time()

camera_ids_list = []
for image_id in image_ids_subset_list:
  input_cursor.execute('SELECT * FROM images WHERE image_id=?;', (image_id,))
  row = input_cursor.fetchone()
  if row is None:
    print('ERROR: image ' + str(image_id) + ' missing from database')
  name = row[1]
  if name.find('.') == -1:
    new_name = name[0:3] + '/' + name[3:5] + '/' + name + '.jpg'
    row = (row[0], new_name, row[2], row[3], row[4], row[5], row[6], row[7], row[8])
  output_cursor.execute('INSERT INTO images VALUES (?,?,?,?,?,?,?,?,?);', row)
  camera_ids_list.append(row[2])
output_connection.commit()

elapsed = time.time() - start
print('Images: ' + str(elapsed) + ' sec')

start = time.time()

for camera_id in camera_ids_list:
  input_cursor.execute('SELECT * FROM cameras WHERE camera_id=?;', (camera_id,))
  row = input_cursor.fetchone()
  output_cursor.execute('INSERT INTO cameras VALUES (?,?,?,?,?,?);', row)
output_connection.commit()

elapsed = time.time() - start
print('Cameras: ' + str(elapsed) + ' sec')

num_inlier_matches = 0;
num_inlier_features = 0;

start = time.time()

for image_id in image_ids_subset_list:
  MAX_NUM_IMAGES = int(2147483647)
  range_min = image_id * MAX_NUM_IMAGES + min_image_id
  range_max = image_id * MAX_NUM_IMAGES + max_image_id
  for row in input_cursor.execute('SELECT * FROM inlier_matches WHERE pair_id >= ' + str(range_min) + ' AND pair_id <= ' + str(range_max) + ';'):
    (image_id1, image_id2) = colmap_pair_id_to_image_ids(row[0])
    if (image_id1 in image_ids_subset_set) and (image_id2 in image_ids_subset_set) and row[1] >= min_num_inliers:
      output_cursor.execute('INSERT INTO inlier_matches VALUES (?,?,?,?,?);', row)
      num_inlier_matches += 1
      num_inlier_features += row[1]
output_connection.commit()

elapsed = time.time() - start
print('Inliers: ' + str(elapsed) + ' sec')

print(str(num_inlier_matches) + ' inlier match pairs')
print(str(num_inlier_features / num_inlier_matches) + ' average inlier feature matches')

input_cursor.close()
input_connection.close()
output_cursor.close()
output_connection.close()
