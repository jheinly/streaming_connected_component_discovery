'''
This script creats a folder hierarchy for the target folder.
For example, let max_folder_num = 3 and root_folder = D:\images
The following folders will be created:
  D:\images\000\00
  D:\images\000\01
  ...
  D:\images\000\99
  D:\images\001\00
  ...
  D:\images\002\00
  ...
  D:\images\003\00
  ...
  D:\images\003\99
'''
  
import os
import sys

if len(sys.argv) != 3:
  print('usage: <max_folder_num> <root_folder>')
  sys.exit()

firstFolder = 0
lastFolder = int(sys.argv[1])
rootFolder = sys.argv[2]

if not os.path.exists(rootFolder):
  print('Path does not exist: ' + rootFolder)
  sys.exit()

count = 0
folders = []
for i in range(firstFolder,lastFolder+1):
  folder = os.path.join(rootFolder, str(i).zfill(3))
  folders.append(folder)
  if not os.path.exists(folder):
    os.mkdir(folder)
    count += 1

for i in range(100):
  for folder in folders:
    path = os.path.join(folder, str(i).zfill(2))
    if not os.path.exists(path):
      os.mkdir(path)
      count += 1

print('Created ' + str(count) + ' folders')
