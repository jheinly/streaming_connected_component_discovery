streaming_connected_component_discovery
=======================================

This is the code that corresponds to the paper: J. Heinly, J.L. Schönberger, E. Dunn, and J.M. Frahm, "Reconstructing the World* in Six Days *(As Captured by the Yahoo 100 Million Image Dataset)", CVPR 2015.

Website:  
<http://jaredheinly.com/reconstructing_the_world.html>

Necessary Software
------------------

The following software is required in order to build and use this project.

### Visual Studio
If using Windows, make sure that Visual Studio has been run at least once and is able to compile C++ code. If Visual Studio has not been run, CMake will have problems when attempting to identify and use the native Visual Studio compiler. The current version that has been used is VS2013, utilizing the 64-bit compiler.

### CMake
CMake is a tool that manages the building of software on multiple different platforms. A developer need only to write a `CMakeLists.txt` file, and CMake can generate a Visual Studio solution for Windows, a Makefile for *nix, etc. The latest binaries can be obtained from:  
<http://www.cmake.org/download>

### CUDA
Several of the libraries in this project depend on CUDA. The latest version can be downloaded here:  
<https://developer.nvidia.com/cuda-downloads>  
The latest version that has been used is CUDA 6.5

### OpenCV 2.4.x
Download the latest version of OpenCV that is 2.4.x from:  
<http://opencv.org/downloads.html>  
OR  
<http://sourceforge.net/projects/opencvlibrary/files/opencv-win/>  
The latest version that has been used is OpenCV 2.4.9

### Boost
Download the latest version of the Boost binaries from:  
<http://sourceforge.net/projects/boost/files/boost-binaries/>  
NOTE: make sure to download the 64-bit boost binaries that match the version of Visual Studio that you are using.  
The latest version that has been used is Boost 1.56

COLMAP
------

This project can be combined with COLMAP (<https://github.com/colmap/colmap>) to generate 3D models from the discovered connected components. This project already includes a minimal subset of COLMAP to save the connected component results to disk.  
*NOTE:* An old version of COLAMP is included. The latest version will be supported shortly.

Included Software
-----------------

The following software packages have already been included.

### ANN
Approximate nearest neighbor library. Included with VocabTree2.  
<http://www.cs.umd.edu/~mount/ANN/>

### Eigen
Linear algebra library.  
<http://eigen.tuxfamily.org/>

### Google Breakpad
Crash handling system.  
<https://chromium.googlesource.com/breakpad/breakpad/>

### ImageLib
Library included with VocabTree2

### libjpeg-turbo
Fast JPEG library.  
<http://libjpeg-turbo.virtualgl.org/>

### msinttypes
Type definitions for Visual Studio compilers.

### SiftGPU
GPU-enabled SIFT computation.  
<http://cs.unc.edu/~ccwu/siftgpu/>

### SQLite
Lightweight SQL database.  
<https://sqlite.org/>

### VocabTree2
Vocabulary tree library.  
<https://github.com/snavely/VocabTree2>

Compilation Instructions
------------------------

Use the following steps to compile the project.

### CMakeLists
In order to compile the project, a `CMakeLists.txt` file has been provided in the root folder. Simply point CMake at this `CMakeLists.txt` file in order to start the process of building the project.

### Find OpenCV
CMake will request the location of the OpenCV library. Set `OpenCV_DIR` in CMake to the OpenCV folder which contains `OpenCVConfig.cmake`, which is typically the `opencv_root/build` folder.

### Find Boost
CMake will request the location of the Boost library. Set `CMH_BOOST_ROOT_DIR` in CMake to the root Boost path on your system. Then set `CMH_BOOST_LIBRARY_DIR` to the Boost folder that contains the compiled Boost binaries, for instance `boost_root/lib64-msvc-12.0` for VS2013.

### CUDA Compute Capability
In order to generate GPU code for a specific CUDA compute capability, check the `CMH_CUDA_COMPUTE_CAPABILITY_*_ENABLED` options in CMake that correspond to the compute capabilities that you want to compile for and enable.

### CUDA 7.0
CUDA 7.0 allows the option to link to a static version of the CUDA runtime. However, this ability is not supported in the current build files, and so the `CUDA_USE_STATIC_CUDA_RUNTIME` option in CMake should be disabled.

### Compile
At this point, the project should be able to be compiled. Make sure to compile in Release mode, as several libraries invoke executables out of the Release folder.

Execution Instructions
----------------------

Use the following steps to run the project.

### Connected Component Discovery

To begin configuring the project for use on a particular dataset, run the main executable, `streaming_connected_component_discovery.exe`, typically located in the `build/bin/Release` folder. An example for how to run the executable is found in `scripts/examples/run_connected_component_discovery.bat`. Assuming that a `config.txt` file does not already exist at the specified path within the script, this will create a default configuration file, `config.txt`, which is used to specify all of the settings for a particular dataset and computer configuration.

A typical dataset will contain the following folders and files, and will be reflected in the dataset's config file:

+ `dataset/`
  + `backup/`
  + `log/`
  + `output_sfm/`
  + `visualization/`
  + `config.txt`
  + `pcdb.txt`
  + `image_lists.txt`
  + `run_connected_component_discovery.bat`
  + `run_densification.bat`
  + `run_all_mapper_tasks.bat`

Typically, to prepare a dataset for processing, its images should be arranged in a hierarchical format, and optionally distributed across multiple hard drives. The folders of images on disk is described in the `pcdb.txt` file (a sample file is in the `lib_v3d_support/examples` folder). Then, depending on the number of threads that should be used to read images from disk, multiple files should be created where each file contains a list of the images to be read by a single thread. A file containing paths to these multiple files should then be created, and for instance, named `image_lists.txt`.

The `config.txt` file is the primary configuration file and contains most of the settings used to run the pipeline. For instance, paths to the above dataset folders (`backup`, `log`, `visualization`, etc.) will need to be specified, as well as the path to a pretrained vocabulary tree (a sample is provided [here](https://drive.google.com/file/d/0B6A-JDvcxvHgbEp1dnBORXRCWHM/view), as well as trees generated with [VocabTree2](https://github.com/snavely/VocabTree2)), mapper project template (provided in the `data` folder), and sqlite database (which will be created by the executable and should not exist prior running the executable). Additionally, settings for the number of registration attempts per image (`max_match_attempts_per_image`), discard rate (`cluster_discard_rate_in_images`), number of CPU threads (`num_main_cpu_threads`), number of GPU threads (`num_cuda_sift_threads`, `num_main_gpu_threads`), GPU indices (`sift_gpu_nums_list`, `main_gpu_nums_list`), image dimension (`max_image_dimension`), and number of SIFT features (`max_num_features`).

After all necessary settings are changed in the `config.txt` file, `run_connected_component_discovery.bat` can be run again, which will process the dataset and discover its connected components of images.

As the code runs, it will print out statistics about its speed to the console, as well as a log file in `dataset/log`. Additionally, the program will track memory usage statistics, and write those to files in `dataset/log`. Periodically, an HTML visualization of the current image clusters and connected components will be saved to `dataset/visualization`. Also, a backup of the current cluster and component states will be saved to `dataset/backup`.

At the end of execution, the final state of the clusters and components will be saved to `dataset/backup`, and a final visualization will be saved in `dataset/visualization`.

### Connected Component Densification

Once connected component discovery concludes, the densification of the connections within a component can begin. Here, update the `densification_*` settings in the `config.txt` file, and then run the `scripts/examples/run_densification.bat` script.

### Structure-from-Motion

Once densification concludes, structure-from-motion can begin by running the `scripts/examples/run_all_mapper_tasks.bat` script.
