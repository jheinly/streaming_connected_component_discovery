cmake_helper
============

https://github.com/jheinly/cmake_helper

This package is designed to assist in the creation and use of modular C++ code that uses [CMake](http://www.cmake.org/). It really shines when your codebase consists of many separate C++ libraries, each of which have various dependencies on each other or 3rd-party packages. Its main advantages are:

* Don't have to precompile all of your libraries. Instead of setting up and precomiling your libraries, and then making sure that you used compatible compilation settings for your current project (ex. same compiler, Debug vs. Release, 32-bit vs. 64-bit, etc.), **cmake_helper** will automatically set up your libraries to compile along with your project.
* Automatic dependency chaining. While CMake does provide functionality to chain dependencies between targets, it is not fully supported and trivial to use with all projects (ex. header-only libraries, CUDA targets, etc.). **cmake_helper** will automatically propagate dependencies between your libraries and executables making configuration and compilation much simpler.
* Simple to use. Libraries and executables designed with **cmake_helper** can easily be added to your project via a normal find_package() command, and both basic and advanced examples are provided for a variety of project types.
* Useful extras. Also included are automatic scripts that help configure CMake for certain usage scenarios. For instance, **cmake_helper** will automatically help you find your CUDA SDK, configure your CUDA compiler, set up your Boost library directory, add Boost compile definitions, add OpenMP compile flags, set compiler optimization flags, allow easy editing of compiler warning levels, and many other useful features.
