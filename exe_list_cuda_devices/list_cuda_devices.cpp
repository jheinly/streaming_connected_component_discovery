#include <cuda_helper/cuda_helper.h>
#include <iostream>

int main(int /*argc*/, char ** /*argv*/)
{
  const int num_devices = cuda_helper::get_num_devices();

  std::cout << std::endl;
  std::cout << num_devices << " CUDA Devices Detected" << std::endl;
  std::cout << std::endl;

  for (int i = 0; i < num_devices; ++i)
  {
    std::cout << "  " << i << ": " << cuda_helper::get_device_name(i) << std::endl;
  }

  std::cout << std::endl;

  return 0;
}
