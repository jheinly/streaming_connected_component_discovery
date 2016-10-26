#include <cuda_sift/cuda_sift.h>
#include <cuda_helper/cuda_helper.h>
#include <features/sift_support.h>
#include <assert/assert.h>
#include <GL/glew.h>
#include <iostream>
#include <string>
#include <vector>

cuda_sift::CudaSift::CudaSift(
  const int max_image_dimension,
  const int max_num_features,
  const int gpu_num,
  const int port_offset)
  : m_sift_gpu(NULL),
  m_original_keypoints(2 * max_num_features),   // Allocate extra space to handle the case where
  m_original_descriptors(2 * max_num_features), // descriptors have multiple orientations.
  m_sorted_keypoints(max_num_features),
  m_sorted_descriptors(max_num_features),
  m_max_image_dimension(max_image_dimension),
  m_max_num_features(max_num_features),
  m_gpu_num(gpu_num),
  m_sort_indices(2 * max_num_features)
{
  ASSERT(max_image_dimension > 0);
  ASSERT(max_num_features > 0);

#ifndef SERVER_SIFTGPU_ENABLED
  cuda_helper::set_device(gpu_num);
  cuda_helper::initialize_cuda_context();
#endif

  std::vector<std::string> args;

  // Upsample the image to start the first octave at -1.
  args.push_back("-fo");
  args.push_back("-1");

  // Set the verbosity to 0.
  args.push_back("-v");
  args.push_back("0");

  char buffer_cuda[4];
#ifdef WIN32
  sprintf_s(buffer_cuda, 4, "%d", gpu_num);
#else
  sprintf(buffer_cuda, "%d", gpu_num);
#endif
  args.push_back("-nogl");
  args.push_back("-cuda");
  args.push_back(buffer_cuda);

  char buffer_p[32];
#ifdef WIN32
  sprintf_s(buffer_p, 32, "%dx%d", max_image_dimension, max_image_dimension);
#else
  sprintf(buffer_p, "%dx%d", max_image_dimension, max_image_dimension);
#endif
  args.push_back("-p");
  args.push_back(buffer_p);

  char buffer_maxd[16];
#ifdef WIN32
  sprintf_s(buffer_maxd, 16, "%d", max_image_dimension);
#else
  sprintf(buffer_maxd, "%d", max_image_dimension);
#endif
  args.push_back("-maxd");
  args.push_back(buffer_maxd);

  char buffer_tc2[16];
#ifdef WIN32
  sprintf_s(buffer_tc2, 16, "%d", max_num_features);
#else
  sprintf(buffer_tc2, "%d", max_num_features);
#endif
  args.push_back("-tc2");
  args.push_back(buffer_tc2);

  //args.push_back("-loweo");

  const char ** argv = new const char*[args.size()];
  for (size_t i = 0; i < args.size(); ++i)
  {
    argv[i] = args[i].c_str();
  }

#ifdef SERVER_SIFTGPU_ENABLED
  std::cout << "CudaSift: GPU # " << gpu_num << std::endl;
  if (port_offset < 0)
  {
    m_sift_gpu = CreateRemoteSiftGPU(7777 + gpu_num);
  }
  else
  {
    // TODO: this should be revisited
    // Added to support multiple simultaneous instances of streaming_ipc2sfm
    //m_sift_gpu = CreateRemoteSiftGPU(7777 + 4 * gpu_num + port_offset);

    m_sift_gpu = CreateRemoteSiftGPU(7777 + port_offset);
  }
#else
  m_sift_gpu = new SiftGPU();
#endif
  m_sift_gpu->ParseParam(static_cast<int>(args.size()), (char **)argv);

  int sift_gpu_supported = m_sift_gpu->CreateContextGL();
  if (sift_gpu_supported != SiftGPU::SIFTGPU_FULL_SUPPORTED)
  {
    std::cerr << "ERROR: CudaSift - SiftGPU not fully supported" << std::endl;
    exit(EXIT_FAILURE);
  }

  delete [] argv;
}

  cuda_sift::CudaSift::~CudaSift()
{
  if (m_sift_gpu != NULL)
  {
    delete m_sift_gpu;
  }
}

  int cuda_sift::CudaSift::compute_using_host_image(
  const unsigned char * host_grayscale_image,
  const int image_width,
  const int image_height)
{
  // Compute SIFT.
  m_sift_gpu->RunSIFT(
    image_width,
    image_height,
    host_grayscale_image,
    GL_LUMINANCE,
    GL_UNSIGNED_BYTE);

  const int original_num_features = m_sift_gpu->GetFeatureNum();
  m_original_keypoints.resize(original_num_features);
  m_original_descriptors.resize(original_num_features);

  // Copy the features into this object's buffers.
  m_sift_gpu->GetFeatureVector(
    (SiftGPU::SiftKeypoint *)&m_original_keypoints[0],
    (float *)&m_original_descriptors[0]);

  features::sift_support::compute_indices_for_keypoints_sorted_by_scale(
    m_original_keypoints,
    m_sort_indices,
    m_max_num_features);

  features::sift_support::rearrange_vector_based_on_indices(
    m_original_keypoints,
    m_sorted_keypoints,
    m_sort_indices);
  features::sift_support::rearrange_vector_based_on_indices(
    m_original_descriptors,
    m_sorted_descriptors,
    m_sort_indices);

  // Return the number of features.
  return static_cast<int>(m_sort_indices.size());
}
