#ifndef FEATURES_H
#define FEATURES_H

#include <cstdlib>
#include <cstring>

namespace V3D
{
  class Features
  {
    public:
      enum FeatureType {
        NoFeatureType,
        SiftFeatureType,
        HarrisBriefFeatureType,
        FreakFeatureType,
        DummyFeatureType};

      struct GenericKeypoint
      {
        float x;
        float y;
      };

      // Create an empty Features object. To allocate data, call init().
      Features()
      : _keypointData(0), _descriptorData(0),
        _size(0), _allocatedSize(0),
        _keypointSize(0), _descriptorSize(0),
        _keypointDataSize(0), _descriptorDataSize(0),
        _featureType(NoFeatureType)
      {}

      // This type of constructor cannot be used as it generates the following error:
      //
      //   template parameter "Feature" is not used in declaring the parameter types
      //   of function template "V3D::Features::Features<Feature>(unsigned int)"
      //
      //template<typename Feature>
      //Features(unsigned int numFeatures)
      //{...}

      Features(const Features& features)
      : _keypointData(0), _descriptorData(0),
        _size(0), _allocatedSize(0),
        _keypointSize(0), _descriptorSize(0),
        _keypointDataSize(0), _descriptorDataSize(0),
        _featureType(NoFeatureType)
      {
        _featureType = features._featureType;

        if (features._keypointData)
        {
          // allocateData sets all member variables except _featureType
          allocateData(features._size, features._keypointSize, features._descriptorSize);
          memcpy(_keypointData,   features._keypointData,   _size * _keypointSize);
          memcpy(_descriptorData, features._descriptorData, _size * _descriptorSize);
        }
        else
        {
          _size           = features._size;
          _allocatedSize  = features._allocatedSize;
          _keypointSize   = features._keypointSize;
          _descriptorSize = features._descriptorSize;
          _keypointData = 0;
          _descriptorData = 0;
          _keypointDataSize = 0;
          _descriptorDataSize = 0;
        }
      }

      Features& operator=(const Features& features)
      {
        _featureType = features._featureType;

        if (features._keypointData)
        {
          allocateData(features._size, features._keypointSize, features._descriptorSize);
          memcpy(_keypointData,   features._keypointData,   _size * _keypointSize);
          memcpy(_descriptorData, features._descriptorData, _size * _descriptorSize);
        }
        else
        {
          _size           = features._size;
          _allocatedSize  = features._allocatedSize;
          _keypointSize   = features._keypointSize;
          _descriptorSize = features._descriptorSize;
          // leave existing data intact
        }

        return *this;
      }

      ~Features()
      {
        clearSizeTypeAndMemory();
      }

      template<typename Feature>
      void init(int numFeatures)
      {
        resize<Feature>(numFeatures);
      }

      // Empties the Features object, but leaves the allocated memory intact.
      void clearSizeAndType()
      {
        _size           = 0;
        _allocatedSize  = 0;
        _featureType    = NoFeatureType;
        _keypointSize   = 0;
        _descriptorSize = 0;
      }

      // Empties the Features object, and frees all allocated memory.
      void clearSizeTypeAndMemory()
      {
        freeData();
        clearSizeAndType();
      }

      bool isEmpty() const
      {
        return _size == 0;
      }

      // Resize the Features object to hold the new number of features.
      // If there is already enough existing memory, don't reallocate
      // and simply use it.
      template<typename Feature>
      void resize(int numFeatures)
      {
        allocateData(numFeatures,
                     sizeof(typename Feature::Keypoint),
                     sizeof(typename Feature::Descriptor));
        _featureType = Feature::Type;
      }

      // Resize the Features object to hold the new number of features.
      // Reallocate any existing memory.
      template<typename Feature>
      void tightResize(int numFeatures)
      {
        // TODO - don't free the data if the new number of features is the
        // exact size of the existing allocated memory.
        freeData();
        allocateData(numFeatures,
                     sizeof(typename Feature::Keypoint),
                     sizeof(typename Feature::Descriptor));
        _featureType = Feature::Type;
      }

      const Features::GenericKeypoint& genericKeypoint(int index) const
      {
        return *(Features::GenericKeypoint*)((char*)_keypointData + index * _keypointSize);
      }

      Features::GenericKeypoint& genericKeypoint(int index)
      {
        return *(Features::GenericKeypoint*)((char*)_keypointData + index * _keypointSize);
      }

      template<typename Feature>
      const typename Feature::Keypoint& keypoint(int index) const
      {
        return ((typename Feature::Keypoint*)_keypointData)[index];
      }

      template<typename Feature>
      typename Feature::Keypoint& keypoint(int index)
      {
        return ((typename Feature::Keypoint*)_keypointData)[index];
      }

      template<typename Feature>
      const typename Feature::Descriptor& descriptor(int index) const
      {
        return ((typename Feature::Descriptor*)_descriptorData)[index];
      }

      template<typename Feature>
      typename Feature::Descriptor& descriptor(int index)
      {
        return ((typename Feature::Descriptor*)_descriptorData)[index];
      }
      
      template<typename Feature>
      typename Feature::Keypoint* keypoints()
      {
        return (typename Feature::Keypoint*)_keypointData;
      }

      template<typename Feature>
      typename Feature::Descriptor* descriptors()
      {
        return (typename Feature::Descriptor*)_descriptorData;
      }

      const void* rawKeypoints() const
      {
        return _keypointData;
      }

      void* rawKeypoints()
      {
        return _keypointData;
      }

      const void* rawDescriptors() const
      {
        return _descriptorData;
      }

      void* rawDescriptors()
      {
        return _descriptorData;
      }

      int size() const
      {
        return _size;
      }

      void setSize(int size)
      {
        _size = size;
      }

      int allocatedSize() const
      {
        return _allocatedSize;
      }

      FeatureType getType() const
      {
        return _featureType;
      }

    private:
      void allocateData(int numFeatures,
                        int keypointSize,
                        int descriptorSize)
      {
        // Check if data has already been allocated
        if (_keypointData)
        {
          int newKeypointDataSize = numFeatures * keypointSize;
          if (newKeypointDataSize > _keypointDataSize)
          {
            _keypointDataSize = newKeypointDataSize;
            _keypointData = realloc(_keypointData, newKeypointDataSize);
          }

          int newDescriptorDataSize = numFeatures * descriptorSize;
          if (newDescriptorDataSize > _descriptorDataSize)
          {
            _descriptorDataSize = newDescriptorDataSize;
            _descriptorData = realloc(_descriptorData, newDescriptorDataSize);
          }
        }
        // If data has not already been allocated
        else
        {
          _keypointDataSize   = numFeatures * keypointSize;
          _descriptorDataSize = numFeatures * descriptorSize;
          _keypointData       = malloc(_keypointDataSize);
          _descriptorData     = malloc(_descriptorDataSize);
        }

        // NOTE - allocatedSize may not be entirely accurate as there may be more space
        //        present that could hold additional features (if data had already been
        //        allocated). However, this is the simplest approach.
        _size           = numFeatures;
        _allocatedSize  = numFeatures;
        _keypointSize   = keypointSize;
        _descriptorSize = descriptorSize;
      }

      void freeData()
      {
        if (_keypointData)
        {
          free(_keypointData);
          free(_descriptorData);
        }

        _keypointDataSize   = 0;
        _descriptorDataSize = 0;
        _keypointData       = 0;
        _descriptorData     = 0;
      }

      void * _keypointData;     // pointer to keypoint data
      void * _descriptorData;   // pointer to descriptor data
      int _size;                // number of features
      int _allocatedSize;       // number of allocated features
      int _keypointSize;        // size of keypoint in bytes
      int _descriptorSize;      // size of descriptor in bytes
      int _keypointDataSize;    // size of allocated keypoint data in bytes
      int _descriptorDataSize;  // size of allocated descriptor data in bytes
      FeatureType _featureType; // the type of the currently loaded feature
  }; // class Features
} // namespace V3D

#endif // FEATURES_H
