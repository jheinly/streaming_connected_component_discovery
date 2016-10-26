#pragma once
#ifndef FEATURE_DATABASE_WRAPPER_H
#define FEATURE_DATABASE_WRAPPER_H

#include <base2d/feature_database.h>
#include <boost/thread/mutex.hpp>

struct FeatureDatabaseWrapper
{
  colmap::FeatureDatabase database;
  boost::mutex mutex;
};

#endif // FEATURE_DATABASE_WRAPPER_H
