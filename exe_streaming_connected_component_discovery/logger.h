#pragma once
#ifndef LOGGER_H
#define LOGGER_H

#include <boost/log/core.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/logger.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/utility/setup/console.hpp>
#include <string>

BOOST_LOG_INLINE_GLOBAL_LOGGER_DEFAULT(my_logger, boost::log::sources::logger_mt);

#define GET_LOGGER() boost::log::sources::logger_mt & logger = my_logger::get()

#define LOGGER BOOST_LOG(logger)

void init_logger(const std::string & log_folder);

#endif // LOGGER_H
