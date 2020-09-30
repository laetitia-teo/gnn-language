import os
import sys
import numpy
import logging
import collections
import time

from .. import utils

def timer(func):
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        return run_time, value
    return wrapper_timer

def flatten_dict(d, parent_key='', sep='_'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def cumulate_value(buffer_dict, new_dict):
    for key, value in flatten_dict(new_dict).items():
        if key not in buffer_dict.keys():
            buffer_dict[key] = value
        else:
            buffer_dict[key] += value
    return buffer_dict

def get_log_dir(log_name):
    return os.path.join(utils.storage_dir(), "logs", log_name)


def get_log_path(log_name):
    return os.path.join(get_log_dir(log_name), "log.log")


def synthesize(array):
    import collections
    d = collections.OrderedDict()
    d["mean"] = numpy.mean(array)
    d["std"] = numpy.std(array)
    d["min"] = numpy.amin(array)
    d["max"] = numpy.amax(array)
    return d


def configure_logging(log_name):
    path = get_log_path(log_name)
    utils.create_folders_if_necessary(path)

    logging.basicConfig(
        level=logging.INFO,
        format="%(name)s: %(asctime)s: %(message)s",
        handlers=[
            logging.FileHandler(filename=path),
            logging.StreamHandler(sys.stdout)
        ]
    )


