from __future__ import absolute_import, division, print_function

import numpy as np 
import tensorflow as tf 
import time
from scipy.misc import imread, imresize
from os import walk
from os.path import join 

data_dir = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/'

def read_images(path):
    filenames = next(walk(path))[2]
    num_files = len(filenames)
    