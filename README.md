# cosine_metric_learning_face_tracking

The data_process file consists of to convert YoutubeFace datasets which can be downloaded from https://www.cs.tau.ac.il/~wolf/ytfaces/ to corresponding datasets. YoutubeFace datasets layout is that one parent folder represents one Youtube Celebrity, one parent folder has a few sub-folders which are a few of different videos having been extracted to image frames.

We first split image datasets to 8:1:1 by the numbers of parent folders: training dataset consists of 1277 celebrities(parent folders), validation dataset consists of 159 celebrities (parent folders), and test dataset consists of 159 celebrities (parent folders).  Same as txt file which is same as csv file and split correpsonding to image datasets.

First, we have to change the directory format of the csv file by running the change_slash.py. 

Second, we have to add label which is "face" in the csv file and add the bbox info by running the finalize_csv_file.py

Third, since later on we have to use tensorflow object_detection API for running the model which only accept x_min, x_max, y_min, y_max(bbox left up corner and bbox right down corner), while the YoutubeFace bbox has x, y, bbox_width, bbox_height(x, y are the center of the bbox). Therefor we have to modify bbox information by running modify_bbox_info.py

Next step is to add the correct headname for bbox information by running the add_gt_bbox_headname.py

Last step is to convert the image along with csv file to tfrecord to let tensorflow read and train the images with groudtruth bboxing by running the train_image_csv_tfrecord.py. Note that the modify the path to recall the functions in the dataset_utils.py for the utility functions for creating TFRecord data sets
