# This program is attempting to change the back slash / to common slash \
# Because the orgional txt file path is writen in windows not linux system

import os 
import numpy as np 

# train_file_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_filename_bbox/'
# train_output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_file_bbox_correct_slash_path/

# test_file_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_filename_bbox/'
# test_output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_file_bbox_correct_slash_path/'

# val_file_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_filename_bbox/'
# val_output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_file_bbox_correct_slash_path/'


def change_slash(input_dir, output_dir):
    array_dic = []
    for line in open(input_dir, "r"):
        data = line.split(",")
        filename = data[0]
        new_filename = filename.replace('\\', '/')
        total_len = len(data)
        features = [float(i) for i in data[1:total_len]]
        x = features[0]
        y = features[1]
        width = features[2]
        height = features[3]
        array_dic.append(np.array([new_filename, x, y, width, height]))
    a = np.array(array_dic)
    np.savetxt(output_dir, a,fmt="%s,%s,%s,%s,%s")


def main():
    input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_filename_bbox/'
    output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_file_bbox_correct_slash_path/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for txtfile in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,txtfile)):
            change_slash(input_path+txtfile,output_path+txtfile)
    print('Successfully changed slash for dir path')


if __name__ == "__main__":
    main()