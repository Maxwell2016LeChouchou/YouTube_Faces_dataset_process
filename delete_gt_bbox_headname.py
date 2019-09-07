import os 
import csv
import numpy as np 


def add_line_bbox(path):
    with open(path, "r") as infile:
        data_in = infile.readlines()
    with open(path, "w") as outfile:
        outfile.writelines(data_in[1:])
    
    # toAdd = ["filename", "x", "y", "bbox_width", "bbox_height"]
    # with open(path, "r") as infile:
    #     reader = list(csv.reader(infile))
    #     reader.insert(0, toAdd)

    # with open(path, "w") as outfile:
    #     writer = csv.writer(outfile)
    #     for line in reader:
    #         writer.writerow(line)

def main():
    input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_filename_bbox/'
    #input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_filename_bbox/'
    #input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_filename_bbox/'
    for txtfile in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,txtfile)):
            add_line_bbox(input_path+txtfile)
    print('Successfully changed slash for dir path')


if __name__ == "__main__":
    main()        