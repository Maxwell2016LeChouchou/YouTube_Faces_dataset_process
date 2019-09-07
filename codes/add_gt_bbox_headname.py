import os 
import csv
import numpy as np 


def add_line_bbox(path):
    toAdd = ["filename", "class", "xmin", "xmax", "ymin", "ymax"]
    with open(path, "r") as infile:
        reader = list(csv.reader(infile))
        reader.insert(0, toAdd)

    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        for line in reader:
            writer.writerow(line)


def main():
    input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_csv_bbox/'
    # (test) = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_file_bbox_correct_slash_path/'
    # (val) = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_file_bbox_correct_slash_path/'
    for txtfile in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,txtfile)):
            add_line_bbox(input_path+txtfile)
    print('Successfully added ground truth bbox headname')


if __name__ == "__main__":
    main()        