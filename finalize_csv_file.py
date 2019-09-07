import os 
import numpy as np 
import csv


def add_class_face(input_file, outpuit_file):
    with open(input_file, 'r') as f:
        data = csv.reader(f, delimiter=',')            

        with open(outpuit_file, 'w') as f:
            for r in data:  
                f.write('{},face,{},{},{},{}\n'.format(*r))
            
def add_line_bbox(path):
    toAdd = ["filename", "class", "x", "y", "bbox_width", "bbox_height"]
    with open(path, "r") as infile:
        reader = list(csv.reader(infile))
        reader.insert(0,toAdd)
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        for line in reader: 
            writer.writerow(line)

def main():
    input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_file_bbox_correct_slash_path/'
    output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_filename_bbox/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for txtfile in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,txtfile)):
            add_class_face(input_path+txtfile,output_path+txtfile)
    print('Successfully added class of face for dir path')

    for txtfile in os.listdir(output_path):
        if os.path.isfile(os.path.join(output_path,txtfile)):
            add_line_bbox(output_path+txtfile)
    print("Successfully add line bbox for dir path")


if __name__ == "__main__":
    main()