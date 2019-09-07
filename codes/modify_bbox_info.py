import os
import numpy as np

def modify_bbox_info(input_dir, output_dir):
    array_dic = []
    for line in open(input_dir, "r"):
        data = line.split(",")
        filename = data[0]
        face_class = data[1]
        total_len = len(data)
        features = [float(i) for i in data[2:total_len]]
        x = features[0]
        y = features[1]
        width = features[2]
        height = features[3]
        x_min = x - width/2
        x_max = x + width/2
        y_min = y - height/2
        y_max = y + height/2
        array_dic.append(np.array([filename, face_class, x_min, x_max, y_min, y_max]))
    a = np.array(array_dic)
    np.savetxt(output_dir, a, fmt="%s,%s,%s,%s,%s,%s")
    print(output_dir)



def main():
    input_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_filename_bbox/'
    output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_csv_bbox/'
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for txtfile in os.listdir(input_path):
        if os.path.isfile(os.path.join(input_path,txtfile)):
            modify_bbox_info(input_path+txtfile, output_path+txtfile)
            print(txtfile)

if __name__ == "__main__":
    main()
