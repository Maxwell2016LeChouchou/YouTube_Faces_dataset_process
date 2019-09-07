import os 
import numpy as np 

input_filename = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_filename_bbox/Aaron_Eckhart.labeled_faces.txt'

output_filename = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/output.txt'

array_dic = []
for line in open(input_filename, "r"):
    data = line.split(",")
    filename = data[0]
    new_filename = filename.replace('\\', '/')
    total_len = len(data)
    features = [float(i) for i in data[1:total_len]]
    print(features)
    x = features[0]
    y = features[1]
    width = features[2]
    height = features[3]
    array_dic.append(np.array([new_filename, x, y, width, height]))
a = np.array(array_dic)
np.savetxt(output_filename, a,fmt="%s,%s,%s,%s,%s")

print(output_filename)
