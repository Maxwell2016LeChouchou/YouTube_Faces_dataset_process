import os 
import numpy as np 

array_dic = []
def generate_bbox(input_dir, output_filename):
    for line in open(input_dir, "r"):
        data = line.split(",")
        filename = data[0]
        filename = bytes(filename, 'utf-8')
        total_len = len(data)
        features = [float(i) for i in data[1:total_len]]
        x = features[1]
        y = features[2]
        width = features[3]
        height = features[4]
        array_dic.append(np.array([x, y, width, height]))
    a = np.array(array_dic)
    if a.shape[0] > 0:
        np.savetxt(output_filename,a,fmt="%.1f,%.1f,%.1f,%.1f")
    print(output_filename)
    # print(a.shape, input_dir)   
    # np.savetxt(output_filename,a)

def main():
    current_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_info_csv/"
    image_bbox_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/train_bbox/"

    #current_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_info_csv/"
    #image_bbox_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/test_bbox/"

    #current_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_info_csv/"
    #image_bbox_path = "/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_bbox/"

    if not os.path.exists(image_bbox_path):
        os.mkdir(image_bbox_path)
    for filename in os.listdir(current_path):
        if os.path.isfile(os.path.join(current_path, filename)):
            generate_bbox(current_path+filename, image_bbox_path+filename)

if __name__ == "__main__":
    main()



    

    