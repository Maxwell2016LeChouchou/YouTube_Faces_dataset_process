import tensorflow as tf 
import os
import io
import pandas as pd 
import sys

sys.path.append('/home/max/Downloads/MTCNN/models/research/')
from PIL import Image
from collections import namedtuple, OrderedDict
from object_detection.utils import dataset_util 

image_dir = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/Val_dataset/'
csv_input = '/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces/val_csv_bbox/'

def class_text_to_int(row_label):
    if row_label == 'face':
        return 1
    else: 
        None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path):
    #with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
    with tf.gfile.GFile(os.path.join(path), 'rb') as fid: 
    #with tf.io.gfile.GFile(os.path.join(path), 'rb') as fid:    
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size    
    filename = group.filename.encode('utf8')
    image_format = b'jpg'

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin']/width)
        xmaxs.append(row['xmax']/width)
        ymins.append(row['ymin']/height)
        ymaxs.append(row['ymax']/height) 
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/width': dataset_util.int64_feature(width),
        'image/height': dataset_util.int64_feature(height),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# def get_imagelist(file_path):
#     image_list = []

#     for home, dirs, files in os.walk(file_path):
#         for filename in files:
#             image_list.append(os.path.join(home,filename))

#     return image_list

# def get_imagelist(file_path):
    # image_list = []

    # for home, dirs, files in os.walk(file_path):
       # for filename in files:
           # image_list.append(os.path.join(home,filename))

    #return image_list
    # print("successfully get image list")
    # return image_list


# def get_csvlist(csv_input):
#     for txt_file in os.listdir(csv_input):
#         if os.path.isfile(os.path.join(csv_input,txt_file)):
#             txt_path = csv_input+txt_file
#             print(txt_path)
#             csv_info = pd.read_csv(txt_path)

#     return csv_info

def get_csvlist(csv_file):
    files = []
    #file_list = sorted((os.listdir(csv_file)))
    for f in sorted(os.listdir(csv_file)):
        domain = os.path.abspath(csv_file)
        f = os.path.join(domain,f)
        
        files += [f]
    print("successfully get csv list")
    return files
 
    

def main(_):

    

    csv_infos = get_csvlist(csv_input)
    #image_list = get_imagelist(image_dir)
    print('print csv info')
    

    # with open(csv_info, 'r') as f:
    #     all_lines = [l.strip() for l in f]
    # all_lines_paths = [line.split(",")[0] for line in all_lines]
    # all_lines_paths = all_lines_paths[1:]

    #final_path = [os.path.join(image_list, path)for path in all_lines_paths]
    #final_path = image_list
   
    output_path = '/home/max/Downloads/cosine_metric_learning-master/datasets/tf_record/tf_val.record'

    writer = tf.python_io.TFRecordWriter(output_path)
    #writer = tf.io.TFRecordWriter(output_path)
    
    for csv_info in csv_infos:
        examples = pd.read_csv(csv_info)
        grouped = split(examples, 'filename')
        count = 0
        for group in grouped:            
            #tf_example = create_tf_example(group, final_path[count])
            tf_example = create_tf_example(group, os.path.join(image_dir, group.filename))
            writer.write(tf_example.SerializeToString())
            count += 1
        # file_txt = os.listdir(csv_input)
        # for txt in file_txt:
        #     file_txt2 = os.path.join(csv_input, txt)
        #     print(file_txt2)
        #     examples = pd.read_csv(file_txt2)
        #     grouped = split(examples, 'filename')
        #     counter = 0
        #     for group in grouped:
        #         tf_example = create_tf_example(group, final_path[counter])
        #         writer.write(tf_example.SerializeToString())
        #         counter += 1


    writer.close()
    print('Successfully created the TFRecords: {}'.format(output_path))       

if __name__ == '__main__':
    tf.app.run()
    #tf.compat.v1.app.run()

