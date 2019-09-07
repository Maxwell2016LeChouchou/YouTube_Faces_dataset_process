import tensorflow as tf 
import numpy as np 
import os
from PIL import Image

slim = tf.contrib.slim

#create TFrecord file

def create_record_file():
    train_filename = "train.tfrecords"
    if os.path.exists(train_filename):
        os.remove(train_filename)

    #create .tfrecord file and write it in
    writer = tf.python_io.TFRecordWriter('/home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces'+train_filename)
    with tf.Session() as sess:
        for i in range(10):
            raw_image = tf.gfile.FastGFile(" /home/max/Downloads/cosine_metric_learning-master/datasets/Youtube_faces"+str(i)+".jpg", 'rb').read()
            decode_data = tf.image.decode_jpeg(raw_image)
            image_shape = decode_data.eval().shape 
            example  =tf.train.Example(feature=tf.train.Feature(
                feature={
                    'encoded':tf.train.Feature(bytes_list = tf.train.BytesList(value = [raw_image])),
                    'format':tf.train.Feature(bytes_list = tf.train.BytesList(value=[b'jpg'])),
                    'width':tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[1]])),
                    'height':tf.train.Feature(int64_list = tf.train.Int64List(value = [image_shape[0]])),
                    'label':tf.train.Feature(int64_list = tf.train.Int64List(value=[i])),
                }))
            writer.write(example.SerializeToString())
        writer.close()
        print("Successfully saved tfrecord files")

#Use slim API to read tfrecord files
def read_record_file():
    tfrecords_filename = "train.tfrecords"
    #tf.train.Example to inverse to store 
    keys_to_features = {
        'encoded': tf.FixedLenFeature((), tf.string, default_value=''),
        'format': tf.FixedLenFeature((), tf.string, default_value='jpeg'),
        'width': tf.FixedLenFeature((),tf.int64),
        'height': tf.FixedLenFeature((), tf.int64),
        'label': tf.FixedLenFeature((), tf.int64),
    }
    #use slim to decode it (inverse)
    items_to_handlers = {
        'image': tf.tfexample_decoder.Image(image_key='encoded',
                                              format_key='format',
                                              channels=3),
        'label': slim.tfexample_decoder.Tensor('label'),
        'height': slim.tfexample_decoder.Tensor('height'),
        'width': slim.tfexample_decoder.Tensor('width'),
    }
    #define the decoder
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)
    #define dataset
    dataset = slim.dataset.Dataset(
        data_sources=tfrecords_filename,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=300000,      #I dont know how many images in the Youtube Faces 
        items_to_descriptions=None,
        num_classes=1,
        )
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=1
        common_queue_capacity=20,
        common_queue_min=1
    )
    [image, label, height, width] = provider.get(['image', 'label', 'height', 'width'])
    with tf.Session() as sess:
        initializer = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator()
        threads=tf.train.start_queue_runners(coord=coord)
        for i in range(10):
            img, l, h, w = sess.run([image, label, height, width])
            img = tf.reshape(img, [h,w,3])
            print(img.shape)
            img=Image.fromarray(img.eval(), 'RGB')
            img.save('./'+str(l)+'.jpg')
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    create_record_file()
    read_record_file()




