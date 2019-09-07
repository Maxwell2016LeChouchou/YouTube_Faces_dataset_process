import os
import tensorflow as tf 
import numpy as np 
from PIL import Image

slim=tf.contrib.slim

#Use slim API to read TFrecord
def read_record_file():
    tfrecords_filename = "train.tfrecords"
    keys_to_features = {
        'filename': tf.FixedLenFeature((), tf.string, default_value = 'jpg'),
        'ignore_0': tf.FixedLenFeature((), tf.float32),
        'x': tf.FixedLenFeature((), tf.float32),
        'y': tf.FixedLenFeature((),tf.float32),
        'width': tf.FixedLenFeature((), tf.float32),
        'height': tf.FixedLenFeature((), tf.float32),
        'ignore_1': tf.FixedLenFeature((), tf.float32),
        'ignore_2': tf.FixedLenFeature((), tf.float32),
    }
    items_to_handlers = {
        'filename': slim.tfexample_decoder.Image(image_key='filename',
                                                 channels = 3),
        'ignore_0': slim.tfexample_decoder.Tensor('ignore_0'),
        'x': slim.tfexample_decoder.Tesnor('x'),
        'y': slim.tfexample_decoder.Tensor('y'),
        'width': slim.tfexample_decoder.Tensor('width'),
        'height': slim.tfexample_decoder.Tensor('height'),
        'ignore_1': slim.tfexample_decoder.Tensor('ignore_1'),
        'ignore_2': slim.tfexample_decoder.Tensor('ignore_2'),
    }
    decoder = slim.tfexample_decoder.TFExampleDecoder(keys_to_features, items_to_handlers)

    dataset = slim.dataset.Dataset(
        data_sources=tfrecords_filename,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples = 10,
        items_to_descriptions=None,
        num_classes=10,
        
    )
    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset, num_readers=1, common_queue_capacity=20, common_queue_min=1
    )

    [filename,x,y,width,height] = provider.get(
    ['filename', 'x', 'y', 'width', 'height']) 
    with tf.Session as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        coord=tf.train.Coordinator() 
        threads=tf.train.start_queue_runners(coord=coord)
        try:
            while not coord.should_stop():
                print ('*********')
        for i in range(10):
            file_name, x_x, y_y, w, h=sess.run([filename, x, y, width, height])
        
        #Here we should add some code to transfer narray to image

        except tf.errors.OutOfRangeError:
            print("Done, now lets kill all the threads....")
        finally:
            coord.request_stop()
            print('All threads are asked to stop')
        coord.join(threads)
        print('all htreads are stopped')

if __name__ == '__main__':
    create_record_file()
    read_record_file()
