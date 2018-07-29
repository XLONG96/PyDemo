import tensorflow as tf
import os
import random
import tensorflow.contrib.slim as slim
import time
import logging
import numpy as np
import pickle
from tensorflow.python.ops import control_flow_ops
from PIL import Image
from tensorflow.python.framework import graph_util

logger = logging.getLogger('Training a chinese write char recognition')
logger.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


tf.app.flags.DEFINE_boolean('random_flip_up_down', False, "Whether to random flip up down")
tf.app.flags.DEFINE_boolean('random_brightness', True, "whether to adjust brightness")
tf.app.flags.DEFINE_boolean('random_contrast', True, "whether to random constrast")

tf.app.flags.DEFINE_integer('charset_size', 3818, "Choose the first `charset_size` character to conduct our experiment.")
tf.app.flags.DEFINE_integer('image_size', 56, "Needs to provide same value as in training.")
tf.app.flags.DEFINE_boolean('gray', True, "whether to change the rbg to gray")
tf.app.flags.DEFINE_integer('max_steps', 100000, 'the max training steps ')
tf.app.flags.DEFINE_integer('eval_steps', 50, "the step num to eval")
tf.app.flags.DEFINE_integer('save_steps', 10, "the steps to save")

tf.app.flags.DEFINE_string('checkpoint_dir', '.\\checkpoint\\', 'the checkpoint dir')
tf.app.flags.DEFINE_string('train_data_dir', '.\\dataset\\train\\', 'the train dataset dir')
tf.app.flags.DEFINE_string('test_data_dir', '.\\dataset\\test\\', 'the test dataset dir')
tf.app.flags.DEFINE_string('log_dir', '.\\log', 'the logging dir')

tf.app.flags.DEFINE_boolean('restore', True, 'whether to restore from checkpoint')
tf.app.flags.DEFINE_boolean('epoch', 1, 'Number of epoches')
tf.app.flags.DEFINE_boolean('batch_size', 100, 'Validation batch size')
tf.app.flags.DEFINE_string('mode', 'inferenceFreezing', 'Running mode. One of {"train", "valid", "test"}')
FLAGS = tf.app.flags.FLAGS

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

class DataIterator:
    def __init__(self, data_dir):
        # Set FLAGS.charset_size to a small value if available computation power is limited.
        truncate_path = data_dir + ('%05d' % FLAGS.charset_size)
        print(truncate_path)
        self.image_names = []
        for root, sub_folder, file_list in os.walk(data_dir):
            if root < truncate_path:
                self.image_names += [os.path.join(root, file_path) for file_path in file_list]
        random.shuffle(self.image_names)
        self.labels = [int(file_name[len(data_dir):].split(os.sep)[0]) for file_name in self.image_names]

    @property
    def size(self):
        return len(self.labels)

    @staticmethod
    def data_augmentation(images):
        if FLAGS.random_flip_up_down:
            images = tf.image.random_flip_up_down(images)
        if FLAGS.random_brightness:
            images = tf.image.random_brightness(images, max_delta=0.3)
        if FLAGS.random_contrast:
            images = tf.image.random_contrast(images, 0.8, 1.2)
        return images

    def input_pipeline(self, batch_size, num_epochs=None, aug=False):
        images_tensor = tf.convert_to_tensor(self.image_names, dtype=tf.string)
        labels_tensor = tf.convert_to_tensor(self.labels, dtype=tf.int64)
        input_queue = tf.train.slice_input_producer([images_tensor, labels_tensor], num_epochs=num_epochs)

        labels = input_queue[1]
        images_content = tf.read_file(input_queue[0])
        images = tf.image.convert_image_dtype(tf.image.decode_png(images_content, channels=1), tf.float32)
        if aug:
            images = self.data_augmentation(images)
        new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
        images = tf.image.resize_images(images, new_size)
        image_batch, label_batch = tf.train.shuffle_batch([images, labels], batch_size=batch_size, capacity=50000,
                                                          min_after_dequeue=10000)
        return image_batch, label_batch


def build_graph():
    # with tf.device('/cpu:0'):
    top_K_64 = tf.placeholder(dtype=tf.int64, shape=[], name='top_k_64')
    keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='keep_prob')
    images = tf.placeholder(dtype=tf.float32, shape=[None, FLAGS.image_size, FLAGS.image_size, 1], name='image_batch')
    labels = tf.placeholder(dtype=tf.int64, shape=[None], name='label_batch')
    is_training = tf.placeholder(dtype=tf.bool, shape=[], name='train_flag')
    with slim.arg_scope([slim.conv2d, slim.fully_connected],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params={'is_training': is_training,'decay': 0.95}):
        conv3_1 = slim.conv2d(images, 64, [3, 3], 1, padding='SAME', scope='conv3_1')
        max_pool_1 = slim.max_pool2d(conv3_1, [2, 2], [2, 2], padding='SAME', scope='pool1')
        conv3_2 = slim.conv2d(max_pool_1, 128, [3, 3], padding='SAME', scope='conv3_2')
        max_pool_2 = slim.max_pool2d(conv3_2, [2, 2], [2, 2], padding='SAME', scope='pool2')
        conv3_3 = slim.conv2d(max_pool_2, 256, [3, 3], padding='SAME', scope='conv3_3')
        max_pool_3 = slim.max_pool2d(conv3_3, [2, 2], [2, 2], padding='SAME', scope='pool3')
        conv3_4 = slim.conv2d(max_pool_3, 512, [3, 3], padding='SAME', scope='conv3_4')
        conv3_5 = slim.conv2d(conv3_4, 512, [3, 3], padding='SAME', scope='conv3_5')
        max_pool_4 = slim.max_pool2d(conv3_5, [2, 2], [2, 2], padding='SAME', scope='pool4')
        d4 = slim.dropout(max_pool_4, keep_prob)

        flatten = slim.flatten(d4)
        fc1 = slim.fully_connected(flatten, 3072, activation_fn=tf.nn.relu, scope='fc1')
        logits = slim.fully_connected(slim.dropout(fc1, keep_prob), FLAGS.charset_size, activation_fn = None, scope='fc2')
        # logits = slim.fully_connected(flatten, FLAGS.charset_size, activation_fn=None, reuse=reuse, scope='fc')
    
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), labels), tf.float32))

    global_step = tf.get_variable("step", [], initializer=tf.constant_initializer(0.0), trainable=False)
    rate = tf.train.exponential_decay(0.1, global_step, decay_steps=2000, decay_rate=0.96, staircase=True)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies([tf.group(*update_ops)]):
        train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(loss, global_step=global_step)
    probabilities = tf.nn.softmax(logits, name="softmax")

    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    merged_summary_op = tf.summary.merge_all()
    top_k = tf.cast(top_K_64, tf.int32, name="cast32")
    predicted_val_top_k, predicted_index_top_k = tf.nn.top_k(probabilities, k=top_k, name="topK_out")
    accuracy_in_top_k = tf.reduce_mean(tf.cast(tf.nn.in_top_k(probabilities, labels, top_K_64), tf.float32))

    return {'images': images,
            'labels': labels,
            'keep_prob': keep_prob,
            'top_k': top_K_64,
            'global_step': global_step,
            'train_op': train_op,
            'loss': loss,
            'is_training': is_training,
            'accuracy': accuracy,
            'accuracy_top_k': accuracy_in_top_k,
            'merged_summary_op': merged_summary_op,
            'predicted_distribution': probabilities,
            'predicted_index_top_k': predicted_index_top_k,
            'predicted_val_top_k': predicted_val_top_k}


def train():
    print('Begin training')
    train_feeder = DataIterator(data_dir='D:\\Code\\Python\\CPS-OCR-Engine-master\\CPS-OCR-Engine-master\\ocr\\data1\\train\\')
    test_feeder = DataIterator(data_dir='D:\\Code\\Python\\CPS-OCR-Engine-master\\CPS-OCR-Engine-master\\ocr\\data1\\test\\')
    with tf.Session(config = config) as sess:
        train_images, train_labels = train_feeder.input_pipeline(batch_size=FLAGS.batch_size, aug=True)
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size)
        graph = build_graph()
        sess.run(tf.global_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        saver = tf.train.Saver()

        train_writer = tf.summary.FileWriter(FLAGS.log_dir + '\\train', sess.graph)
        test_writer = tf.summary.FileWriter(FLAGS.log_dir + '\\val')
        start_step = 0
        if FLAGS.restore:
            ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            if ckpt:
                saver.restore(sess, ckpt)
                print("restore from the checkpoint {0}".format(ckpt))
                start_step += int(ckpt.split('-')[-1])

        logger.info(':::Training Start:::')
        try:
            while not coord.should_stop():
                start_time = time.time()
                train_images_batch, train_labels_batch = sess.run([train_images, train_labels])
                feed_dict = {
                             graph['top_k']: 1,
                             graph['images']: train_images_batch,
                             graph['labels']: train_labels_batch,
                             graph['keep_prob']: 0.5,
                             graph['is_training']: True}
                _, loss_val, train_summary, step = sess.run(
                    [graph['train_op'], graph['loss'], graph['merged_summary_op'], graph['global_step']],
                    feed_dict=feed_dict)
                train_writer.add_summary(train_summary, step)
                end_time = time.time()
                logger.info("the step {0} takes {1} loss {2}".format(step, end_time - start_time, loss_val))
                if step > FLAGS.max_steps:
                    break
                if step % FLAGS.eval_steps == 1:
                    test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                    feed_dict = {graph['top_k']: 1,
                                 graph['images']: test_images_batch,
                                 graph['labels']: test_labels_batch,
                                 graph['keep_prob']: 1.0,
                                 graph['is_training']: False}
                    accuracy_test, test_summary = sess.run(
                        [graph['accuracy'], graph['merged_summary_op']],
                        feed_dict=feed_dict)
                    test_writer.add_summary(test_summary, step)
                    logger.info('===============Eval a batch=======================')
                    logger.info('the step {0} test accuracy: {1}'
                                .format(step, accuracy_test))
                    logger.info('===============Eval a batch=======================')
                if step % FLAGS.save_steps == 1:
                    logger.info('Save the ckpt of {0}'.format(step))
                    saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'),
                               global_step=graph['global_step'])
        except tf.errors.OutOfRangeError:
            logger.info('==================Train Finished================')
            saver.save(sess, os.path.join(FLAGS.checkpoint_dir, 'my-model'), global_step=graph['global_step'])
        finally:
            coord.request_stop()
        coord.join(threads)


def validation():
    print('validation')
    test_feeder = DataIterator(data_dir='D:\\Code\\Python\\CPS-OCR-Engine-master\\CPS-OCR-Engine-master\\ocr\\data1\\test\\')

    final_predict_val = []
    final_predict_index = []
    groundtruth = []

    with tf.Session() as sess:
        test_images, test_labels = test_feeder.input_pipeline(batch_size=FLAGS.batch_size, num_epochs=1)
        graph = build_graph()

        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())  # initialize test_feeder's inside state

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
            print("restore from the checkpoint {0}".format(ckpt))

        logger.info(':::Start validation:::')
        try:
            i = 0
            acc_top_1, acc_top_k = 0.0, 0.0
            while not coord.should_stop():
                i += 1
                start_time = time.time()
                test_images_batch, test_labels_batch = sess.run([test_images, test_labels])
                feed_dict = {graph['top_k']: 3,
                            graph['images']: test_images_batch,
                             graph['labels']: test_labels_batch,
                             graph['keep_prob']: 1.0,
                             graph['is_training']: False}
                batch_labels, probs, indices, acc_1, acc_k = sess.run([graph['labels'],
                                                                       graph['predicted_val_top_k'],
                                                                       graph['predicted_index_top_k'],
                                                                       graph['accuracy'],
                                                                       graph['accuracy_top_k']], feed_dict=feed_dict)
                final_predict_val += probs.tolist()
                final_predict_index += indices.tolist()
                groundtruth += batch_labels.tolist()
                acc_top_1 += acc_1
                acc_top_k += acc_k
                end_time = time.time()
                logger.info("the batch {0} takes {1} seconds, accuracy = {2}(top_1) {3}(top_k)"
                            .format(i, end_time - start_time, acc_1, acc_k))

        except tf.errors.OutOfRangeError:
            logger.info('==================Validation Finished================')
            if test_feeder.size != 0:
                acc_top_1 = acc_top_1 * FLAGS.batch_size / test_feeder.size
                acc_top_k = acc_top_k * FLAGS.batch_size / test_feeder.size
                logger.info('top 1 accuracy {0} top k accuracy {1}'.format(acc_top_1, acc_top_k))
        finally:
            coord.request_stop()
        coord.join(threads)
    return {'prob': final_predict_val, 'indices': final_predict_index, 'groundtruth': groundtruth}


# 获待预测图像文件夹内的图像名字
def get_file_list(path):
    list_name=[]
    files = os.listdir(path)
    files.sort()
    for file in files:
        file_path = os.path.join(path, file)
        list_name.append(file_path)
    return list_name

# 图像二值化，需注意待预测的汉字是黑底白字还是白底黑字
def binary_pic(name_list):
    for image in name_list:
        temp_image = cv2.imread(image)
        #print image
        GrayImage=cv2.cvtColor(temp_image,cv2.COLOR_BGR2GRAY) 
        ret,thresh1=cv2.threshold(GrayImage,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        single_name = image.split('t/')[1]
        print(single_name)
        cv2.imwrite('..\\data\\tmp\\'+single_name,thresh1)

# 获取汉字label映射表
def get_label_dict():
    f=open('.\\chinese_labels','rb')
    label_dict = pickle.load(f)
    f.close()
    return label_dict

def inference(name_list):
    image_set=[]
    # 对每张图进行尺寸标准化和归一化
    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
        image_set.append(temp_image)

    with tf.Session() as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        graph = build_graph(top_k=3)
        saver = tf.train.Saver()
        ckpt = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
        if ckpt:
            saver.restore(sess, ckpt)
        val_list=[]
        idx_list=[]
        # 预测每一张图
        for item in image_set:
            temp_image = item
            predict_val, predict_index = sess.run([graph['predicted_val_top_k'], graph['predicted_index_top_k']],
                                              feed_dict={graph['images']: temp_image,
                                                         graph['keep_prob']: 1.0,
                                                         graph['is_training']: False})
            val_list.append(predict_val)
            idx_list.append(predict_index)
        tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'train.pbtxt')
    #return predict_val, predict_index
    return val_list,idx_list


def freeze_graph(model_folder):
    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_folder)
    input_checkpoint = checkpoint.model_checkpoint_path
    print(input_checkpoint)
    # We precise the file fullname of our freezed graph
    absolute_model_folder = "/".join(input_checkpoint.split('/')[:-1])
    output_graph = absolute_model_folder + "/frozen_model.pb"
    print(output_graph)
    # Before exporting our graph, we need to precise what is our output node
    # this variables is plural, because you can have multiple output nodes
    # freeze之前必须明确哪个是输出结点,也就是我们要得到推论结果的结点
    # 输出结点可以看我们模型的定义
    # 只有定义了输出结点,freeze才会把得到输出结点所必要的结点都保存下来,或者哪些结点可以丢弃
    # 所以,output_node_names必须根据不同的网络进行修改
    output_node_names = "softmax,topK_out"

    # We clear the devices, to allow TensorFlow to control on the loading where it wants operations to be calculated
    clear_devices = True

    # We import the meta graph and retrive a Saver
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We retrieve the protobuf graph definition
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()

    # We start a session and restore the graph weights
    # 这边已经将训练好的参数加载进来,也即最后保存的模型是有图,并且图里面已经有参数了,所以才叫做是frozen
    # 相当于将参数已经固化在了图当中
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)

        # We use a built-in TF helper to export variables to constant
        output_graph_def = graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")  # We split on comma for convenience
        )

        # Finally we serialize and dump the output graph to the filesystem
        with tf.gfile.GFile(output_graph, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))


def load_graph(frozen_graph_filename):
    # We parse the graph_def file
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

        # We load the graph_def in the default graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(
            graph_def,
            input_map=None,
            return_elements=None,
            name="prefix",
            op_dict=None,
            producer_op_list=None
        )
    return graph


def inferenceFreezing(name_list):
    image_set = []
    # 对每张图进行尺寸标准化和归一化
    for image in name_list:
        temp_image = Image.open(image).convert('L')
        temp_image = temp_image.resize((FLAGS.image_size, FLAGS.image_size), Image.ANTIALIAS)
        temp_image = np.asarray(temp_image) / 255.0
        temp_image = temp_image.reshape([-1, FLAGS.image_size, FLAGS.image_size, 1])
        image_set.append(temp_image)

    graph = load_graph("./checkpoint/frozen_model.pb")
    for op in graph.get_operations():
        print(op.name, op.values())
    top_k_64 = graph.get_tensor_by_name('prefix/top_k_64:0')
    keep_prob= graph.get_tensor_by_name('prefix/keep_prob:0')
    train_flag = graph.get_tensor_by_name('prefix/train_flag:0')
    image_batch = graph.get_tensor_by_name('prefix/image_batch:0')
    softmax = graph.get_tensor_by_name('prefix/softmax:0')
    index = graph.get_tensor_by_name('prefix/topK_out:0')
    val = graph.get_tensor_by_name('prefix/topK_out:1')

    with tf.Session(graph=graph) as sess:
        logger.info('========start inference============')
        # images = tf.placeholder(dtype=tf.float32, shape=[None, 64, 64, 1])
        # Pass a shadow label 0. This label will not affect the computation graph.
        val_list=[]
        idx_list=[]
        # 预测每一张图
        for item in image_set:
            temp_image = item

            softmax_out = sess.run([softmax,index,val],
                     feed_dict={
                                 top_k_64: 3,
                                 keep_prob: 1.0,
                                 train_flag: False,
                                 image_batch: temp_image})
            predict_val, predict_index = tf.nn.top_k(softmax_out[0], k=3)
            predict_val = softmax_out[1]
            predict_index = softmax_out[2]
            val_list.append(predict_val)
            idx_list.append(predict_index)

    return val_list,idx_list


def main(_):
    print(FLAGS.mode)
    if FLAGS.mode == "train":
        train()
        input("input:")
    elif FLAGS.mode == 'validation':
        dct = validation()
        result_file = 'result.dict'
        logger.info('Write result into {0}'.format(result_file))
        with open(result_file, 'wb') as f:
            pickle.dump(dct, f)
        logger.info('Write file ends')
        input('input:')
    elif FLAGS.mode == 'inference':
        label_dict = get_label_dict()
        name_list = get_file_list('D:\Code\Python\CPS-OCR-Engine-master\CPS-OCR-Engine-master\\tmp')
        #binary_pic(name_list)
        #tmp_name_list = get_file_list('../data/tmp')
        # 将待预测的图片名字列表送入predict()进行预测，得到预测的结果及其index
        final_predict_val, final_predict_index = inference(name_list)
        final_reco_text =[]  # 存储最后识别出来的文字串
        # 给出top 3预测，candidate1是概率最高的预测
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0][0]
            candidate2 = final_predict_index[i][0][1]
            candidate3 = final_predict_index[i][0][2]
            final_reco_text.append(label_dict[int(candidate1)])
            logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i], 
                label_dict[int(candidate1)],label_dict[int(candidate2)],label_dict[int(candidate3)],final_predict_index[i],final_predict_val[i]))
        print ('=====================OCR RESULT=======================\n')
        # 打印出所有识别出来的结果（取top 1）
        for i in range(len(final_reco_text)):
           print(final_reco_text[i],)
    elif FLAGS.mode == 'freezing':
        freeze_graph("./checkpoint/")
    elif FLAGS.mode == 'inferenceFreezing':
        label_dict = get_label_dict()
        name_list = get_file_list('D:\Code\Python\CPS-OCR-Engine-master\CPS-OCR-Engine-master\\tmp')
        #binary_pic(name_list)
        #tmp_name_list = get_file_list('../data/tmp')
        # 将待预测的图片名字列表送入predict()进行预测，得到预测的结果及其index
        final_predict_val, final_predict_index = inferenceFreezing(name_list)
        final_reco_text =[]  # 存储最后识别出来的文字串
        # 给出top 3预测，candidate1是概率最高的预测
        for i in range(len(final_predict_val)):
            candidate1 = final_predict_index[i][0][0]
            candidate2 = final_predict_index[i][0][1]
            candidate3 = final_predict_index[i][0][2]
            final_reco_text.append(label_dict[int(candidate1)])
            logger.info('[the result info] image: {0} predict: {1} {2} {3}; predict index {4} predict_val {5}'.format(name_list[i],
                label_dict[int(candidate1)].strip(),label_dict[int(candidate2)].strip(),label_dict[int(candidate3)].strip(),final_predict_index[i],final_predict_val[i]))
        print ('=====================OCR RESULT=======================\n')
        # 打印出所有识别出来的结果（取top 1）
        for i in range(len(final_reco_text)):
           print(final_reco_text[i],)
if __name__ == "__main__":
    tf.app.run()