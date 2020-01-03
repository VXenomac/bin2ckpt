import tensorflow as tf
import torch


def convert(bin_path, ckpt_path):
    """
    用于转换 PyTorch bin 模型至 TensorFlow ckpt 模型
    :param bin_path: bin 模型路径
    :param ckpt_path: ckpt 模型路径
    """
    with tf.Session() as sess:
        for var_name, value in torch.load(bin_path, map_location='cpu').items():
            print('%s mapping...' % var_name)
            tf.Variable(initial_value=value, name=var_name)
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckpt_path)


if __name__ == '__main__':
    bin_path = '2nd_General_TinyBERT_6L_768D/pytorch_model.bin'
    ckpt_path = '2nd_General_TinyBERT_6L_768D/bert_model.ckpt'
    convert(bin_path, ckpt_path)
