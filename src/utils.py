import tensorflow as tf
import argparse

def map_label(label, classes):
    mapped_label = tf.zeros(label.size(), dtype=tf.int32)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
