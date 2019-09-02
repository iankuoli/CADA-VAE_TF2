import numpy as np
import scipy.io as sio
from sklearn import preprocessing
import sys
import os
from pathlib import Path
import pickle
import copy
import tensorflow as tf

from src.utils import map_label


class DataLoader(object):
    def __init__(self, dataset, aux_data_source):

        print("The current working directory is %s" % os.getcwd())
        
        folder = str(Path(os.getcwd()))
        project_directory = str(Path(os.getcwd()).parent) if folder[-5:] == 'model' else folder
        print('Project Directory: %s' % project_directory)
        data_path = project_directory + '/data'
        print('Data Path: %s' % data_path)
        sys.path.append(data_path)

        self.data_path = data_path
        self.dataset = dataset
        self.aux_data_src = aux_data_source
        self.aux_data = None
        self.all_data_sources = ['resnet_features'] + [self.aux_data_src]

        if self.dataset == 'CUB':
            self.data_dir = self.data_path + '/CUB/'
        elif self.dataset == 'SUN':
            self.data_dir = self.data_path + '/SUN/'
        elif self.dataset == 'AWA1':
            self.data_dir = self.data_path + '/AWA1/'
        elif self.dataset == 'AWA2':
            self.data_dir = self.data_path + '/AWA2/'

        self.seen_classes = None
        self.novel_classes = None
        self.num_train = None
        self.num_train_class = None
        self.num_test_class = None
        self.train_class = None
        self.all_classes = None
        self.train_mapped_label = None
        self.data = {}
        self.novel_class_aux_data = None
        self.seen_class_aux_data = None
        self.num_train_unseen = 0
        self.num_train_mixed = 0

        self.read_mat_dataset()
        self.index_in_epoch = 0
        self.epochs_completed = 0

    def next_batch(self, batch_size):
        """
        Gets batch from train_feature = 7057 samples from 150 train classes
        :param batch_size: batch size of data loading
        :return: batch label, [batch_feature, batch_att]
        """
        idx = tf.random.shuffle(tf.range(self.num_train[0:batch_size]))
        batch_feature = self.data['train_seen']['resnet_features'][idx]
        batch_label = self.data['train_seen']['labels'][idx]
        batch_att = self.aux_data[batch_label]
        return batch_label, [batch_feature, batch_att]

    def read_mat_dataset(self):
        """
        Read a dataset in *.mat format
        :return: return nothing, but set the attribute of the object
        """
        path = self.data_dir + 'res101.mat'
        print('_____')
        print(path)
        mat_content = sio.loadmat(path)
        feature = mat_content['features'].T
        label = mat_content['labels'].astype(int).squeeze() - 1

        path = self.data_dir + 'att_splits.mat'
        mat_content = sio.loadmat(path)
        # numpy array index starts from 0, MatLab starts from 1
        train_val_loc = mat_content['train_val_loc'].squeeze() - 1
        test_seen_loc = mat_content['test_seen_loc'].squeeze() - 1
        test_unseen_loc = mat_content['test_unseen_loc'].squeeze() - 1
        # train_loc = mat_content['train_loc'].squeeze() - 1     # --> train_feature = TRAIN SEEN
        # val_unseen_loc = mat_content['val_loc'].squeeze() - 1  # --> test_unseen_feature = TEST UNSEEN

        if self.aux_data_src == 'attributes':
            self.aux_data = tf.convert_to_tensor(mat_content['att'].T, dtype=tf.float32)
        else:
            if self.dataset != 'CUB':
                print('the specified auxiliary data_source is not available for this dataset')
            else:
                with open(self.data_dir + 'CUB_supporting_data.p', 'rb') as h:
                    x = pickle.load(h)
                    self.aux_data = tf.convert_to_tensor(x[self.aux_data_src], dtype=tf.float32)

                print('loaded ',  self.aux_data_src)

        scaler = preprocessing.MinMaxScaler()

        train_feature = scaler.fit_transform(feature[train_val_loc])
        test_seen_feature = scaler.fit_transform(feature[test_seen_loc])
        test_unseen_feature = scaler.fit_transform(feature[test_unseen_loc])

        train_feature = tf.convert_to_tensor(train_feature, dtype=tf.float32)
        test_seen_feature = tf.convert_to_tensor(test_seen_feature, dtype=tf.float32)
        test_unseen_feature = tf.convert_to_tensor(test_unseen_feature, dtype=tf.float32)

        train_label = tf.convert_to_tensor(label[train_val_loc], dtype=tf.int32)
        test_unseen_label = tf.convert_to_tensor(label[test_unseen_loc], dtype=tf.int32)
        test_seen_label = tf.convert_to_tensor(label[test_seen_loc], dtype=tf.int32)

        self.seen_classes = tf.convert_to_tensor(np.unique(label[train_val_loc]))
        self.novel_classes = tf.convert_to_tensor(np.unique(label[test_unseen_loc]))
        self.num_train = train_feature.size()[0]
        self.num_train_class = self.seen_classes.size(0)
        self.num_test_class = self.novel_classes.size(0)
        self.train_class = self.seen_classes.clone()
        self.all_classes = tf.range(0, self.num_train_class+self.num_test_class, dtype=tf.int32)

        self.train_mapped_label = map_label(train_label, self.seen_classes)

        self.data['train_seen'] = {}
        self.data['train_seen']['resnet_features'] = train_feature
        self.data['train_seen']['labels'] = train_label
        self.data['train_seen'][self.aux_data_src] = self.aux_data[train_label]

        self.data['train_unseen'] = {}
        self.data['train_unseen']['resnet_features'] = None
        self.data['train_unseen']['labels'] = None

        self.data['test_seen'] = {}
        self.data['test_seen']['resnet_features'] = test_seen_feature
        self.data['test_seen']['labels'] = test_seen_label

        self.data['test_unseen'] = {}
        self.data['test_unseen']['resnet_features'] = test_unseen_feature
        self.data['test_unseen'][self.aux_data_src] = self.aux_data[test_unseen_label]
        self.data['test_unseen']['labels'] = test_unseen_label

        self.novel_class_aux_data = self.aux_data[self.novel_classes]
        self.seen_class_aux_data = self.aux_data[self.seen_classes]

    def transfer_features(self, n, num_queries='num_features'):
        print('size before')
        print(self.data['test_unseen']['resnet_features'].size())
        print(self.data['train_seen']['resnet_features'].size())

        print('o' * 100)
        print(self.data['test_unseen'].keys())

        new_train_unseen_embed = None
        new_train_unseen_label = None
        new_test_unseen = None
        new_test_unseen_label = None
        new_train_unseen = None

        for i, s in enumerate(self.novel_classes):
            indices = self.data['test_unseen']['labels'] == s
            features_of_that_class = self.data['test_unseen']['resnet_features'][indices, :]
            num_features = features_of_that_class.size(0)
            indices = tf.random.shuffle(tf.range(num_features))
            if num_queries != 'num_features':
                indices = indices[:n+num_queries]
                
            embed_of_the_class = self.data['test_unseen'][self.aux_data_src][self.data['test_unseen']['labels'] == s, :]

            print(features_of_that_class.size())

            if i == 0:
                new_train_unseen = features_of_that_class[indices[:n], :]
                new_train_unseen_label = s.repeat(n)
                new_test_unseen = features_of_that_class[indices[n:], :]
                new_test_unseen_label = s.repeat(len(indices[n:]))
                new_train_unseen_embed = embed_of_the_class[indices[:n], :]
            else:
                new_train_unseen = tf.concat([new_train_unseen, features_of_that_class[indices[:n], :]], axis=0)
                new_train_unseen_label = tf.concat([new_train_unseen_label, s.repeat(n)], axis=0)
                new_test_unseen = tf.concat([new_test_unseen, features_of_that_class[indices[n:], :]], axis=0)
                new_test_unseen_label = tf.concat([new_test_unseen_label, s.repeat(len(indices[n:]))], axis=0)
                new_train_unseen_embed = tf.concat([new_train_unseen_embed, embed_of_the_class[indices[:n], :]], axis=0)

        print('new_test_unseen.size(): ', new_test_unseen.size())
        print('new_test_unseen_label.size(): ', new_test_unseen_label.size())
        print('new_train_unseen.size(): ', new_train_unseen.size())
        print('new_train_unseen_label.size(): ', new_train_unseen_label.size())
        print('>> num novel classes: ' + str(len(self.novel_classes)))

        #######
        ##
        #######

        self.data['test_unseen']['resnet_features'] = copy.deepcopy(new_test_unseen)
        self.data['test_unseen']['labels'] = copy.deepcopy(new_test_unseen_label)
        self.data['train_unseen']['resnet_features'] = copy.deepcopy(new_train_unseen)
        self.data['train_unseen']['labels'] = copy.deepcopy(new_train_unseen_label)
        self.num_train_unseen = self.data['train_unseen']['resnet_features'].size(0)
        self.data['train_unseen'][self.aux_data_src] = copy.deepcopy(new_train_unseen_embed)

        ####
        self.data['train_seen_unseen_mixed'] = {}
        t1 = self.data['train_seen']['resnet_features']
        t2 = self.data['train_unseen']['resnet_features']
        self.data['train_seen_unseen_mixed']['resnet_features'] = tf.concat([t1, t2], axis=0)

        self.num_train_mixed = self.data['train_seen_unseen_mixed']['resnet_features'].size(0)

        t1 = self.data['train_seen'][self.aux_data_src]
        t2 = self.data['train_unseen'][self.aux_data_src]
        self.data['train_seen_unseen_mixed'][self.aux_data_src] = tf.concat([t1, t2], axis=0)
