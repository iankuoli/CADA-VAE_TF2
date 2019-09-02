#vaemodel

import copy
import tensorflow as tf
from tensorflow.keras import Model, layers, regularizers

from src.data_loader import DataLoader
import src.final_classifier as classifier
from src.utils import map_label


class LinearLogSoftMax(layers.Layer):

    def compute_output_signature(self, input_signature):
        pass

    def __init__(self, input_dim, num_class):
        super(LinearLogSoftMax, self).__init__()

        self.fc = layers.Dense(num_class, activation=tf.nn.relu,
                               kernel_regularizer=regularizers.l2(0.01),
                               bias_regularizer=regularizers.l2(0.01))
        self.logic = tf.nn.log_softmax()
        self.loss_function = tf.nn.softmax_cross_entropy_with_logits()

    def call(self, x, training=False):
        x = self.logic(self.fc(x))
        return x


class Model(layers.Layer):

    def __init__(self, hyperparameters):
        super(Model, self).__init__()

        self.auxiliary_data_source = hyperparameters['auxiliary_data_source']
        self.all_data_sources = ['resnet_features', self.auxiliary_data_source]
        self.DATASET = hyperparameters['dataset']
        self.num_shots = hyperparameters['num_shots']
        self.latent_size = hyperparameters['latent_size']
        self.batch_size = hyperparameters['batch_size']
        self.hidden_size_rule = hyperparameters['hidden_size_rule']
        self.warm_up = hyperparameters['model_specifics']['warm_up']
        self.generalized = hyperparameters['generalized']
        self.classifier_batch_size = 32
        self.img_seen_samples = hyperparameters['samples_per_class'][self.DATASET][0]
        self.att_seen_samples = hyperparameters['samples_per_class'][self.DATASET][1]
        self.att_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][2]
        self.img_unseen_samples = hyperparameters['samples_per_class'][self.DATASET][3]
        self.recog_loss_function = hyperparameters['loss']
        self.num_epoch = hyperparameters['epochs']
        self.lr_cls = hyperparameters['lr_cls']
        self.cross_reconstruction = hyperparameters['model_specifics']['cross_reconstruction']
        self.cls_train_epochs = hyperparameters['cls_train_steps']
        self.dataset = DataLoader(self.DATASET, copy.deepcopy(self.auxiliary_data_source))
        self.reparameterize_with_noise = False
        self.current_epoch = 0

        if self.DATASET == 'CUB':
            self.num_classes = 200
            self.num_novel_classes = 50
        elif self.DATASET == 'SUN':
            self.num_classes = 717
            self.num_novel_classes = 72
        elif self.DATASET == 'AWA1' or self.DATASET == 'AWA2':
            self.num_classes = 50
            self.num_novel_classes = 10

        feature_dimensions = [2048, self.dataset.aux_data.size(1)]

        # Here, the encoders and decoders for all modalities are created and put into dict

        self.encoder = {}

        for datatype, dim in zip(self.all_data_sources,feature_dimensions):
            self.encoder[datatype] = models.encoder_template(dim, self.latent_size, self.hidden_size_rule[datatype])
            print(str(datatype) + ' ' + str(dim))

        self.decoder = {}
        for datatype, dim in zip(self.all_data_sources, feature_dimensions):
            self.decoder[datatype] = models.decoder_template(self.latent_size, dim, self.hidden_size_rule[datatype])

        # An optimizer for all encoders and decoders is defined here
        self.parameters_to_optimize = list(self.parameters())
        for datatype in self.all_data_sources:
            self.parameters_to_optimize += list(self.encoder[datatype].parameters())
            self.parameters_to_optimize += list(self.decoder[datatype].parameters())

        self.optimizer = tf.optimizers.Adam(learning_rate=hyperparameters['lr_gen_model'], beta_1=0.9, beta_2=0.999,
                                            epsilon=1e-08, decay=0, amsgrad=True)

        if self.recog_loss_function == 'l2':
            # size_average=False
            self.reconstruction_criterion = tf.keras.losses.MSE()

        elif self.recog_loss_function == 'l1':
            # size_average=False
            self.reconstruction_criterion = tf.keras.losses.MAE()

    def reparameterize(self, mu, log_var):
        if self.reparameterize_with_noise:
            sigma = tf.math.exp(log_var)
            eps = tf.random.normal(sigma.shape, mean=0.0, stddev=1.0)
            return mu + sigma * eps
        else:
            return mu

    def forward(self):
        pass

    def train_step(self, img, attr):

        # --------------------------------------------------------------------------------------------------------------
        # scale the loss terms according to the warm_up schedule
        tmp1 = 1.0 * (self.current_epoch - self.warm_up['cross_reconstruction']['start_epoch'])
        tmp2 = 1.0 * (self.warm_up['cross_reconstruction']['end_epoch'] -
                      self.warm_up['cross_reconstruction']['start_epoch'])
        f1 = tmp1 / tmp2
        f1 = f1 * (1.0 * self.warm_up['cross_reconstruction']['factor'])
        cross_reconstruction_factor = tf.zeros(min(max(f1, 0), self.warm_up['cross_reconstruction']['factor']),
                                               dtype=tf.float32)

        tmp1 = 1.0 * (self.current_epoch - self.warm_up['beta']['start_epoch'])
        tmp2 = 1.0 * (self.warm_up['beta']['end_epoch'] - self.warm_up['beta']['start_epoch'])
        f2 = tmp1 / tmp2
        f2 = f2 * (1.0 * self.warm_up['beta']['factor'])
        beta = tf.zeros(min(max(f2, 0), self.warm_up['beta']['factor']), dtype=tf.float32)

        tmp1 = 1.0 * (self.current_epoch - self.warm_up['distance']['start_epoch'])
        tmp2 = 1.0 * (self.warm_up['distance']['end_epoch'] - self.warm_up['distance']['start_epoch'])
        f3 = tmp1 / tmp2
        f3 = f3 * (1.0 * self.warm_up['distance']['factor'])
        distance_factor = tf.zeros(min(max(f3, 0), self.warm_up['distance']['factor']))

        with tf.GradientTape() as tape:
            # ----------------------------------------------------------------------------------------------------------
            # Encode image features and additional features
            mu_img, log_var_img = self.encoder['resnet_features'](img)
            z_from_img = self.reparameterize(mu_img, log_var_img)

            mu_att, log_var_att = self.encoder[self.auxiliary_data_source](attr)
            z_from_att = self.reparameterize(mu_att, log_var_att)

            # ----------------------------------------------------------------------------------------------------------
            # Reconstruct inputs
            img_from_img = self.decoder['resnet_features'](z_from_img)
            att_from_att = self.decoder[self.auxiliary_data_source](z_from_att)

            reconstruction_loss = self.reconstruction_criterion(img_from_img, img) + \
                                  self.reconstruction_criterion(att_from_att, attr)

            # ----------------------------------------------------------------------------------------------------------
            # Cross Reconstruction Loss
            img_from_att = self.decoder['resnet_features'](z_from_att)
            att_from_img = self.decoder[self.auxiliary_data_source](z_from_img)

            cross_reconstruction_loss = self.reconstruction_criterion(img_from_att, img) + \
                                        self.reconstruction_criterion(att_from_img, attr)

            # ----------------------------------------------------------------------------------------------------------
            # KL-Divergence
            kld = (0.5 * tf.reduce_sum(1 + log_var_att - mu_att.pow(2) - log_var_att.exp())) + \
                  (0.5 * tf.reduce_sum(1 + log_var_img - mu_img.pow(2) - log_var_img.exp()))

            # ----------------------------------------------------------------------------------------------------------
            # Distribution Alignment
            tmp1 = tf.reduce_sum((mu_img - mu_att) ** 2, dim=1)
            tmp2 = tf.reduce_sum((tf.math.sqrt(log_var_img.exp()) - tf.math.sqrt(log_var_att.exp())) ** 2, dim=1)
            distance = tf.reduce_sum(tf.sqrt(tmp1 + tmp2))

            # ----------------------------------------------------------------------------------------------------------
            # Put the loss together and call the optimizer
            loss = reconstruction_loss - beta * kld

            if cross_reconstruction_loss > 0:
                loss += cross_reconstruction_factor * cross_reconstruction_loss
            if distance_factor > 0:
                loss += distance_factor * distance

        gradients = tape.gradient(loss, self.parameters_to_optimize)
        self.optimizer.apply_gradients(zip(gradients, self.parameters_to_optimize))

        return loss

    def train_vae(self):

        losses = []

        # leave both statements
        self.train()
        self.reparameterize_with_noise = True

        print('train for reconstruction')
        for epoch in range(0, self.num_epoch):
            self.current_epoch = epoch
            i = -1
            for iters in range(0, self.dataset.ntrain, self.batch_size):
                i += 1
                label, data_from_modalities = self.dataset.next_batch(self.batch_size)
                loss = self.train_step(data_from_modalities[0], data_from_modalities[1])

                if i % 50 == 0:
                    print('epoch  %d | %d \t | loss = %f' % (epoch, i, loss))
                    if i > 0:
                        losses.append(loss)

        # turn into evaluation mode:
        for key, value in self.encoder.items():
            self.encoder[key].eval()
        for key, value in self.decoder.items():
            self.decoder[key].eval()

        return losses

    def train_classifier(self):

        if self.num_shots > 0 :
            print('================  transfer features from test to train ==================')
            self.dataset.transfer_features(self.num_shots, num_queries='num_features')

        history = []  # stores accuracies

        cls_seen_classes = self.dataset.seen_classes
        cls_novel_classes = self.dataset.novel_classes

        train_seen_feat = self.dataset.data['train_seen']['resnet_features']
        train_seen_label = self.dataset.data['train_seen']['labels']

        novel_class_aux_data = self.dataset.novel_class_aux_data
        seen_class_aux_data = self.dataset.seen_class_aux_data

        novel_corresponding_labels = self.dataset.novel_classes.long().to(self.device)
        seen_corresponding_labels = self.dataset.seen_classes.long().to(self.device)

        # The resnet_features for testing the classifier are loaded here
        novel_test_feat = self.dataset.data['test_unseen']['resnet_features']
        seen_test_feat = self.dataset.data['test_seen']['resnet_features']
        test_seen_label = self.dataset.data['test_seen']['labels']
        test_novel_label = self.dataset.data['test_unseen']['labels']
        train_unseen_feat = self.dataset.data['train_unseen']['resnet_features']
        train_unseen_label = self.dataset.data['train_unseen']['labels']

        # in ZSL mode:
        if not self.generalized:
            # there are only 50 classes in ZSL (for CUB)
            # novel_corresponding_labels = list of all novel classes (as tensor)
            # test_novel_label = mapped to 0-49 in classifier function
            # those are used as targets, they have to be mapped to 0-49 right here:
            novel_corresponding_labels = map_label(novel_corresponding_labels, novel_corresponding_labels)

            if self.num_shots > 0:
                # not generalized and at least 1 shot means normal FSL setting (use only unseen classes)
                train_unseen_label = map_label(train_unseen_label, cls_novel_classes)

            # for FSL, we train_seen contains the unseen class examples
            # for ZSL, train seen label is not used
            # if self.num_shots>0:
            #    train_seen_label = map_label(train_seen_label,cls_novel_classes)
            test_novel_label = map_label(test_novel_label, cls_novel_classes)

            # map cls novel_classes last
            cls_novel_classes = map_label(cls_novel_classes, cls_novel_classes)

        if self.generalized:
            print('mode: gzsl')
            clf = LinearLogSoftMax(self.latent_size, self.num_classes)
        else:
            print('mode: zsl')
            clf = LinearLogSoftMax(self.latent_size, self.num_novel_classes)

        clf.apply(models.weights_init)

        # --------------------------------------------------------------------------------------------------------------
        # Model Inference
        #

        # -------------------------------------------------------------------------------------------------------------#
        # Preparing the test set:
        #   convert raw test data into z vectors
        self.reparameterize_with_noise = False

        mu1, var1 = self.encoder['resnet_features'](novel_test_feat)
        test_novel_x = self.reparameterize(mu1, var1).to(self.device).data
        test_novel_y = test_novel_label.to(self.device)

        mu2, var2 = self.encoder['resnet_features'](seen_test_feat)
        test_seen_x = self.reparameterize(mu2, var2).to(self.device).data
        test_seen_y = test_seen_label.to(self.device)

        # -------------------------------------------------------------------------------------------------------------#
        # Preparing the train set:
        #   chose n random image features per class. If n exceeds the number of image features per class, duplicate
        #   some. Next, convert them to latent z features.
        self.reparameterize_with_noise = True

        def sample_train_data(features, label, sample_per_class):
            """
            Sample_train_data_on_sample_per_class_basis
            :param features: 
            :param label: 
            :param sample_per_class: 
            :return: 
            """
            sample_per_class = int(sample_per_class)

            if sample_per_class != 0 and len(label) != 0:
                classes = label.unique()

                features_to_return = 0
                labels_to_return = 0
                for i, s in enumerate(classes):
                    # order of features and labels must coincide
                    features_of_that_class = features[label == s, :]
                    
                    # if number of selected features is smaller than the number of features we want per class:
                    tmp = max(1, sample_per_class / features_of_that_class.size(0))
                    multiplier = tf.cast(tf.math.ceil(tmp), dtype=tf.int32)

                    features_of_that_class = features_of_that_class.repeat(multiplier, 1)
                    if i == 0:
                        features_to_return = features_of_that_class[:sample_per_class, :]
                        labels_to_return = s.repeat(sample_per_class)
                    else:
                        features_to_return = tf.concat([features_to_return, 
                                                        features_of_that_class[:sample_per_class, :]], dim=0)
                        labels_to_return = tf.concat([labels_to_return, s.repeat(sample_per_class)], dim=0)

                return features_to_return, labels_to_return
            else:
                return None, None

        # Some of the following might be empty tensors if the specified number of samples is zero :
        img_seen_feat, img_seen_label = sample_train_data(train_seen_feat, train_seen_label, 
                                                          self.img_seen_samples)
        img_unseen_feat, img_unseen_label = sample_train_data(train_unseen_feat, train_unseen_label, 
                                                              self.img_unseen_samples)

        att_unseen_feat, att_unseen_label = sample_train_data(novel_class_aux_data, novel_corresponding_labels, 
                                                              self.att_unseen_samples )
        att_seen_feat, att_seen_label = sample_train_data(seen_class_aux_data, seen_corresponding_labels, 
                                                          self.att_seen_samples)

        def convert_datapoints_to_z(features, encoder):
            if features.size(0) != 0:
                mu_, log_var_ = encoder(features)
                return self.reparameterize(mu_, log_var_)
            else:
                return None

        z_seen_img = convert_datapoints_to_z(img_seen_feat, self.encoder['resnet_features'])
        z_unseen_img = convert_datapoints_to_z(img_unseen_feat, self.encoder['resnet_features'])

        z_seen_att = convert_datapoints_to_z(att_seen_feat, self.encoder[self.auxiliary_data_source])
        z_unseen_att = convert_datapoints_to_z(att_unseen_feat, self.encoder[self.auxiliary_data_source])

        train_z = [z_seen_img, z_unseen_img, z_seen_att, z_unseen_att]
        train_l = [img_seen_label, img_unseen_label, att_seen_label, att_unseen_label]

        # empty tensors are sorted out
        train_x = [train_z[i] for i in range(len(train_z)) if train_z[i].size(0) != 0]
        train_y = [train_l[i] for i in range(len(train_l)) if train_z[i].size(0) != 0]

        # -------------------------------------------------------------------------------------------------------------#
        # Initializing the classifier and train one epoch
        cls = classifier.Classifier(clf, train_x, train_y, test_seen_x, test_seen_y, test_novel_x,
                                    test_novel_y,
                                    cls_seen_classes, cls_novel_classes,
                                    self.num_classes, self.device, self.lr_cls, 0.5, 1,
                                    self.classifier_batch_size,
                                    self.generalized)

        for k in range(self.cls_train_epochs):
            if k > 0:
                if self.generalized:
                    cls.acc_seen, cls.acc_novel, cls.H = cls.fit(train_x, train_y,
                                                                 test_seen_x, test_seen_y, cls_seen_classes,
                                                                 test_novel_x, test_novel_y, cls_novel_classes)
                else:
                    cls.acc = cls.fit_zsl(train_x, train_y, test_novel_x, test_novel_y, cls_novel_classes)

            if self.generalized:

                print('[%.1f]  novel=%.4f, seen=%.4f, h=%.4f, loss=%.4f' %
                      (k, cls.acc_novel, cls.acc_seen, cls.H, cls.average_loss))

                history.append([tf.convert_to_tensor(cls.acc_seen, dtype=tf.float32), 
                                tf.convert_to_tensor(cls.acc_novel, dtype=tf.float32),
                                tf.convert_to_tensor(cls.H, dtype=tf.float32)])
            else:
                print('[%.1f]  acc=%.4f ' % (k, cls.acc))
                history.append([0, tf.convert_to_tensor(cls.acc, dtype=tf.float32), 0])

        if self.generalized:
            return tf.convert_to_tensor(cls.acc_seen, dtype=tf.float32), \
                   tf.convert_to_tensor(cls.acc_novel, dtype=tf.float32), \
                   tf.convert_to_tensor(cls.H, dtype=tf.float32)
        else:
            return 0, tf.convert_to_tensor(cls.acc, dtype=tf.float32), 0, history
