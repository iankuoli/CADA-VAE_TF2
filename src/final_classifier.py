from torch.utils.data import Dataset, DataLoader
import tensorflow as tf


def map_label(label, classes):
    mapped_label = tf.zeros(label.size(), dtype=tf.int32)
    for i in range(classes.size(0)):
        mapped_label[label == classes[i]] = i

    return mapped_label


class Classifier:
    def __init__(self, model, train_x, train_y, test_seen_x, test_seen_y, test_novel_x, test_novel_y, 
                 seen_classes, novel_classes, num_class, 
                 lr=0.001, beta1=0.5, num_epoch=20, batch_size=100, generalized=True,
                 train_only=False, test_only=False, do_nothing=False):

        self.train_only = train_only
        
        self.batch_size = batch_size
        self.num_epoch = num_epoch
        self.num_class = num_class
        self.input_dim = train_x.size(1)
        print('self.input_dim = %d' % self.input_dim)

        self.average_loss = 0
        self.model = model
        self.criterion = model.lossfunction

        self.lr = lr
        self.beta1 = beta1

        self.optimizer = tf.optimizers.Adam(learning_rate=lr, beta_1=0.999)

        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.num_train = train_x.size()[0]
        
        self.loss = 0

        self.current_epoch = 0

        self.acc_novel = 0
        self.acc_seen = 0
        self.H = 0
        self.acc = 0
        self.intra_epoch_accuracies = [()]*10
        
        if not do_nothing:
            if not generalized:
                print('...')

            if not test_only:
                if generalized:
                    self.acc_seen, self.acc_novel, self.H = self.fit(train_x, train_y, 
                                                                     test_seen_x, test_seen_y, seen_classes, 
                                                                     test_novel_x, test_novel_y, novel_classes)
                else:
                    self.acc = self.fit_zsl(train_x, train_y, 
                                            test_novel_x, test_novel_y, novel_classes)
            else:
                if generalized:
                    best_seen = 0
                    best_novel = 0
                    best_h = -1
                    acc_seen = self.val(test_seen_x, test_seen_y, seen_classes, use_gzsl=True)
                    acc_novel = self.val(test_novel_x, test_novel_y, novel_classes, use_gzsl=True)

                    h = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel) if (acc_seen + acc_novel) > 0 else 0

                    if h > best_h:
                        best_seen = acc_seen
                        best_novel = acc_novel
                        best_h = h
                    self.acc_seen = best_seen
                    self.acc_novel = best_novel
                    self.H = best_h
                else:
                    acc = self.val(test_novel_x, test_novel_y, novel_classes)
                    self.acc = acc

    def fit_zsl(self, train_x, train_y, test_novel_x, test_novel_y, novel_classes):
        best_acc = 0
        
        for epoch in range(self.num_epoch):
            for i in range(0, self.num_train, self.batch_size):
                batch_input, batch_label = self.next_batch(train_x, train_y, self.batch_size)
                
                with tf.GradientTape() as tape:        
                    output = self.model(batch_input)
                    loss = self.criterion(output, batch_label)

                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            self.loss = loss
            self.current_epoch += 1
            
            acc = 0
            if not self.train_only:
                acc = self.val(test_novel_x, test_novel_y, novel_classes)
            best_acc = acc if acc > best_acc else best_acc
        
        return best_acc

    def fit(self, train_x, train_y, test_seen_x, test_seen_y, seen_classes, test_novel_x, test_novel_y, novel_classes):
        best_h = -1
        best_seen = 0
        best_novel = 0

        data_loader = DataLoader(TrainDataset(train_x, train_y), 
                                 batch_size=self.batch_size, shuffle=True, drop_last=True)

        iterations_per_epoch = int(self.num_train / self.batch_size)
                
        for epoch in range(self.num_epoch):
            self.average_loss = 0
            i = 0
            for batch in data_loader:
                with tf.GradientTape() as tape:
                    output = self.model(batch['x'])
                    loss = self.criterion(output, batch['y'])
                gradients = tape.gradient(loss, self.model.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                if i > 0.8 * iterations_per_epoch:
                    self.average_loss += loss.item() / (0.2 * iterations_per_epoch)
                i += 1
            self.current_epoch += 1
            
            acc_seen = 0
            acc_novel = 0
            if not self.train_only:
                acc_seen = self.val(test_seen_x, test_seen_y, seen_classes, use_gzsl=True)
                acc_novel = self.val(test_novel_x, test_novel_y, novel_classes, use_gzsl=True)

            h = (2 * acc_seen * acc_novel) / (acc_seen + acc_novel) if (acc_seen + acc_novel) > 0 else 0

            if h > best_h:
                best_seen = acc_seen
                best_novel = acc_novel
                best_h = h
        self.loss = loss
        
        return best_seen, best_novel, best_h

    def next_batch(self, train_x, train_y, batch_size):
        start = self.index_in_epoch

        # At the first epoch, shuffle the data -------------------------------------------------------------------------
        if self.epochs_completed == 0 and start == 0:      
            perm = tf.random.shuffle(tf.range(self.num_train))
            train_x = train_x[perm]
            train_y = train_y[perm]

        # At the last batch, go back to the start ----------------------------------------------------------------------
        if start + batch_size > self.num_train:
            self.epochs_completed += 1
            
            # shuffle the data
            perm = tf.random.shuffle(tf.range(self.num_train))
            train_x = train_x[perm]
            train_y = train_y[perm]
            
            # start next epoch
            rest_num_examples = self.num_train - start
            self.index_in_epoch = batch_size - rest_num_examples
            x_new_part = train_x[0:self.index_in_epoch]
            y_new_part = train_y[0:self.index_in_epoch]              
                
            if rest_num_examples > 0:
                x_rest_part = train_x[start:self.num_train]
                y_rest_part = train_y[start:self.num_train]
                return tf.concat([x_rest_part, x_new_part], 0), tf.concat([y_rest_part, y_new_part], 0)
            else:
                return x_new_part, y_new_part
        else:
            # from index start to index end-1
            self.index_in_epoch += batch_size
            end = self.index_in_epoch
            return train_x[start:end], train_y[start:end]

    def val(self, test_x, test_label, target_classes, use_gzsl=False):
        start = 0
        num_test = test_x.size()[0]
        predicted_label = tf.zeros(test_label.size(), dtype=tf.float32)
        
        for i in range(0, num_test, self.batch_size):
            end = min(num_test, start+self.batch_size)
            output = self.model(test_x[start:end])
            predicted_label[start:end] = tf.argmax(output.data, 1)
            start = end
        
        if use_gzsl:
            acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        else:
            new_target_classes = tf.range(target_classes.size(0))
            acc = self.compute_per_class_acc_gzsl(map_label(test_label, target_classes), predicted_label,
                                                  new_target_classes)
        return acc

    @staticmethod
    def compute_per_class_acc_gzsl(test_label, predicted_label, target_classes):
        per_class_accuracies = tf.zeros(target_classes.size()[0], dtype=tf.float32)

        for i in range(target_classes.size()[0]):
            is_class = (test_label == target_classes[i])
            tmp_a = tf.dtypes.cast(tf.reduce_sum(predicted_label[is_class] == test_label[is_class]), dtype=tf.float32)
            tmp_b = tf.dtypes.cast(tf.reduce_sum(is_class), dtype=tf.float32)
            per_class_accuracies[i] = tf.math.divide(tmp_a, tmp_b)
        return tf.reduce_mean(per_class_accuracies)


class TrainDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, train_x, train_y):
        self.train_x = train_x
        self.train_y = train_y.long()

    def __len__(self):
        return self.train_x.size(0)

    def __getitem__(self, idx):
        return {'x': self.train_x[idx, :], 'y': self.train_y[idx]}
