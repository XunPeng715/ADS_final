__author__ = 'mangate'

import tensorflow as tf
import numpy as np

import os
import sys

tf.app.flags.DEFINE_integer('model_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

class Model(object):

    def __init__(self):
        self.graph = tf.Graph()

    def build_model(self, embedding_dim, vocab_size, filter_sizes, num_filters, vector_length,
                    num_classes=2, trainable=True):
        with self.graph.as_default():

            self.num_classes = num_classes

            # Placehodlers for regular data
            self.input_x = tf.placeholder(tf.int32, [None, vector_length], name="input_x")
            self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")

            # placeholder for dropout prob
            self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
            
            # Embedding layer
            # with tf.device('/cpu:0'), tf.name_scope("embedding"):
            embedd = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]),
                                 trainable=trainable, name="embedd")

            # Initizalizing the layer
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = embedd.assign(self.embedding_placeholder)

            # Setting the embedding as input to the network
            embedded_sent = tf.nn.embedding_lookup(embedd, self.input_x)
            # Adding 1 channel to the embedding
            embedded_sent_expended = tf.expand_dims(embedded_sent, -1)

            l2_loss = tf.constant(0.0)

            # Pleaceholder for learning rate
            self.tf_learning_rate_decay = tf.placeholder(tf.float32)

            # Creating conv layer and pooling for each of the filters
            pooled_outputs = []
            for i, filter_size in enumerate(filter_sizes):
                with tf.name_scope("conv-maxpool-%s" % filter_size):
                    # Convolution Layer
                    filter_shape = [filter_size, embedding_dim, 1, num_filters]
                    # Weights for each filter
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    # Bias for each filter
                    b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                    # The actual convolution
                    conv = tf.nn.conv2d(
                        embedded_sent_expended,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    # Apply nonlinearity
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                    # Maxpooling over the outputs
                    pooled = tf.nn.max_pool(
                        h,
                        ksize=[1, vector_length - filter_size + 1, 1, 1],
                        strides=[1, 1, 1, 1],
                        padding='VALID',
                        name="pool")
                    pooled_outputs.append(pooled)

            # Combine all the pooled features
            num_filters_total = num_filters * len(filter_sizes)
            h_pool = tf.concat(pooled_outputs, 3)
            h_pool_flat = tf.reshape(h_pool, [-1, num_filters_total])

            # Add dropout
            with tf.name_scope("dropout"):
                h_drop = tf.nn.dropout(h_pool_flat, self.dropout_keep_prob)

            # Final (unnormalized) scores and predictions
            with tf.name_scope("output"):
                W = tf.get_variable(
                    "W",
                    shape=[num_filters_total, num_classes],
                    initializer=tf.contrib.layers.xavier_initializer())
                b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
                # TODO bring back L2 loss
                l2_loss += tf.nn.l2_loss(W)
                l2_loss += tf.nn.l2_loss(b)
                scores = tf.nn.xw_plus_b(h_drop, W, b, name="scores")
                self.predictions = tf.argmax(scores, 1, name="predictions")
                
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=self.input_y)
            # TODO bring back l2 loss
            self.loss = tf.reduce_mean(losses) + 0.15 * l2_loss

            # Training
            # TODO add decaying learning rate?
            learning_rate = 1e-3 * self.tf_learning_rate_decay
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)
            #self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=learning_rate,epsilon=1e-6).minimize(self.loss)
            # Accuracy
            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

                
    def run(self, revs, embedding, word_idx_map, max_l, k_fold=2):
        with tf.Session(graph=self.graph) as sess:
            # Initialization
            sess.run(tf.initialize_all_variables())
            sess.run(self.embedding_init, feed_dict={self.embedding_placeholder: embedding})
            # Training steps
            batch_size = 100

            epoch_num = 0
            learning_rate_decay = 1
            
            # train_data, train_labels, test_data, test_labels = self.create_data(revs, word_idx_map, 1, max_l, self.num_classes)
            train_data_0, train_labels_0, train_data_1, train_labels_1, test_data, test_labels =     self.make_idx_data_cv(revs, word_idx_map, 1, max_l, 5)
            print(len(test_data))
            for count in range(9):              
                train_data, train_labels = self.create_data(train_data_0, train_labels_0, train_data_1, train_labels_1, count)
                num_epochs = 25
                # 1332 / 50 = 26
                steps_per_epoche = int(len(train_labels) / batch_size)
                # 390
                num_steps = steps_per_epoche * num_epochs
                for step in range(num_steps):
                    # Shuffle the data in each epoch
                    if (step % steps_per_epoche == 0):
                        shuffle_indices = np.random.permutation(np.arange(len(train_data)))
                        train_data = train_data[shuffle_indices]
                        train_labels = train_labels[shuffle_indices]
                        print("epoche number %d" % epoch_num)
                        # Get test results on each epoch
                        feed_dict = {self.input_x: test_data, self.input_y: test_labels, self.dropout_keep_prob: 1.0}
                        accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
                        print('Test accuracy: %.3f' % accuracy_out)
                        # On each 8 epochs we decay the learning rate
                        if(epoch_num == 8):
                            learning_rate_decay = learning_rate_decay*0.5
                        if(epoch_num == 16):
                            learning_rate_decay = learning_rate_decay*0.1
                        epoch_num += 1
                    offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
                    batch_data = train_data[offset:(offset + batch_size), :]
                    batch_labels = train_labels[offset:(offset + batch_size), :]
                    # Setting all placeholders
                    feed_dict = {self.input_x: batch_data, self.input_y: batch_labels, self.dropout_keep_prob: 0.5,
                                     self.tf_learning_rate_decay: learning_rate_decay}
                    _, l, accuracy_out = sess.run(
                            [self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
             # Testing
            feed_dict = {self.input_x: test_data, self.input_y: test_labels, self.dropout_keep_prob: 1.0}
            accuracy_out = sess.run(self.accuracy, feed_dict=feed_dict)
            print('Test accuracy: %.3f' % accuracy_out)
			
			# Export model
            export_path_base = '/tmp/cnn_tweet_model'
            export_path = os.path.join(
                    tf.compat.as_bytes(export_path_base),
                    tf.compat.as_bytes(str(FLAGS.model_version)))
            print 'Exporting trained model to', export_path
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # Build the signature_def_map.
			
            '''
			classification_inputs = tf.saved_model.utils.build_tensor_info(
                    serialized_tf_example)
            classification_outputs_classes = tf.saved_model.utils.build_tensor_info(
                    prediction_classes)
            classification_outputs_scores = tf.saved_model.utils.build_tensor_info(values)

            classification_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={
                                    tf.saved_model.signature_constants.CLASSIFY_INPUTS:
                                            classification_inputs
                            },
                            outputs={
                                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_CLASSES:
                                            classification_outputs_classes,
                                    tf.saved_model.signature_constants.CLASSIFY_OUTPUT_SCORES:
                                            classification_outputs_scores
                            },
                            method_name=tf.saved_model.signature_constants.CLASSIFY_METHOD_NAME))
            '''
							
            tensor_info_x = tf.saved_model.utils.build_tensor_info(self.input_x)
            tensor_info_y = tf.saved_model.utils.build_tensor_info(self.predictions)
            tensor_info_dropout_keep_prob = tf.saved_model.utils.build_tensor_info(self.dropout_keep_prob)

            prediction_signature = (
                    tf.saved_model.signature_def_utils.build_signature_def(
                            inputs={'images': tensor_info_x, 'prob': tensor_info_dropout_keep_prob},
                            outputs={'scores': tensor_info_y},
                            method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))

            legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')
            builder.add_meta_graph_and_variables(
                    sess, [tf.saved_model.tag_constants.SERVING],
                    signature_def_map={
                            'predict_images':
                                    prediction_signature,
                    },
                    legacy_init_op=legacy_init_op)

            builder.save()

            print 'Done exporting!'

    # maximuam of count is 8
    def create_data(self, train_data_0, train_labels_0, train_data_1, train_labels_1, count):
        train_data, train_labels = [], []
        length = len(train_labels_1)
    
        if (count + 1) * length < len(train_data_0) or (count + 1) * length == len(train_data_0):
            for i in range(length):
                train_data.append(train_data_1[i])
                train_labels.append(train_labels_1[i])
                train_data.append(train_data_0[i + count * length])
                train_labels.append(train_labels_0[i + count * length])
        elif count * length < len(train_data_0):
            prev = len(train_data_0) - count * length
            for i in range(prev):
                train_data.append(train_data_0[i + count * length])
                train_labels.append(train_labels_0[i + count * length])
                train_data.append(train_data_1[i])
                train_labels.append(train_labels_1[i])
                
    
        train_data = np.array(train_data, dtype="int")
        train_labels = np.asarray(train_labels, dtype=np.float32)
    
        shuffle_indices = np.random.permutation(np.arange(len(train_data)))
        train_data = train_data[shuffle_indices]
        train_labels = train_labels[shuffle_indices]
        return train_data, train_labels


    def make_idx_data_cv(self, revs, word_idx_map, cv, max_l, filter_h=5):
        """
        Transforms sentences into a 2-d matrix.
        """

        train_data_0, train_labels_0 = [], []
        train_data_1, train_labels_1 = [], []
        test_data, test_labels = [], []

        for rev in revs:
            # convert sentence to 1 * 64 (represent by number)
            sent = self.get_idx_from_sent(rev["text"], word_idx_map, max_l, filter_h)
            label = [0] * self.num_classes
            label[rev["y"]] = 1
            if rev["split"] == cv:
                test_data.append(sent)
                test_labels.append(label)
            else:
                if rev["y"] == 0:
                    train_data_0.append(sent)
                    train_labels_0.append(label)
                elif rev["y"] == 1:
                    train_data_1.append(sent)
                    train_labels_1.append(label)
        return [train_data_0, train_labels_0, train_data_1, train_labels_1, test_data, test_labels]


    def get_idx_from_sent(self, sent, word_idx_map, max_l, filter_h=5):
        """
        Transforms sentence into a list of indices. Pad with zeroes.
        """
        x = []
        pad = filter_h - 1
        for i in xrange(pad):
            x.append(0)
        words = sent.split()
        for word in words:
            if word in word_idx_map:
                x.append(word_idx_map[word])
        while len(x) < max_l + 2 * pad:
            x.append(0)
        return x