import math
import data_helper
import numpy as np
import tensorflow as tf
import classifyDataHelper
dataPath=dataPath = 'D:/Work/Name Entity Recognition/bi_LSTM_CRF_test/data/'
#dataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/'

class CNN(object):
    def __init__(self, config):
        # Parameter
        self.keep_prob = config.keep_prob
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.batch_size = config.batch_size

        self.max_accuracy = 0
        self.hidden_dim = config.hidden_neural_size
        self.vocabulary_size = config.vocabulary_size
        self.learning_rate=config.lr
        self.emb_dim = config.embed_dim
        self.num_layers = config.hidden_layer_num
        self.num_epochs = config.num_epoch
        self.num_classes = config.class_num
        self.num_step = config.num_step
        self.num_filters=config.num_filters
        self.filter_sizes=config.filter_sizes
        l2_reg_lambda=config.l2_reg_lambda

        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_step])
        self.targets = tf.placeholder(tf.int64, [None])

        self.embedding = tf.get_variable("embedding", shape=[self.vocabulary_size, self.emb_dim], dtype=tf.float32)

        self.new_embedding = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name="new_embedding")
        self.update_embedding = tf.assign(self.embedding, self.new_embedding)

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.embedded_chars_expanded = tf.expand_dims(self.inputs_emb, -1)
        l2_loss = tf.constant(0.0)
        pooled_outputs = []
        for i, filter_size in enumerate(config.filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, self.emb_dim, 1, self.num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[self.num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv")
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                # Maxpooling over the outputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.num_step - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")
                pooled_outputs.append(pooled)

        # Combine all the pooled features
        num_filters_total = self.num_filters * len(self.filter_sizes)
        self.h_pool = tf.concat(pooled_outputs, 3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, self.num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[self.num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.prediction = tf.argmax(self.scores, 1)

        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):


            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.scores, labels=self.targets)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.prediction, self.targets)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        self.cost=self.loss
        self.train_summary = tf.summary.scalar("cost", self.cost)
        self.val_summary = tf.summary.scalar("cost", self.cost)
    def train(self, sess, save_file, X_train, y_train, X_val, y_val):
        saver = tf.train.Saver()

        # merged = tf.merge_all_summaries()
        summary_writer_train = tf.summary.FileWriter('loss_log/classification/train_loss', sess.graph)
        summary_writer_val = tf.summary.FileWriter('loss_log/classification/val_loss', sess.graph)

        num_iterations = int(math.ceil(1.0 * len(X_train) / self.batch_size))

        cnt = 0
        for epoch in range(self.num_epochs):
            # shuffle train in each epoch
            sh_index = np.arange(len(X_train))
            np.random.shuffle(sh_index)
            X_train = X_train[sh_index]
            y_train = y_train[sh_index]
            print("current epoch: %d" % (epoch))
            for iteration in range(num_iterations):
                # train
                X_train_batch, y_train_batch = data_helper.nextBatch(X_train, y_train,
                                                                     start_index=iteration * self.batch_size,
                                                                     batch_size=self.batch_size)
                proweight = []

                _, cost_train, accuracy,train_summary = \
                    sess.run([
                        self.optimizer,
                        self.cost,
                        self.accuracy,
                        self.train_summary
                    ],
                        feed_dict={
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                            self.is_training: 1

                        })

                if iteration % 10 == 0:
                    cnt += 1
                    summary_writer_train.add_summary(train_summary, cnt)
                    print(
                        "iteration: %5d, train cost: %7.5f, train precision: %.5f" % (
                            iteration, cost_train, accuracy))

                # validation
                if iteration % 10 == 0:
                    X_val_batch, y_val_batch = data_helper.nextRandomBatch(X_val, y_val, batch_size=self.batch_size)
                    proweight = []

                    prediction,cost_val,accuracy_val, val_summary = \
                        sess.run([
                            self.prediction,
                            self.cost,
                            self.accuracy,
                            self.val_summary
                        ],
                            feed_dict={
                                self.inputs: X_val_batch,
                                self.targets: y_val_batch,
                                self.is_training: 0

                            })

                    summary_writer_val.add_summary(val_summary, cnt)
                    print(
                        "iteration: %5d, valid cost: %7.5f, valid precision: %.5f" % (
                            iteration, cost_val, accuracy_val))

                    if accuracy_val > self.max_accuracy:

                        self.max_accuracy = accuracy_val
                        #save_path = saver.save(sess, save_file)
                        print("saved the best model with accuracy: %.5f" % (self.max_accuracy))

    def test(self, sess, X_test,Y_test,fileName):
        dataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/'

        label2id, id2label = data_helper.loadMap("tagDic.pkl")

        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print("number of iteration: " + str(num_iterations))

        for i in range(num_iterations):
            print("iteration: " + str(i + 1))
            results = []
            X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
            Y_test_batch = Y_test[i * self.batch_size: (i + 1) * self.batch_size]

            if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                X_test_batch = list(X_test_batch)

                last_size = len(X_test_batch)
                X_test_batch += [[0 for j in range(self.num_step)] for i in range(self.batch_size - last_size)]

                X_test_batch = np.array(X_test_batch)
                prediction, cost_val, accuracy_val,  val_summary = \
                    sess.run([
                        self.prediction,
                        self.cost,
                        self.accuracy,
                        self.val_summary
                    ],
                        feed_dict={
                            self.inputs: X_test_batch,
                            self.targets:Y_test_batch,

                            self.is_training: 0

                        })
                classifyDataHelper.getTestResult(prediction, [], dataPath + 'predictResult.txt',
                                                 fileName)

            else:
                X_test_batch = np.array(X_test_batch)
                prediction, cost_val, accuracy_val, val_summary = \
                    sess.run([
                        self.prediction,
                        self.cost,
                        self.accuracy,
                        self.val_summary
                    ],
                        feed_dict={
                            self.inputs: X_test_batch,
                            self.targets:Y_test_batch,
                            self.is_training: 0

                        })
                classifyDataHelper.getTestResult(prediction, [], dataPath + 'predictResult.txt',
                                                 fileName)
    def assign_embedding(self, session, new_embedding):

        session.run(self.update_embedding, feed_dict={self.new_embedding: new_embedding})


