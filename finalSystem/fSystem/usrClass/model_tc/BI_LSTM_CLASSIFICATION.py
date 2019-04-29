import math
from fSystem.usrClass import data_helper
import numpy as np
import tensorflow as tf
from fSystem.usrClass import classifyDataHelper
dataPath=dataPath = 'D:/Work/Name Entity Recognition/bi_LSTM_CRF_test/data/'
#dataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/'

class BI_LSTM(object):
    def __init__(self, config,is_crf = True):
        # Parameter
        self.keep_prob = config.keep_prob
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.batch_size = config.batch_size
        self.precision=0
        self.recall=0
        self.max_accuracy = 0
        self.max_f1=0
        self.hidden_dim = config.hidden_neural_size
        self.vocabulary_size = config.vocabulary_size
        self.learning_rate=config.lr
        self.emb_dim = config.embed_dim
        self.num_layers = config.hidden_layer_num
        self.num_epochs = config.num_epoch
        self.num_classes = config.class_num
        self.num_step = config.num_step

        # placeholder of x, y and weight
        self.inputs = tf.placeholder(tf.int32, [None, self.num_step])
        self.targets = tf.placeholder(tf.int32, [None])

        self.embedding = tf.get_variable("embedding", shape=[self.vocabulary_size, self.emb_dim], dtype=tf.float32)

        self.new_embedding = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name="new_embedding")
        self.update_embedding = tf.assign(self.embedding, self.new_embedding)

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)
        self.inputs_emb = tf.transpose(self.inputs_emb, [1, 0, 2])
        self.inputs_emb = tf.reshape(self.inputs_emb, [-1, self.emb_dim])
        self.inputs_emb = tf.split(self.inputs_emb,self.num_step,0)
        # lstm cell
        lstm_cell_fw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)
        lstm_cell_bw = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim)

        # dropout
        if self.is_training ==1:
            lstm_cell_fw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_fw, output_keep_prob=self.keep_prob)
            lstm_cell_bw = tf.nn.rnn_cell.DropoutWrapper(lstm_cell_bw, output_keep_prob=self.keep_prob)

        lstm_cell_fw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_fw] * self.num_layers)
        lstm_cell_bw = tf.nn.rnn_cell.MultiRNNCell([lstm_cell_bw] * self.num_layers)

        # get the length of each sample
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)

        # forward and backward
        self.outputs, _, _ = tf.nn.static_bidirectional_rnn(
            lstm_cell_fw,
            lstm_cell_bw,
            self.inputs_emb,
            dtype=tf.float32,
            sequence_length=self.length
        )
        self.outputs = tf.reduce_mean(self.outputs, 0)
        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs,1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        labels = tf.expand_dims(self.targets, 1)
        indices = tf.expand_dims(tf.range(0, self.batch_size), 1)
        concated = tf.concat([indices, labels], 1)
        onehot_labels = tf.sparse_to_dense(
            concated, tf.stack([self.batch_size, self.num_classes]), 1.0, 0.0)

        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits + 1e-10, labels=onehot_labels)
        self.cost = tf.reduce_mean(self.loss)
        self.prediction = tf.argmax(self.logits, 1)
        correct_prediction = tf.equal(self.prediction, tf.argmax(onehot_labels, 1))
        self.correct_num = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
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

                _, cost_train, length, accuracy,train_summary = \
                    sess.run([
                        self.optimizer,
                        self.cost,
                        self.length,
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

                    prediction,cost_val,accuracy_val, length, val_summary = \
                        sess.run([
                            self.prediction,
                            self.cost,
                            self.accuracy,
                            self.length,
                            self.val_summary
                        ],
                            feed_dict={
                                self.inputs: X_val_batch,
                                self.targets: y_val_batch,
                                self.is_training: 0

                            })
                    precision_val,recall_val,f1_val=self.evaluate(y_val_batch.tolist(),prediction)

                    summary_writer_val.add_summary(val_summary, cnt)
                    print(
                        "iteration: %5d, valid cost: %7.5f, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (
                            iteration, cost_val, precision_val, recall_val, f1_val))


                    if f1_val > self.max_f1:
                        self.max_f1=f1_val
                        self.precision=precision_val
                        self.recall=recall_val

                        save_path = saver.save(sess, save_file)
                        print("saved the best model with f1_score: %.5f" % (self.max_f1))

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
                Y_test_batch = list(Y_test_batch)

                last_size = len(X_test_batch)
                X_test_batch += [[0 for j in range(self.num_step)] for i in range(self.batch_size - last_size)]
                Y_test_batch += [0 for i in range(self.batch_size - last_size)]

                X_test_batch = np.array(X_test_batch)
                Y_test_batch = np.array(Y_test_batch)
                prediction, cost_val, accuracy_val, length, val_summary = \
                    sess.run([
                        self.prediction,
                        self.cost,
                        self.accuracy,
                        self.length,
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
                prediction, cost_val, accuracy_val, length, val_summary = \
                    sess.run([
                        self.prediction,
                        self.cost,
                        self.accuracy,
                        self.length,
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
    def evaluate(self,y,prediction):
        y=[int(item) for item in y]
        prediction=[int(item) for item in prediction]
        precision = [-1.0 for i in range (self.num_classes)]
        recall = [-1.0 for i in range (self.num_classes)]
        f1 = [-1.0 for i in range (self.num_classes)]
        TP = [0 for i in range (self.num_classes)]  # 预测为正，实际为正
        FP = [0 for i in range (self.num_classes)]  # 被模型预测为正的负样本；可以称作误报率
        FN = [0 for i in range (self.num_classes)]  # 被模型预测为负的正样本；可以称作漏报率

        for i in range(len(y)):

            if y[i] == prediction[i]:
                TP[y[i]] += 1

            elif y[i] != prediction[i]:
                FP[prediction[i]] += 1
                FN[y[i]] += 1
        for i in range(self.num_classes):
            if TP[i] + FP[i] != 0:
                precision[i] = 1.0 * TP[i] / (TP[i] + FP[i])
            if TP[i] + FN[i] != 0:
                recall[i] = 1.0 * TP[i] / (TP[i] + FN[i])

            if precision[i] > 0 and recall[i] > 0:
                f1[i] = 2.0 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        precision=[item for item in precision if item>=0]
        recall=[item for item in recall if item>=0]
        f1=[item for item in f1 if item>=0]
        return sum(precision)/len(precision), sum(recall)/len(recall), sum(f1)/len(f1)

