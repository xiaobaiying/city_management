import math
import data_helper
import numpy as np
import tensorflow as tf


class LSTM_CRF(object):
    def __init__(self, config,is_crf = True):
        # Parameter
        self.keep_prob = config.keep_prob
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.batch_size = config.batch_size
        self.recall = 0
        self.precesion = 0
        self.max_f1 = 0
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
        self.targets = tf.placeholder(tf.int32, [None, self.num_step])
        self.targets_transition = tf.placeholder(tf.int32, [None])
        self.targets_weight = tf.placeholder(tf.float32, [None, self.num_step])

        self.embedding = tf.get_variable("embedding", shape=[self.vocabulary_size, self.emb_dim], dtype=tf.float32)

        self.new_embedding = tf.placeholder(tf.float32, shape=[None, self.emb_dim], name="new_embedding")
        self.update_embedding = tf.assign(self.embedding, self.new_embedding)

        self.inputs_emb = tf.nn.embedding_lookup(self.embedding, self.inputs)

        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dim, forget_bias=0.0, state_is_tuple=True)
        if self.keep_prob < 1:
            lstm_cell = tf.nn.rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob
            )

        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers, state_is_tuple=True)

        self._initial_state = cell.zero_state(self.batch_size, dtype=tf.float32)
        self.length = tf.reduce_sum(tf.sign(self.inputs), reduction_indices=1)
        self.length = tf.cast(self.length, tf.int32)
        out_put = []
        state = self._initial_state
        with tf.variable_scope("LSTM_layer"):
            for time_step in range(self.num_step):
                if time_step > 0:tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(self.inputs_emb[:, time_step, :], state)
                out_put.append(cell_output)
        self.outputs=out_put
        #self.outputs = tf.reduce_mean(out_put, 0)
        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs,1), [-1, self.hidden_dim])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b

        if not is_crf:
            pass
        else:
            self.tags_scores = tf.reshape(self.logits, [self.batch_size, self.num_step, self.num_classes])
            self.transitions = tf.get_variable("transitions", [self.num_classes + 1, self.num_classes + 1])

            dummy_val = -1000
            class_pad = tf.Variable(dummy_val*np.ones((self.batch_size, self.num_step, 1)),dtype=tf.float32)
            self.observations = tf.concat([self.tags_scores, class_pad],2)

            begin_vec = tf.Variable(np.array([[dummy_val] * self.num_classes + [0] for _ in range(self.batch_size)]),
                                    trainable=False, dtype=tf.float32)
            end_vec = tf.Variable(np.array([[0] + [dummy_val] * self.num_classes for _ in range(self.batch_size)]),
                                  trainable=False, dtype=tf.float32)
            begin_vec = tf.reshape(begin_vec, [self.batch_size, 1, self.num_classes + 1])
            end_vec = tf.reshape(end_vec, [self.batch_size, 1, self.num_classes + 1])

            self.observations = tf.concat([begin_vec, self.observations, end_vec],1)

            self.mask = tf.cast(tf.reshape(tf.sign(self.targets), [self.batch_size * self.num_step]), tf.float32)

            # point score
            self.point_score = tf.gather(tf.reshape(self.tags_scores, [-1]),
                                         tf.range(0, self.batch_size * self.num_step) * self.num_classes + tf.reshape(
                                             self.targets, [self.batch_size * self.num_step]))
            self.point_score *= self.mask

            # transition score
            self.trans_score = tf.gather(tf.reshape(self.transitions, [-1]), self.targets_transition)

            # real score
            self.target_path_score = tf.reduce_sum(self.point_score) + tf.reduce_sum(self.trans_score)

            # tf.initialize_all_variables()
            # sess = tf.Session()
            # sess.run(self.transitions.eval())

            # all path score
            self.total_path_score, self.max_scores, self.max_scores_pre = self.forward(self.observations,
                                                                                       self.transitions, self.length)

            # loss
            self.loss = - (self.target_path_score - self.total_path_score)

        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

#totalPath score?
    def logsumexp(self, x, axis=None):
        x_max = tf.reduce_max(x, reduction_indices=axis, keep_dims=True)
        x_max_ = tf.reduce_max(x, reduction_indices=axis)
        #防止数据溢出
        return x_max_ + tf.log(tf.reduce_sum(tf.exp(x - x_max), reduction_indices=axis))
#在条件随机场中用到的
    def forward(self, observations, transitions, length, is_viterbi=True, return_best_seq=True):
        length = tf.reshape(length, [self.batch_size])
        transitions = tf.reshape(tf.concat([transitions] * self.batch_size,0), [self.batch_size, self.num_classes+1, self.num_classes+1])
        observations = tf.reshape(observations, [self.batch_size, self.num_step + 2, self.num_classes+1, 1])
        observations = tf.transpose(observations, [1, 0, 2, 3])
        previous = observations[0, :, :, :]
        max_scores = []
        max_scores_pre = []
        alphas = [previous]
        for t in range(1, self.num_step + 2):
            previous = tf.reshape(previous, [self.batch_size, self.num_classes+1, 1])
            current = tf.reshape(observations[t, :, :, :], [self.batch_size, 1, self.num_classes+1])
            alpha_t = previous + current + transitions
            if is_viterbi:
                max_scores.append(tf.reduce_max(alpha_t, reduction_indices=1))
                max_scores_pre.append(tf.argmax(alpha_t, axis=1))
            alpha_t = tf.reshape(self.logsumexp(alpha_t, axis=1), [self.batch_size, self.num_classes+1, 1])
            alphas.append(alpha_t)
            previous = alpha_t

        alphas = tf.reshape(tf.concat( alphas,0), [self.num_step + 2, self.batch_size, self.num_classes+1, 1])
        alphas = tf.transpose(alphas, [1, 0, 2, 3])
        alphas = tf.reshape(alphas, [self.batch_size * (self.num_step + 2), self.num_classes+1, 1])

        last_alphas = tf.gather(alphas, tf.range(0, self.batch_size) * (self.num_step + 2) + length)
        last_alphas = tf.reshape(last_alphas, [self.batch_size, self.num_classes+1, 1])

        max_scores = tf.reshape(tf.concat(max_scores,0), (self.num_step + 1, self.batch_size, self.num_classes+1))
        max_scores_pre = tf.reshape(tf.concat(max_scores_pre,0), (self.num_step + 1, self.batch_size, self.num_classes+1))
        max_scores = tf.transpose(max_scores, [1, 0, 2])
        max_scores_pre = tf.transpose(max_scores_pre, [1, 0, 2])

        return tf.reduce_sum(self.logsumexp(last_alphas, axis=1)), max_scores, max_scores_pre

    def train(self, sess, save_file, X_train, y_train, X_val, y_val,matchListB,matchListE):
        dataPath = '/root/PycharmProjects/NER/bi_LSTM_CRF_test/data/'
        saver = tf.train.Saver()
        label2id, id2label = data_helper.loadMap("tagDic.pkl")
        labelidsB = [label2id[item] for item in matchListB]
        labelidsE = [label2id[item] for item in matchListE]

        #merged = tf.merge_all_summaries()
        summary_writer_train = tf.summary.FileWriter('loss_log/train_loss', sess.graph)
        summary_writer_val = tf.summary.FileWriter('loss_log/val_loss', sess.graph)

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
                proweight =[]
                for item in y_train_batch:
                    temp =[(k in labelidsE or k in labelidsB) for k in item ]
                    proweight.append(temp)

                y_train_weight_batch = 1 + np.array(proweight,float)
                transition_batch = data_helper.getTransition(y_train_batch,self.num_classes)

                _, loss_train, max_scores, max_scores_pre, length, train_summary = \
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.max_scores,
                        self.max_scores_pre,
                        self.length,
                        self.train_summary
                    ],
                        feed_dict={
                            self.targets_transition: transition_batch,
                            self.inputs: X_train_batch,
                            self.targets: y_train_batch,
                            self.targets_weight: y_train_weight_batch,
                            self.is_training:1

                        })

                predicts_train = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                if iteration % 10 == 0:
                    cnt += 1
                    #precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch,
                    #                                                        predicts_train,label2id)
                    precision_train, recall_train, f1_train = self.evaluateNew(X_train_batch, y_train_batch,
                                                                            predicts_train, label2id)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print( "iteration: %5d, train loss: %5d, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (
                        iteration, loss_train, precision_train, recall_train, f1_train))

                # validation
                if iteration % 10 == 0:
                    X_val_batch, y_val_batch = data_helper.nextRandomBatch(X_val, y_val, batch_size=self.batch_size)
                    proweight = []
                    for item in y_val_batch:
                        temp = [(k in labelidsE or k in labelidsB) for k in item]
                        proweight.append(temp)

                    y_val_weight_batch = 1 + np.array(proweight, float)


                    transition_batch = data_helper.getTransition(y_val_batch,self.num_classes)

                    loss_val, max_scores, max_scores_pre, length, val_summary = \
                        sess.run([
                            self.loss,
                            self.max_scores,
                            self.max_scores_pre,
                            self.length,
                            self.val_summary
                        ],
                            feed_dict={
                                self.targets_transition: transition_batch,
                                self.inputs: X_val_batch,
                                self.targets: y_val_batch,
                                self.targets_weight: y_val_weight_batch,
                                self.is_training:0

                            })

                    predicts_val = self.viterbi(max_scores, max_scores_pre, length, predict_size=self.batch_size)
                    #precision_val, recall_val, f1_val = self.evaluate(X_val_batch, y_val_batch, predicts_val,label2id)
                    precision_val, recall_val, f1_val = self.evaluateNew(X_val_batch, y_val_batch, predicts_val, label2id)
                    summary_writer_val.add_summary(val_summary, cnt)
                    print(
                        "iteration: %5d, valid loss: %5d, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (
                        iteration, loss_val, precision_val, recall_val, f1_val))

                    if f1_val > self.max_f1:
                        self.recall = recall_val
                        self.precesion = precision_val
                        self.max_f1 = f1_val
                        save_path = saver.save(sess, save_file)
                        print("saved the best model with f1: %.5f" % (self.max_f1))

    def test(self, sess, X_test, X_test_str, output_path):
        label2id, id2label = data_helper.loadMap("label2id")
        num_iterations = int(math.ceil(1.0 * len(X_test) / self.batch_size))
        print("number of iteration: " + str(num_iterations))
        with open(output_path, "wb") as outfile:
            for i in range(num_iterations):
                print("iteration: " + str(i + 1))
                results = []
                X_test_batch = X_test[i * self.batch_size: (i + 1) * self.batch_size]
                X_test_str_batch = X_test_str[i * self.batch_size: (i + 1) * self.batch_size]
                if i == num_iterations - 1 and len(X_test_batch) < self.batch_size:
                    X_test_batch = list(X_test_batch)
                    X_test_str_batch = list(X_test_str_batch)
                    last_size = len(X_test_batch)
                    X_test_batch += [[0 for j in range(self.num_step)] for i in range(self.batch_size - last_size)]
                    X_test_str_batch += [['x' for j in range(self.num_step)] for i in
                                         range(self.batch_size - last_size)]
                    X_test_batch = np.array(X_test_batch)
                    X_test_str_batch = np.array(X_test_str_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)
                    results = results[:last_size]
                else:
                    X_test_batch = np.array(X_test_batch)
                    results = self.predictBatch(sess, X_test_batch, X_test_str_batch, id2label)

                for i in range(len(results)):
                    doc = ''.join(X_test_str_batch[i])
                    outfile.write(doc + "<@>" + results[i] + "\n")
#找到预测的最佳标记序列
    def viterbi(self, max_scores, max_scores_pre, length, predict_size=128):
        best_paths = []
        for m in range(predict_size):
            path = []
            last_max_node = np.argmax(max_scores[m][length[m]])
            # last_max_node = 0
            for t in range(1, length[m] + 1)[::-1]:
                last_max_node = max_scores_pre[m][t][last_max_node]
                path.append(last_max_node)
            path = path[::-1]
            best_paths.append(path)
        return best_paths
#test集做预测，并找到entity返回
    def predictBatch(self, sess, X, X_str, id2label):
        results = []
        length, max_scores, max_scores_pre = sess.run([self.length, self.max_scores, self.max_scores_pre],
                                                      feed_dict={self.inputs: X})
        predicts = self.viterbi(max_scores, max_scores_pre, length, self.batch_size)
        for i in range(len(predicts)):
            x = ''.join(X_str[i]).decode("utf-8")
            y_pred = ' '.join([id2label[val] for val in predicts[i] if val != 5 and val != 0])

            results.append(y_pred)
        return results
#calculate the precision,recall and f1
    def evaluate(self, X, y_true, y_pred,lable2id):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        hit_num = 0
        pred_num = 0
        true_num = 0


        for i in range(len(y_true)):
            for j in range(len(y_true[i])):
                if y_true[i][j]==y_pred[i][j] :
                   hit_num+=1
                elif y_true[i][j] == lable2id['pad'] and y_pred[i][j]==lable2id['O']:
                    hit_num+=1



            pred_num += len(y_pred[i])
            true_num += len(y_true[i])
        if pred_num != 0:
            precision = 1.0 * hit_num / pred_num
        if true_num != 0:
            recall = 1.0 * hit_num / true_num
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1

    # calculate the precision,recall and f1,O and pad are negtive
    def evaluateNew(self, X, y_true, y_pred, lable2id):
        precision = -1.0
        recall = -1.0
        f1 = -1.0
        TP = 0#预测为正，实际为正
        TN = 0#预测为负，实际为负
        FP = 0#被模型预测为正的负样本；可以称作误报率
        FN = 0#被模型预测为负的正样本；可以称作漏报率

        for i in range(len(y_true)):
            for j in range(len(y_pred[i])):
                if y_true[i][j] == y_pred[i][j] and (y_pred[i][j] != lable2id['O'] and y_pred[i][j] != lable2id['pad']):
                    TP+=1
                elif (y_pred[i][j] == lable2id['O'] or y_pred[i][j] == lable2id['pad']) and (y_true[i][j] == lable2id['O'] or y_true[i][j] == lable2id['pad']):
                    TN+=1
                elif (y_true[i][j]!=y_pred[i][j]) and (y_pred[i][j]!=lable2id['O'] and y_pred[i][j]!=lable2id['pad'] ):
                    FP+=1
                elif (y_true[i][j]!=lable2id['O'] and y_true[i][j]!=lable2id['pad'])and y_pred[i][j]!=y_true[i][j]  :
                    FN+=1

        if TP+FP != 0:
            precision = 1.0 * TP / (TP+FP)#所有标了非Otag里标对的百分比
        if TP+FN != 0:
            recall = 1.0 * TP / (TP+FN)#所有应该标了非Otag里标对的百分比
        if precision > 0 and recall > 0:
            f1 = 2.0 * (precision * recall) / (precision + recall)
        return precision, recall, f1
    def assign_embedding(self, session, new_embedding):

        session.run(self.update_embedding, feed_dict={self.new_embedding: new_embedding})

    def get_embedding(self):
        return self.embedding