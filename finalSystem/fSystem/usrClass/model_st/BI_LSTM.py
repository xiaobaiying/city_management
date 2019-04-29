import math
import sys
sys.path.append("..")
import data_helper
import numpy as np
import tensorflow as tf


class BI_LSTM_Tagging(object):
    def __init__(self, config,is_crf = True):
        # Parameter
        self.keep_prob = config.keep_prob
        self.is_training = tf.placeholder(tf.int32, name="is_training")
        self.batch_size = config.batch_size

        self.max_f1 = 0
        self.precesion=0
        self.recall=0
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

        # softmax
        self.outputs = tf.reshape(tf.concat(self.outputs,1), [-1, self.hidden_dim * 2])
        self.softmax_w = tf.get_variable("softmax_w", [self.hidden_dim * 2, self.num_classes])
        self.softmax_b = tf.get_variable("softmax_b", [self.num_classes])
        self.logits = tf.matmul(self.outputs, self.softmax_w) + self.softmax_b
        self.logits = tf.reshape(self.logits, [self.batch_size, self.num_step, self.num_classes])
        self.prediction = tf.argmax(self.logits, 2)
        cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logits+1e-10,labels=self.targets))
        self.loss=cost
        # summary
        self.train_summary = tf.summary.scalar("loss", self.loss)
        self.val_summary = tf.summary.scalar("loss", self.loss)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


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

                _, loss_train, prediction_val,length, train_summary = \
                    sess.run([
                        self.optimizer,
                        self.loss,
                        self.prediction,
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


                if iteration % 10 == 0:
                    cnt += 1
                    #precision_train, recall_train, f1_train = self.evaluate(X_train_batch, y_train_batch,
                    #                                                        predicts_train,label2id)

                    precision_train, recall_train, f1_train = self.evaluateNew(X_train_batch, y_train_batch,
                                                                            prediction_val, label2id)
                    summary_writer_train.add_summary(train_summary, cnt)
                    print( "iteration: %5d, train loss: %5f, train precision: %.5f, train recall: %.5f, train f1: %.5f" % (
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

                    loss_val,  prediction_val,length, val_summary = \
                        sess.run([
                            self.loss,
                            self.prediction,
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

                    #precision_val, recall_val, f1_val = self.evaluate(X_val_batch, y_val_batch, predicts_val,label2id)
                    precision_val, recall_val, f1_val = self.evaluateNew(X_val_batch, y_val_batch, prediction_val, label2id)
                    summary_writer_val.add_summary(val_summary, cnt)
                    print(
                        "iteration: %5d, valid loss: %5f, valid precision: %.5f, valid recall: %.5f, valid f1: %.5f" % (
                        iteration, loss_val, precision_val, recall_val, f1_val))

                    if f1_val > self.max_f1:

                        self.max_f1 = f1_val
                        self.recall=recall_val
                        self.precesion=precision_val
                        #save_path = saver.save(sess, save_file)
                        print("saved the best model with f1: %.5f" % (self.max_f1))



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